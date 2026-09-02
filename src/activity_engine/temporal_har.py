"""Temporal Human Activity Recognition sliding-window classifier.

Extracts structured 154-dimensional feature vectors per frame, buffers a rolling 60-frame
temporal window, executes the Conv1D-BiGRU-Attention neural network classifier, and applies
confidence gating with multi-frame debouncing/smoothing.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..core.interfaces import TemporalHARInterface
from ..core.models import FramePerceptionResult
from .feature_extractor import PerceptionFeatureExtractor, FEATURE_DIM
from .temporal_buffer import TemporalFeatureBuffer
from .har_model import Conv1DBiGRUHAR, HAR_ACTION_CLASSES, NUM_HAR_CLASSES


class PredictionSmoother:
    """Temporal prediction smoother and debouncer for sliding-window HAR inference.

    Features:
      1. Moving average of multi-class softmax probability distributions over K frames.
      2. Confidence thresholding: flags predictions below threshold as UNCERTAIN.
      3. State debouncing: requires persistent agreement before switching active output action.
    """

    def __init__(
        self,
        window_size: int = 5,
        confidence_threshold: float = 0.65,
        min_consecutive_switch: int = 2,
    ):
        self.window_size = max(1, window_size)
        self.confidence_threshold = confidence_threshold
        self.min_consecutive_switch = min_consecutive_switch

        self.prob_history: deque = deque(maxlen=self.window_size)
        self.raw_history: deque = deque(maxlen=self.window_size)

        self.current_stable_class_id: int = 0
        self.current_stable_action: str = HAR_ACTION_CLASSES.get(0, "ACTION_IDLE")
        self.consecutive_candidate_count: int = 0
        self.last_candidate_id: int = 0

    def update(self, probs: np.ndarray) -> Tuple[str, float, str, float, bool]:
        """Update smoother with new frame probability distribution (shape: (num_classes,)).

        Returns:
            (stable_action_name, smoothed_confidence, raw_action_name, raw_confidence, is_confident)
        """
        # 1. Raw top prediction
        raw_id = int(np.argmax(probs))
        raw_conf = float(probs[raw_id])
        raw_action = HAR_ACTION_CLASSES.get(raw_id, f"ACTION_{raw_id}")

        self.prob_history.append(probs)
        self.raw_history.append(raw_id)

        # 2. Moving average probability across recent window
        avg_probs = np.mean(np.array(self.prob_history), axis=0)
        candidate_id = int(np.argmax(avg_probs))
        smoothed_conf = float(avg_probs[candidate_id])
        candidate_action = HAR_ACTION_CLASSES.get(candidate_id, f"ACTION_{candidate_id}")

        # 3. Confidence Gating
        is_confident = smoothed_conf >= self.confidence_threshold

        # 4. State transition debouncing
        if candidate_id == self.current_stable_class_id:
            self.consecutive_candidate_count = 0
        else:
            if candidate_id == self.last_candidate_id:
                self.consecutive_candidate_count += 1
            else:
                self.consecutive_candidate_count = 1
                self.last_candidate_id = candidate_id

            # Switch state if candidate is consistently observed
            if self.consecutive_candidate_count >= self.min_consecutive_switch:
                self.current_stable_class_id = candidate_id
                self.current_stable_action = candidate_action
                self.consecutive_candidate_count = 0

        # Output label: if not confident, provide informative UNCERTAIN flag
        if is_confident:
            output_action = self.current_stable_action
        else:
            output_action = f"UNCERTAIN ({candidate_action})"

        return output_action, smoothed_conf, raw_action, raw_conf, is_confident

    def reset(self) -> None:
        """Reset smoother histories."""
        self.prob_history.clear()
        self.raw_history.clear()
        self.current_stable_class_id = 0
        self.current_stable_action = HAR_ACTION_CLASSES.get(0, "ACTION_IDLE")
        self.consecutive_candidate_count = 0
        self.last_candidate_id = 0


class TemporalHARClassifier(TemporalHARInterface):
    """Sliding-window temporal action classifier powered by Conv1D-BiGRU with temporal smoothing."""

    def __init__(
        self,
        window_size: int = 60,
        feature_dim: int = FEATURE_DIM,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[Conv1DBiGRUHAR] = None,
        confidence_threshold: float = 0.65,
        smoothing_window: int = 5,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        # Device selection: prefer CUDA if available
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.extractor = PerceptionFeatureExtractor(feature_dim=feature_dim)
        self.buffer = TemporalFeatureBuffer(window_size=window_size, feature_dim=feature_dim)
        self.smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )

        self.model: Optional[Conv1DBiGRUHAR] = None
        self.model_loaded: bool = False

        # Live telemetry state
        self.last_action_name: str = "ACTION_IDLE"
        self.last_confidence: float = 1.0
        self.last_raw_action: str = "ACTION_IDLE"
        self.last_raw_confidence: float = 1.0
        self.last_is_confident: bool = True
        self.last_latency_ms: float = 0.0

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
            self.model_loaded = True
        elif checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.exists():
                self.load_checkpoint(ckpt_p)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load trained model weights from checkpoint file."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(str(ckpt_p), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        config = ckpt.get("config", {})

        model = Conv1DBiGRUHAR(
            input_dim=config.get("input_dim", self.feature_dim),
            seq_len=config.get("seq_len", self.window_size),
            conv_channels=config.get("conv_channels", 128),
            gru_hidden_dim=config.get("gru_hidden_dim", 64),
            num_classes=config.get("num_classes", NUM_HAR_CLASSES),
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self.model_loaded = True

    def update_and_classify(self, perception: FramePerceptionResult) -> Tuple[str, float]:
        """Ingest frame perception, update temporal window, and return (action_name, confidence)."""
        fv = self.extractor.extract(perception)
        self.buffer.push(fv)

        if self.model is None or not self.model_loaded:
            # Baseline fallback when no model checkpoint is loaded
            self.last_action_name = "ACTION_IDLE"
            self.last_confidence = 1.0
            self.last_raw_action = "ACTION_IDLE"
            self.last_raw_confidence = 1.0
            self.last_is_confident = True
            self.last_latency_ms = 0.0
            return self.last_action_name, self.last_confidence

        # Extract rolling 60-frame window
        window_np = self.buffer.get_window()  # Shape (60, 154)

        t0 = time.perf_counter()
        with torch.inference_mode():
            window_tensor = torch.as_tensor(window_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 60, 154)
            logits = self.model(window_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Apply smoothing and confidence thresholding
        action_name, conf, raw_action, raw_conf, is_conf = self.smoother.update(probs)

        self.last_action_name = action_name
        self.last_confidence = conf
        self.last_raw_action = raw_action
        self.last_raw_confidence = raw_conf
        self.last_is_confident = is_conf

        return self.last_action_name, self.last_confidence

    def get_telemetry(self) -> Dict[str, object]:
        """Return structured diagnostics for visualization and logging."""
        return {
            "action_name": self.last_action_name,
            "confidence": round(self.last_confidence, 4),
            "raw_action": self.last_raw_action,
            "raw_confidence": round(self.last_raw_confidence, 4),
            "is_confident": self.last_is_confident,
            "latency_ms": round(self.last_latency_ms, 3),
            "buffer_fill": self.buffer.current_length(),
            "buffer_capacity": self.window_size,
            "is_full": self.buffer.is_full(),
            "device": self.device.upper(),
            "model_loaded": self.model_loaded,
        }

    def reset_buffer(self) -> None:
        """Clear temporal feature buffer and prediction smoother."""
        self.buffer.reset()
        self.smoother.reset()
