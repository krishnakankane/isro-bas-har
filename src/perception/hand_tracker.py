"""Hand Tracker implementation using Google MediaPipe Tasks HandLandmarker.

Extracts 21 3D landmarks per hand and detects grasp/manipulation pinch gestures.
"""

import time
import math
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2

from ..core.interfaces import HandTrackerInterface
from ..core.models import HandTrackingResult, HandLandmark


class HandTracker(HandTrackerInterface):
    """MediaPipe Tasks 21-Landmark Dual Hand Landmarker."""

    def __init__(
        self,
        model_path: str = "models/mediapipe/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize Hand Tracker using MediaPipe Tasks Vision API.

        Args:
            model_path: Path to hand landmarker .task bundle.
            max_num_hands: Maximum hands to detect (default: 2).
            min_detection_confidence: Detection threshold.
            min_tracking_confidence: Tracking threshold.
        """
        self.model_path = model_path
        self.max_num_hands = max_num_hands
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence

        self._detector = None
        self.last_latency_ms: float = 0.0
        self._init_mediapipe()

    def _resolve_model_path(self, path_str: str) -> Path:
        """Resolve model path against working directory and project root."""
        candidate = Path(path_str)
        if candidate.is_file():
            return candidate

        # Check relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate_project = project_root / path_str
        if candidate_project.is_file():
            return candidate_project

        # Check in models/mediapipe/
        candidate_mediapipe = project_root / "models" / "mediapipe" / candidate.name
        if candidate_mediapipe.is_file():
            return candidate_mediapipe

        # Check in models/
        candidate_models = project_root / "models" / candidate.name
        if candidate_models.is_file():
            return candidate_models

        return candidate

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Tasks Vision HandLandmarker."""
        resolved_path = self._resolve_model_path(self.model_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"[HandTracker] Required model asset not found at '{self.model_path}' "
                f"(resolved: '{resolved_path}'). Ensure task models are present in 'models/mediapipe/'."
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(resolved_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            print(f"[HandTracker] MediaPipe Tasks HandLandmarker initialized successfully ({resolved_path.name}).")
        except Exception as e:
            raise RuntimeError(f"[HandTracker] Failed to initialize MediaPipe Tasks HandLandmarker: {e}") from e

    def track(self, frame: np.ndarray) -> Tuple[Optional[HandTrackingResult], Optional[HandTrackingResult]]:
        """Track left and right hand landmarks from BGR video frame.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            Tuple of (left_hand_result, right_hand_result).
        """
        if frame is None or frame.size == 0 or self._detector is None:
            self.last_latency_ms = 0.0
            return None, None

        start_t = time.perf_counter()

        left_result: Optional[HandTrackingResult] = None
        right_result: Optional[HandTrackingResult] = None

        try:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = self._detector.detect(mp_image)

            if results.hand_landmarks and results.handedness:
                for hand_lms, handedness_list in zip(results.hand_landmarks, results.handedness):
                    label = handedness_list[0].display_name or handedness_list[0].category_name or "Right"

                    landmarks: List[HandLandmark] = []
                    for idx, lm in enumerate(hand_lms):
                        landmarks.append(
                            HandLandmark(
                                id=idx,
                                x=float(lm.x),
                                y=float(lm.y),
                                z=float(lm.z),
                            )
                        )

                    wrist_lm = landmarks[0]
                    wrist_pos = (wrist_lm.x, wrist_lm.y)

                    # Simple grasp pinch heuristic: distance between Thumb Tip (4) and Index Tip (8)
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    pinch_dist = math.sqrt(
                        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.z - index_tip.z) ** 2
                    )
                    is_grasping = (pinch_dist < 0.08)

                    hand_res = HandTrackingResult(
                        handedness=label,
                        landmarks=landmarks,
                        wrist_pos=wrist_pos,
                        is_grasping=is_grasping,
                        is_detected=True,
                    )

                    if label == "Left":
                        left_result = hand_res
                    else:
                        right_result = hand_res

            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return left_result, right_result

        except Exception as e:
            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            raise RuntimeError(f"[HandTracker] Inference error during hand tracking: {e}") from e

    def close(self) -> None:
        """Release detector resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
