"""ISRO BAS-HAR — Operator UI Backend Server.

FastAPI application providing live video streaming (MJPEG), WebSocket telemetry streaming,
and REST control APIs around the unified perception, temporal HAR, and deterministic
protocol state machine.

Usage
-----
    python -m src.ui.server --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── System Imports ───────────────────────────────────────────────────────────
from config.settings import (
    BASE_DIR,
    CONFIG_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
)
from src.activity_engine.feature_extractor import PerceptionFeatureExtractor
from src.activity_engine.har_model import Conv1DBiGRUHAR, HAR_ACTION_CLASSES, NUM_HAR_CLASSES
from src.activity_engine.temporal_buffer import TemporalFeatureBuffer
from src.activity_engine.temporal_har import PredictionSmoother, TemporalHARClassifier
from src.core.models import FramePerceptionResult, InteractionState
from src.core.state_machine import AnomalyType, ProtocolStateMachine
from src.perception.hand_tracker import HandTracker
from src.perception.hoi_detector import HOIDetector, HOIConfig
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.visualization.perception_visualizer import PerceptionVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ISRO-UI")

# ─── Pipeline Worker Thread ───────────────────────────────────────────────────
class PipelineWorker:
    """Manages continuous video capture and inference execution in a background thread."""

    def __init__(self):
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self.source = "0"
        self.mode = "webcam"  # 'webcam' or 'video'
        self.cap: Optional[cv2.VideoCapture] = None

        # Lifecycle & Mission State
        self.lifecycle_state = "PROTOCOL ARMED"
        self.mission_start_time: Optional[float] = None
        self.mission_duration_sec: float = 0.0
        self.max_anomaly_score: float = 0.0
        self.max_protocol_anomaly_score: float = 0.0
        self._injected_anomaly: Optional[str] = None
        self.error_message: Optional[str] = None

        # Thread-safe Frame Buffers
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_raw_frame_id: int = 0
        self.capture_fps: float = 0.0
        self.display_fps: float = 0.0

        # Telemetry State
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.latest_frame_id: int = 0
        self.latest_metrics: Dict[str, Any] = self._default_metrics()
        self.event_log: List[Dict[str, Any]] = []
        self.anomaly_log: List[Dict[str, Any]] = []
        self.uncertain_log: List[Dict[str, Any]] = []

        # Running History & Averages
        self.confidence_history: List[float] = []
        self.fps_history: List[float] = []
        self.latency_history: List[float] = []
        self.total_frames_processed: int = 0

        # Core Components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = ObjectDetector(model_name_or_path="yolov8n.pt", conf_thresh=0.45, device=self.device)
        self.pose_estimator = PoseEstimator(model_path="models/mediapipe/pose_landmarker_lite.task")
        self.hand_tracker = HandTracker(model_path="models/mediapipe/hand_landmarker.task")
        self.hoi_detector = HOIDetector(config=HOIConfig())
        self.visualizer = PerceptionVisualizer()

        # HAR Classifier & Protocol State Machine
        self.har_checkpoint_path = MODELS_DIR / "har" / "conv1d_bigru_best.pt"
        self.har_classifier = TemporalHARClassifier(
            checkpoint_path=str(self.har_checkpoint_path) if self.har_checkpoint_path.exists() else None,
            window_size=60,
            feature_dim=154,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            device=self.device,
        )

        self.protocol_path = CONFIG_DIR / "experiment_protocols.json"
        self.state_machine = ProtocolStateMachine(
            protocol_json_path=self.protocol_path,
            min_dwell_frames=5,
            min_confidence=DEFAULT_CONFIDENCE_THRESHOLD,
            log_dir=BASE_DIR / "data" / "runtime" / "protocol_events",
            session_id="UI_SESSION",
        )

        # Warmup perception and ML models to eliminate first-frame latency
        try:
            dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with torch.inference_mode():
                self.detector.detect(dummy_frame)
                self.pose_estimator.estimate(dummy_frame)
                self.hand_tracker.track(dummy_frame)
            self.lifecycle_state = "PROTOCOL ARMED"
        except Exception as e:
            logger.warning(f"Worker warmup skipped: {e}")
            self.lifecycle_state = "PROTOCOL ARMED"

    def get_readiness(self) -> Dict[str, Any]:
        """Return system readiness checklist state."""
        chk = {
            "camera": (self.cap is not None and self.cap.isOpened()) if self.running else True,
            "object_detector": self.detector.model is not None,
            "pose_estimator": self.pose_estimator._detector is not None,
            "hand_tracker": self.hand_tracker._detector is not None,
            "hoi_analysis": True,
            "feature_extractor": True,
            "har_model": self.har_classifier.model_loaded,
            "protocol_fsm": self.state_machine.protocol is not None,
        }
        return {
            "lifecycle_state": self.lifecycle_state,
            "checklist": chk,
            "all_ready": all(chk.values()),
        }

    def inject_anomaly(self, scenario: str) -> Dict[str, Any]:
        """Queue a controlled software-injected anomaly scenario for evaluation."""
        valid_scenarios = [
            "OUT_OF_ORDER_ACTION",
            "UNEXPECTED_ACTION",
            "REPEATED_ACTION",
            "LOW_CONFIDENCE",
            "TIMEOUT_EXCEEDED",
        ]
        if scenario not in valid_scenarios:
            raise ValueError(f"Unknown anomaly scenario: {scenario}. Must be one of {valid_scenarios}")

        self._injected_anomaly = scenario
        return {"status": "SUCCESS", "injected_scenario": scenario, "mode": "SIMULATED / INJECTED"}

    def _default_metrics(self) -> Dict[str, Any]:
        return {
            "status": "OFFLINE",
            "lifecycle_state": "PROTOCOL ARMED",
            "fps_actual": 0.0,
            "pipeline_fps": 0.0,
            "capture_fps": 0.0,
            "display_fps": 0.0,
            "pipeline_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "detector_ms": 0.0,
            "pose_ms": 0.0,
            "hand_ms": 0.0,
            "hoi_ms": 0.0,
            "har_latency_ms": 0.0,
            "device": self.device.upper() if hasattr(self, "device") else "CPU",
            "har_action": "ACTION_IDLE",
            "har_confidence": 1.0,
            "har_is_confident": True,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "protocol_id": "HOME_DEMO_PROTOCOL_01",
            "current_step_id": 1,
            "total_steps": 10,
            "current_step_name": "IDLE",
            "expected_action": "ACTION_IDLE",
            "observed_action": "ACTION_IDLE",
            "fsm_status": "WAITING",
            "fsm_valid": True,
            "anomaly_score": 0.0,
            "is_completed": False,
            "completed_steps": [],
            "step_timestamps": {},
            "source_type": "NONE",
            "source_path": "",
            "buffer_fill": 0,
            "buffer_cap": 60,
            "error_message": None,
            "mission_summary": {
                "protocol_id": "HOME_DEMO_PROTOCOL_01",
                "session_id": "SES_001_NOMINAL",
                "lifecycle_state": "PROTOCOL ARMED",
                "completed_steps": 0,
                "remaining_steps": 10,
                "total_steps": 10,
                "completion_pct": 0.0,
                "mission_duration": "00:00.0s",
                "har_uncertain_count": 0,
                "protocol_anomaly_count": 0,
                "anomaly_count": 0,
                "max_protocol_anomaly_score": 0.0,
                "max_anomaly_score": 0.0,
                "avg_confidence": 100.0,
                "current_confidence": 100.0,
                "current_action": "ACTION_IDLE",
                "expected_action": "ACTION_IDLE",
                "last_anomaly_reason": "NONE",
                "avg_fps": 0.0,
                "avg_pipeline_latency_ms": 0.0,
                "mission_status": "STANDBY",
            },
        }

    def start(self, source: str = "0", mode: str = "webcam") -> None:
        """Start decoupled capture and inference pipeline."""
        with self.lock:
            if self.running:
                self.stop()

            self.source = source
            self.mode = mode
            self._stop_event.clear()
            self.running = True
            self.lifecycle_state = "CAMERA CHECK"
            self.error_message = None

            # Open Video Capture
            src_val = int(source) if source.isdigit() else str(source)
            self.cap = cv2.VideoCapture(src_val)
            if not self.cap or not self.cap.isOpened():
                self.running = False
                self.lifecycle_state = "SYSTEM ERROR"
                self.error_message = f"Cannot open video source: {source}"
                raise RuntimeError(self.error_message)

            # Configure resolution
            if mode == "webcam":
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.mission_start_time = time.time()
            self.lifecycle_state = "MISSION ACTIVE"

            # Start Decoupled Capture & Inference Threads
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

            self.capture_thread.start()
            self.inference_thread.start()
            logger.info(f"Decoupled pipeline started on source: {source} (mode: {mode})")

    def stop(self) -> None:
        """Stop running capture and inference threads."""
        self._stop_event.set()
        with self.lock:
            self.running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)

        self.capture_thread = None
        self.inference_thread = None
        self.lifecycle_state = "MISSION STOPPED"
        self.latest_metrics["status"] = "OFFLINE"
        self.latest_metrics["lifecycle_state"] = "MISSION STOPPED"
        logger.info("Decoupled pipeline stopped.")

    def reset_session(self) -> None:
        """Reset state machine and telemetry buffers."""
        with self.lock:
            self.state_machine.reset()
            self.har_classifier.reset_buffer()
            self.event_log.clear()
            self.anomaly_log.clear()
            self.uncertain_log.clear()
            self.confidence_history.clear()
            self.fps_history.clear()
            self.latency_history.clear()
            self.total_frames_processed = 0
            self.mission_start_time = time.time() if self.running else None
            self.max_anomaly_score = 0.0
            self.max_protocol_anomaly_score = 0.0
            self._injected_anomaly = None
            self.lifecycle_state = "MISSION ACTIVE" if self.running else "PROTOCOL ARMED"
            logger.info("Protocol session reset.")

    def _capture_loop(self) -> None:
        """Dedicated high-frequency video capture thread."""
        cap_tracker: List[float] = []
        raw_idx = 0

        while self.running and not self._stop_event.is_set() and self.cap and self.cap.isOpened():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.mode == "video":
                    # Loop video seamlessly for continuous demonstration
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.005)
                    continue

            raw_idx += 1
            dt = time.perf_counter() - t0
            cap_tracker.append(1.0 / max(dt, 1e-4))
            if len(cap_tracker) > 30:
                cap_tracker.pop(0)
            self.capture_fps = float(np.mean(cap_tracker))

            with self._frame_lock:
                self._latest_raw_frame = frame
                self._latest_raw_frame_id = raw_idx

            # In video mode, pace capture to ~30 FPS to avoid speeding ahead
            if self.mode == "video":
                time.sleep(1.0 / 32.0)

    def _inference_loop(self) -> None:
        """Dedicated inference, temporal HAR, state machine, and HUD rendering thread."""
        fps_tracker: List[float] = []
        last_processed_id = -1
        proc_idx = 0

        while self.running and not self._stop_event.is_set():
            frame = None
            frame_id = -1

            with self._frame_lock:
                if self._latest_raw_frame is not None and self._latest_raw_frame_id != last_processed_id:
                    frame = self._latest_raw_frame.copy()
                    frame_id = self._latest_raw_frame_id
                    last_processed_id = frame_id

            if frame is None:
                time.sleep(0.002)
                continue

            proc_idx += 1
            self.total_frames_processed += 1
            t0 = time.perf_counter()
            h, w = frame.shape[:2]

            # ── 1. Perception Stack ──────────────────────────────────────────
            t_pipe_start = time.perf_counter()

            with torch.inference_mode():
                t_det = time.perf_counter()
                objects = self.detector.detect(frame)
                det_ms = (time.perf_counter() - t_det) * 1000.0

                t_pose = time.perf_counter()
                pose_res = self.pose_estimator.estimate(frame)
                pose_ms = (time.perf_counter() - t_pose) * 1000.0

                t_hand = time.perf_counter()
                left_hand, right_hand = self.hand_tracker.track(frame)
                hand_ms = (time.perf_counter() - t_hand) * 1000.0

                t_hoi = time.perf_counter()
                interactions = self.hoi_detector.analyze_interaction(objects, left_hand, right_hand)
                hoi_ms = (time.perf_counter() - t_hoi) * 1000.0

                # ── 2. Temporal HAR Model Inference ──────────────────────────────
                frame_res = FramePerceptionResult(
                    frame_id=proc_idx,
                    timestamp_sec=time.time(),
                    objects=objects,
                    body_pose=pose_res,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    interactions=interactions,
                )

                t_har = time.perf_counter()
                action_name, confidence = self.har_classifier.update_and_classify(frame_res)
                har_telem = self.har_classifier.get_telemetry()
                har_ms = (time.perf_counter() - t_har) * 1000.0

            # ── 3. Deterministic Protocol State Machine ──────────────────────
            eval_action = action_name
            eval_conf = confidence
            eval_ts = float(proc_idx) / 30.0
            is_injected = False
            injected_tag = ""

            if self._injected_anomaly is not None:
                is_injected = True
                scenario = self._injected_anomaly
                injected_tag = f" [SIMULATED / INJECTED: {scenario}]"
                if scenario == "OUT_OF_ORDER_ACTION":
                    eval_action = "ACTION_CLOSE_BOX"
                    eval_conf = 0.95
                elif scenario == "UNEXPECTED_ACTION":
                    eval_action = "ACTION_UNKNOWN_OBJECT"
                    eval_conf = 0.90
                elif scenario == "REPEATED_ACTION":
                    eval_action = "ACTION_IDLE"
                    eval_conf = 0.95
                elif scenario == "LOW_CONFIDENCE":
                    eval_conf = 0.40
                elif scenario == "TIMEOUT_EXCEEDED":
                    eval_ts = 300.0  # Force timeout
                self._injected_anomaly = None

            val_res = self.state_machine.evaluate_action(
                action_name=eval_action,
                confidence=eval_conf,
                timestamp=eval_ts,
                frame=proc_idx,
            )

            if val_res.is_completed:
                self.lifecycle_state = "MISSION COMPLETE"

            is_protocol_anomaly = (not val_res.is_valid and val_res.status_text != "UNCERTAIN" and val_res.anomaly_type != AnomalyType.LOW_CONFIDENCE) or is_injected
            is_uncertain_event = (val_res.status_text == "UNCERTAIN" or val_res.anomaly_type == AnomalyType.LOW_CONFIDENCE)

            if is_protocol_anomaly:
                self.max_protocol_anomaly_score = max(self.max_protocol_anomaly_score, val_res.anomaly_score)

            self.max_anomaly_score = max(self.max_anomaly_score, val_res.anomaly_score)

            pipeline_elapsed = time.perf_counter() - t_pipe_start
            pipeline_latency_ms = pipeline_elapsed * 1000.0
            pipeline_fps = 1000.0 / max(pipeline_latency_ms, 1e-4)

            # Record transition / anomaly / uncertainty events for UI
            if eval_action != getattr(self, "_last_ui_action", "") or is_injected or not val_res.is_valid:
                self._last_ui_action = eval_action
                event_entry = {
                    "time": time.strftime("%H:%M:%S"),
                    "action": eval_action + injected_tag,
                    "confidence": round(eval_conf * 100.0, 1),
                    "step_id": val_res.current_step_id,
                    "step_name": val_res.current_step_name,
                    "status": val_res.status_text,
                    "anomaly_type": val_res.anomaly_type.name if val_res.anomaly_type.name != "NONE" else None,
                    "anomaly_score": val_res.anomaly_score,
                    "is_simulated": is_injected,
                    "is_protocol_anomaly": is_protocol_anomaly,
                    "is_uncertain": is_uncertain_event,
                }
                self.event_log.append(event_entry)
                if len(self.event_log) > 100:
                    self.event_log.pop(0)

                if is_protocol_anomaly:
                    self.anomaly_log.append(event_entry)
                    if len(self.anomaly_log) > 50:
                        self.anomaly_log.pop(0)
                elif is_uncertain_event:
                    self.uncertain_log.append(event_entry)
                    if len(self.uncertain_log) > 50:
                        self.uncertain_log.pop(0)

            # Accumulate running history
            self.confidence_history.append(float(eval_conf))
            if len(self.confidence_history) > 50:
                self.confidence_history.pop(0)

            # ── 4. Latency & FPS Calculation ─────────────────────────────────
            total_elapsed = time.perf_counter() - t0
            total_ms = total_elapsed * 1000.0
            fps_tracker.append(1.0 / max(total_elapsed, 1e-4))
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            fps_actual = float(np.mean(fps_tracker))

            self.fps_history.append(fps_actual)
            if len(self.fps_history) > 60:
                self.fps_history.pop(0)

            self.latency_history.append(pipeline_latency_ms)
            if len(self.latency_history) > 60:
                self.latency_history.pop(0)

            # ── 5. Render HUD Overlay & Encode Frame (JPEG Q75) ──────────────
            fsm_str = f"Step {val_res.current_step_id}/10: {val_res.current_step_name}"
            metrics_dict = {
                "fps_actual": fps_actual,
                "pipeline_fps": pipeline_fps,
                "detector_ms": det_ms,
                "pose_ms": pose_ms,
                "hand_ms": hand_ms,
                "hoi_ms": hoi_ms,
                "har_latency_ms": har_ms,
                "pipeline_latency_ms": pipeline_latency_ms,
                "total_latency_ms": total_ms,
                "resolution": f"{w}x{h}",
                "device": self.device.upper(),
                "buffer_fill": har_telem.get("buffer_fill", 0),
                "buffer_cap": har_telem.get("buffer_capacity", 60),
                "hoi_state": interactions[0].state.name if interactions else "—",
                "har_action": eval_action,
                "har_confidence": eval_conf,
                "har_is_confident": har_telem.get("is_confident", True),
                "protocol_id": "HOME_DEMO_PROTOCOL_01",
                "expected_action": val_res.expected_action,
                "fsm_status": val_res.status_text,
                "anomaly_score": val_res.anomaly_score,
                "fsm_step": fsm_str,
                "fsm_valid": val_res.is_valid,
                "fsm_anomaly": val_res.anomaly_message if not val_res.is_valid else "",
            }

            canvas = self.visualizer.render(
                frame, objects, pose_res, left_hand, right_hand,
                metrics=metrics_dict, interactions=interactions,
            )

            # Encode as JPEG at Quality 75 (fast encoding, minimal artifacts)
            _, jpeg_buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])

            # Mission Summary stats
            comp_count = len(val_res.completed_step_ids)
            comp_pct = round((comp_count / 10.0) * 100.0, 1)
            avg_conf = round(float(np.mean(self.confidence_history)) * 100.0, 1) if self.confidence_history else 100.0
            avg_fps = round(float(np.mean(self.fps_history)), 1) if self.fps_history else fps_actual
            avg_lat = round(float(np.mean(self.latency_history)), 1) if self.latency_history else pipeline_latency_ms

            dur_sec = (time.time() - self.mission_start_time) if self.mission_start_time else 0.0
            mins = int(dur_sec // 60)
            secs = dur_sec % 60
            dur_str = f"{mins:02d}:{secs:04.1f}s"

            mission_stat = "COMPLETED" if val_res.is_completed and len(self.anomaly_log) == 0 else "ANOMALY_DETECTED" if len(self.anomaly_log) > 0 else "NOMINAL"

            mission_summary_dict = {
                "protocol_id": "HOME_DEMO_PROTOCOL_01",
                "session_id": "SES_001_NOMINAL",
                "lifecycle_state": self.lifecycle_state,
                "completed_steps": comp_count,
                "remaining_steps": max(0, 10 - comp_count),
                "total_steps": 10,
                "completion_pct": comp_pct,
                "mission_duration": dur_str,
                "har_uncertain_count": len(self.uncertain_log),
                "protocol_anomaly_count": len(self.anomaly_log),
                "anomaly_count": len(self.anomaly_log),
                "max_protocol_anomaly_score": round(self.max_protocol_anomaly_score, 2),
                "max_anomaly_score": round(self.max_protocol_anomaly_score, 2),
                "avg_confidence": avg_conf,
                "current_confidence": round(eval_conf * 100.0, 1),
                "current_action": eval_action,
                "expected_action": val_res.expected_action,
                "last_anomaly_reason": val_res.anomaly_type.name if not val_res.is_valid else "NONE",
                "avg_fps": avg_fps,
                "avg_pipeline_latency_ms": avg_lat,
                "mission_status": mission_stat,
            }

            with self.lock:
                self.latest_jpeg_bytes = jpeg_buf.tobytes()
                self.latest_frame_id = proc_idx
                self.latest_metrics = {
                    "status": "ONLINE",
                    "lifecycle_state": self.lifecycle_state,
                    "fps_actual": round(fps_actual, 1),
                    "pipeline_fps": round(pipeline_fps, 1),
                    "capture_fps": round(self.capture_fps, 1),
                    "display_fps": round(self.display_fps, 1),
                    "pipeline_latency_ms": round(pipeline_latency_ms, 1),
                    "total_latency_ms": round(total_ms, 1),
                    "detector_ms": round(det_ms, 1),
                    "pose_ms": round(pose_ms, 1),
                    "hand_ms": round(hand_ms, 1),
                    "hoi_ms": round(hoi_ms, 2),
                    "har_latency_ms": round(har_ms, 2),
                    "device": self.device.upper(),
                    "har_action": eval_action,
                    "har_confidence": round(eval_conf, 4),
                    "har_is_confident": har_telem.get("is_confident", True),
                    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                    "protocol_id": "HOME_DEMO_PROTOCOL_01",
                    "current_step_id": val_res.current_step_id,
                    "total_steps": 10,
                    "current_step_name": val_res.current_step_name,
                    "expected_action": val_res.expected_action,
                    "observed_action": eval_action,
                    "fsm_status": val_res.status_text,
                    "fsm_valid": val_res.is_valid,
                    "anomaly_score": val_res.anomaly_score,
                    "anomaly_message": val_res.anomaly_message,
                    "is_completed": val_res.is_completed,
                    "completed_steps": val_res.completed_step_ids,
                    "step_timestamps": val_res.step_timestamps,
                    "source_type": self.mode.upper(),
                    "source_path": str(self.source),
                    "buffer_fill": har_telem.get("buffer_fill", 0),
                    "buffer_cap": har_telem.get("buffer_capacity", 60),
                    "error_message": self.error_message,
                    "mission_summary": mission_summary_dict,
                }


# ─── FastAPI Web Application ──────────────────────────────────────────────────
app = FastAPI(title="ISRO BAS-HAR Operator Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker = PipelineWorker()

# Mount static folder
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class StartRequest(BaseModel):
    source: str = "0"
    mode: str = "webcam"


class AnomalyInjectRequest(BaseModel):
    scenario: str = "OUT_OF_ORDER_ACTION"


# ─── REST API Endpoints ───────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    """Return overall pipeline, lifecycle state, and subsystem health."""
    metrics = worker.latest_metrics
    return {
        "pipeline_status": metrics.get("status", "OFFLINE"),
        "lifecycle_state": worker.lifecycle_state,
        "device": worker.device.upper(),
        "system_prototype_description": "AI/ML-based astronaut activity recognition and protocol monitoring prototype",
        "models": {
            "yolo_detector": "READY",
            "mediapipe_pose": "READY",
            "mediapipe_hands": "READY",
            "hoi_interaction": "READY",
            "har_bigru": "READY" if worker.har_classifier.model_loaded else "NOT_LOADED",
            "protocol_fsm": "READY",
        },
        "session": {
            "session_id": "SES_001_NOMINAL",
            "actor_id": "ACTOR_01",
            "protocol_id": "HOME_DEMO_PROTOCOL_01",
            "model_architecture": "Conv1D-BiGRU-Attention",
            "feature_dim": 154,
            "window_size": 60,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        },
    }


@app.get("/api/readiness")
def get_readiness():
    """Return compact system readiness checklist."""
    return worker.get_readiness()


@app.post("/api/demo/inject_anomaly")
def inject_demo_anomaly(req: AnomalyInjectRequest):
    """Evaluator demonstration endpoint for controlled simulated anomaly injection."""
    try:
        res = worker.inject_anomaly(req.scenario)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/protocol")
def get_protocol():
    """Return active protocol steps and current progress."""
    steps_data = []
    if worker.state_machine.protocol:
        for s in worker.state_machine.protocol.steps:
            steps_data.append({
                "step_id": s.step_id,
                "name": s.name,
                "expected_action": s.expected_action,
                "is_completed": s.step_id in worker.state_machine.completed_steps,
                "is_current": s.step_id == worker.latest_metrics.get("current_step_id", 1),
                "timestamp_sec": worker.state_machine.step_completion_timestamps.get(s.step_id),
            })
    return {
        "protocol_id": "HOME_DEMO_PROTOCOL_01",
        "is_completed": worker.state_machine.is_completed,
        "completed_count": len(worker.state_machine.completed_steps),
        "total_count": 10,
        "steps": steps_data,
        "step_timestamps": worker.state_machine.step_completion_timestamps,
    }


@app.get("/api/anomalies")
def get_anomalies():
    """Return detected anomaly and uncertainty records."""
    return {
        "current_anomaly_score": worker.latest_metrics.get("anomaly_score", 0.0),
        "fsm_status": worker.latest_metrics.get("fsm_status", "OK"),
        "fsm_valid": worker.latest_metrics.get("fsm_valid", True),
        "har_uncertain_count": len(worker.uncertain_log),
        "protocol_anomaly_count": len(worker.anomaly_log),
        "total_anomalies": len(worker.anomaly_log),
        "max_protocol_anomaly_score": round(worker.max_protocol_anomaly_score, 2),
        "recent_anomalies": list(reversed(worker.anomaly_log[-20:])),
        "recent_protocol_anomalies": list(reversed(worker.anomaly_log[-20:])),
        "recent_uncertain_events": list(reversed(worker.uncertain_log[-20:])),
    }


@app.get("/api/events")
def get_events():
    """Return chronological telemetry event history."""
    return {
        "total_events": len(worker.event_log),
        "events": list(reversed(worker.event_log[-30:])),
    }


@app.get("/api/mission_summary")
def get_mission_summary():
    """Return comprehensive mission summary statistics."""
    return worker.latest_metrics.get("mission_summary", {})


@app.get("/api/performance")
def get_performance():
    """Return real-time pipeline latency telemetry."""
    m = worker.latest_metrics
    return {
        "fps": m.get("fps_actual", 0.0),
        "pipeline_fps": m.get("pipeline_fps", 0.0),
        "capture_fps": m.get("capture_fps", 0.0),
        "display_fps": m.get("display_fps", 0.0),
        "pipeline_latency_ms": m.get("pipeline_latency_ms", 0.0),
        "total_latency_ms": m.get("total_latency_ms", 0.0),
        "breakdown_ms": {
            "yolo_detector": m.get("detector_ms", 0.0),
            "pose_estimator": m.get("pose_ms", 0.0),
            "hand_tracker": m.get("hand_ms", 0.0),
            "hoi_analysis": m.get("hoi_ms", 0.0),
            "har_inference": m.get("har_latency_ms", 0.0),
        },
        "device": m.get("device", "CPU"),
        "buffer": f"{m.get('buffer_fill', 0)}/{m.get('buffer_cap', 60)}",
    }


@app.post("/api/start")
def start_pipeline(req: StartRequest):
    """Start video ingestion and real-time processing."""
    try:
        worker.start(source=req.source, mode=req.mode)
        return {"status": "SUCCESS", "source": req.source, "mode": req.mode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop")
def stop_pipeline():
    """Stop running video ingestion."""
    worker.stop()
    return {"status": "SUCCESS", "message": "Pipeline stopped."}


@app.post("/api/reset")
def reset_pipeline():
    """Reset protocol session and event log."""
    worker.reset_session()
    return {"status": "SUCCESS", "message": "Protocol session reset."}


# ─── Video Stream Endpoint ────────────────────────────────────────────────────
def gen_frames():
    """Generator streaming JPEG frames as multipart HTTP stream."""
    last_sent_frame = -1
    disp_times: List[float] = []
    while True:
        if worker.latest_frame_id != last_sent_frame:
            frame_bytes = worker.latest_jpeg_bytes
            if frame_bytes is not None:
                t_now = time.perf_counter()
                last_sent_frame = worker.latest_frame_id
                disp_times.append(t_now)
                if len(disp_times) > 30:
                    disp_times.pop(0)
                if len(disp_times) > 1:
                    worker.display_fps = round(float(len(disp_times) - 1) / max(disp_times[-1] - disp_times[0], 1e-4), 1)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        time.sleep(0.01)


@app.get("/video_feed")
def video_feed():
    """MJPEG live video stream."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─── WebSocket Telemetry Endpoint ─────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """Real-time 20Hz telemetry WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = {
                "metrics": worker.latest_metrics,
                "confidence_history": worker.confidence_history[-40:],
                "events": worker.event_log[-5:],
                "completed_steps": list(worker.state_machine.completed_steps),
                "step_timestamps": worker.state_machine.step_completion_timestamps,
                "is_completed": worker.state_machine.is_completed,
                "mission_summary": worker.latest_metrics.get("mission_summary", {}),
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.05)  # 20 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─── Main HTML Dashboard Endpoint ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serve the complete single-page aerospace operator dashboard."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ISRO BAS-HAR Dashboard Static files loading...</h1>"


def main():
    parser = argparse.ArgumentParser(description="ISRO BAS-HAR Operator UI Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--auto_start", action="store_true", help="Auto-start video pipeline on launch")
    parser.add_argument("--video", default="data/raw/videos/SES_001_NOMINAL.mp4", help="Default demo video path")
    args = parser.parse_args()

    if args.auto_start and Path(args.video).exists():
        logger.info(f"Auto-starting video mode with: {args.video}")
        worker.start(source=args.video, mode="video")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
