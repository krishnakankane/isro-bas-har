"""ISRO BAS HAR — Live Perception + HOI + Temporal HAR Demo.

Runs the full end-to-end multi-modal perception and action recognition stack:
    Camera Feed
    → YOLOv8n Object Detector
    → MediaPipe Pose Landmarker
    → MediaPipe Hand Landmarker
    → HOI Spatial Interaction Engine
    → 154-D Feature Extractor
    → Rolling 60-Frame Temporal Buffer
    → Conv1D-BiGRU Temporal HAR Neural Network
    → Protocol Sequence State Machine (FSM)

Displays a live OpenCV HUD with:
    - YOLOv8n object bounding boxes
    - MediaPipe Pose skeletal wireframe
    - MediaPipe Hand landmarks and grasp state
    - HOI interaction state panel
    - Top HAR activity recognition banner (action name, confidence, debouncing status)
    - Full telemetry HUD (FPS, YOLO/Pose/Hands/HOI/HAR latencies, buffer fill gauge, FSM step)

Usage
-----
# Webcam (camera index 0):
    python -m scripts.run_perception_demo --source 0

# Local video file:
    python -m scripts.run_perception_demo --source data/raw/videos/SES_001_NOMINAL.mp4

# Synthetic test feed (headless / no camera):
    python -m scripts.run_perception_demo --source synthetic --max_frames 100

Controls
--------
    Q / Esc  — Quit
    S        — Save current annotated frame → demo_snapshot.jpg
    P        — Pause / Resume
    R        — Reset FSM protocol state machine and temporal buffer
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.camera_feed import CameraFeed
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.hand_tracker import HandTracker
from src.perception.hoi_detector import HOIDetector, HOIConfig
from src.activity_engine.feature_extractor import PerceptionFeatureExtractor, FEATURE_DIM
from src.activity_engine.temporal_buffer import TemporalFeatureBuffer
from src.activity_engine.temporal_har import TemporalHARClassifier
from src.core.state_machine import ProtocolStateMachine
from src.visualization.perception_visualizer import PerceptionVisualizer
from src.core.models import FramePerceptionResult, InteractionState


# ─── FPS tracker using a rolling window ──────────────────────────────────────
class RollingFPS:
    """Smooth FPS estimator using a sliding window of frame timestamps."""

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Main demo loop ───────────────────────────────────────────────────────────
def run_demo(
    source: str,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    show_window: bool = True,
    max_frames: int = 0,
    record_path: str = "",
    checkpoint_path: str = "models/har/conv1d_bigru_best.pt",
    confidence_threshold: float = 0.65,
    smoothing_window: int = 5,
    protocol_path: str = "config/experiment_protocols.json",
    device: Optional[str] = None,
) -> dict:
    """Run the full live perception + HOI + temporal HAR demo and return summary metrics.

    Args:
        source:               Camera index ('0'), video path, or 'synthetic'.
        width:                Target frame width (pixels).
        height:               Target frame height (pixels).
        target_fps:           Target acquisition FPS.
        show_window:          Whether to display the OpenCV window.
        max_frames:           Maximum frames to process (0 = unlimited).
        record_path:          Optional JSONL path for feature recording.
        checkpoint_path:      Path to trained PyTorch HAR checkpoint (.pt).
        confidence_threshold: Minimum confidence to consider action confirmed.
        smoothing_window:     Number of recent frames to average predictions over.
        protocol_path:        Path to experiment protocol JSON.
        device:               'cuda' or 'cpu' (default: auto).

    Returns:
        Dict with execution summary statistics.
    """
    # ── Parse source ──────────────────────────────────────────────────────────
    cam_source: object = int(source) if source.isdigit() else source
    exec_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("ISRO BAS HAR — LIVE PERCEPTION + TEMPORAL HAR PIPELINE")
    print("=" * 68)
    print(f"  Source          : {source!r}")
    print(f"  Resolution      : {width}x{height} @ {target_fps} FPS target")
    print(f"  Compute Device  : {exec_device.upper()}")
    if torch.cuda.is_available():
        print(f"  GPU Hardware    : {torch.cuda.get_device_name(0)}")
    print(f"  HAR Checkpoint  : {checkpoint_path}")
    print(f"  Confidence Gate : >= {confidence_threshold * 100:.0f}% (else UNCERTAIN)")
    print(f"  Debounce Window : {smoothing_window} frames")
    print(f"  Protocol Path   : {protocol_path}")
    print("=" * 68)

    # ── Initialise Ingestion ──────────────────────────────────────────────────
    feed = CameraFeed(
        source=cam_source,
        target_fps=target_fps,
        loop_video=True,
        frame_width=width,
        frame_height=height,
    )
    feed.start()
    time.sleep(0.4)

    # ── Initialise Perception Models ──────────────────────────────────────────
    yolo_device = "cuda" if (exec_device == "cuda" and torch.cuda.is_available()) else "cpu"
    detector = ObjectDetector(model_name_or_path="yolov8n.pt", device=yolo_device)
    pose_est = PoseEstimator(model_path="models/mediapipe/pose_landmarker_lite.task")
    hand_trk = HandTracker(model_path="models/mediapipe/hand_landmarker.task")
    hoi_det  = HOIDetector(config=HOIConfig())

    # ── Initialise Temporal HAR Classifier ────────────────────────────────────
    har_classifier = TemporalHARClassifier(
        window_size=60,
        feature_dim=FEATURE_DIM,
        checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window,
        device=exec_device,
    )

    if har_classifier.model_loaded:
        print(f"[HAR] Successfully loaded checkpoint '{checkpoint_path}' on {exec_device.upper()}.")
    else:
        print(f"[HAR] WARNING: Checkpoint not found at '{checkpoint_path}'. Falling back to baseline.")

    # ── Initialise Protocol State Machine ─────────────────────────────────────
    state_machine = ProtocolStateMachine()
    if protocol_path and Path(protocol_path).exists():
        state_machine.load_from_json(protocol_path)
        print(f"[FSM] Loaded protocol: '{state_machine.protocol.protocol_name}' ({len(state_machine.protocol.steps)} steps)")
    else:
        print("[FSM] No protocol loaded.")

    visualizer = PerceptionVisualizer(show_hud=True)
    fps_tracker = RollingFPS(window=30)

    # ── Metric accumulators ───────────────────────────────────────────────────
    detector_ms_acc: List[float] = []
    pose_ms_acc: List[float]     = []
    hand_ms_acc: List[float]     = []
    hoi_ms_acc: List[float]      = []
    har_ms_acc: List[float]      = []
    total_ms_acc: List[float]    = []
    fps_acc: List[float]         = []

    yolo_detection_count = 0
    pose_detection_count = 0
    hand_detection_count = 0
    hoi_event_count      = 0
    frames_processed     = 0
    paused               = False
    canvas               = None

    predicted_actions_dist: Dict[str, int] = {}

    if show_window:
        cv2.namedWindow("ISRO BAS HAR — Live Inference Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ISRO BAS HAR — Live Inference Demo", width, height)

    print("[Demo] Pipeline started. Press Q to quit, S to snapshot, P to pause, R to reset FSM.\n")

    try:
        while True:
            # ── Key handling ───────────────────────────────────────────────
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[Demo] Quit requested.")
                    break
                elif key == ord("s") and canvas is not None:
                    cv2.imwrite("demo_snapshot.jpg", canvas)
                    print("[Demo] Snapshot saved → demo_snapshot.jpg")
                elif key == ord("p"):
                    paused = not paused
                    print(f"[Demo] {'Paused' if paused else 'Resumed'}.")
                elif key == ord("r"):
                    har_classifier.reset_buffer()
                    state_machine.reset()
                    print("[Demo] Reset FSM state machine and HAR temporal buffer.")

            if paused:
                time.sleep(0.05)
                continue

            # ── Frame acquisition ──────────────────────────────────────────
            success, frame, _ts = feed.read_frame(timeout_sec=0.5)
            if not success or frame is None:
                time.sleep(0.05)
                continue

            # ── Perception pipeline ────────────────────────────────────────
            t_pipe_start = time.perf_counter()

            objects = detector.detect(frame)
            pose_res = pose_est.estimate(frame)
            left_hand, right_hand = hand_trk.track(frame)

            # HOI spatial analysis
            t_hoi_start = time.perf_counter()
            interactions = hoi_det.analyze_interaction(objects, left_hand, right_hand)
            hoi_elapsed_ms = (time.perf_counter() - t_hoi_start) * 1000.0

            # ── Build frame perception result ──────────────────────────────
            frame_result = FramePerceptionResult(
                frame_id=frames_processed,
                timestamp_sec=time.time(),
                objects=objects,
                body_pose=pose_res,
                left_hand=left_hand,
                right_hand=right_hand,
                interactions=interactions,
            )

            # ── Temporal HAR inference ─────────────────────────────────────
            action_name, confidence = har_classifier.update_and_classify(frame_result)
            har_telem = har_classifier.get_telemetry()
            har_elapsed_ms = float(har_telem["latency_ms"])

            # ── Protocol FSM evaluation ────────────────────────────────────
            # Only evaluate valid, confident actions in the state machine
            clean_action = har_telem["raw_action"] if har_telem["is_confident"] else "ACTION_IDLE"
            val_res = state_machine.evaluate_action(
                action_name=clean_action,
                confidence=confidence,
                timestamp=time.time(),
            )

            total_ms = (time.perf_counter() - t_pipe_start) * 1000.0

            # ── Accumulate metrics ─────────────────────────────────────────
            det_ms = detector.last_latency_ms
            pse_ms = pose_est.last_latency_ms
            hnd_ms = hand_trk.last_latency_ms

            detector_ms_acc.append(det_ms)
            pose_ms_acc.append(pse_ms)
            hand_ms_acc.append(hnd_ms)
            hoi_ms_acc.append(hoi_elapsed_ms)
            har_ms_acc.append(har_elapsed_ms)
            total_ms_acc.append(total_ms)

            if objects:              yolo_detection_count += 1
            if pose_res.is_detected: pose_detection_count += 1
            if (left_hand and left_hand.is_detected) or (right_hand and right_hand.is_detected):
                hand_detection_count += 1
            if interactions:
                hoi_event_count += len(interactions)

            fps = fps_tracker.tick()
            fps_acc.append(fps)
            frames_processed += 1

            predicted_actions_dist[action_name] = predicted_actions_dist.get(action_name, 0) + 1

            # ── Current HOI state for HUD ──────────────────────────────────
            top_hoi_state = "—"
            if interactions:
                prio = {
                    InteractionState.MANIPULATING: 5,
                    InteractionState.GRASPING: 4,
                    InteractionState.APPROACHING: 3,
                    InteractionState.RELEASED: 2,
                    InteractionState.NONE: 1,
                    InteractionState.UNKNOWN: 0,
                }
                best = max(interactions, key=lambda e: prio.get(e.state, 0))
                top_hoi_state = f"{best.state.name} ({best.hand_side[:1]}H↔{best.target_object_name[:8]})"

            # ── FSM status string ──────────────────────────────────────────
            fsm_str = ""
            if state_machine.protocol:
                curr_id = val_res.current_step_id
                total_steps = len(state_machine.protocol.steps)
                fsm_str = f"Step {curr_id}/{total_steps}: {val_res.current_step_name}"

            # ── Build metrics dict for HUD ─────────────────────────────────
            metrics = {
                "fps_actual":        fps,
                "detector_ms":       det_ms,
                "pose_ms":           pse_ms,
                "hand_ms":           hnd_ms,
                "hoi_ms":            hoi_elapsed_ms,
                "har_latency_ms":    har_elapsed_ms,
                "total_latency_ms":  total_ms,
                "resolution":        f"{width}x{height}",
                "device":            exec_device.upper(),
                "buffer_fill":       har_telem["buffer_fill"],
                "buffer_cap":        har_telem["buffer_capacity"],
                "hoi_state":         top_hoi_state,
                "har_action":        action_name,
                "har_confidence":    confidence,
                "har_is_confident":  har_telem["is_confident"],
                "protocol_id":       state_machine.protocol.protocol_id if state_machine.protocol else "HOME_DEMO_PROTOCOL_01",
                "expected_action":   val_res.expected_action,
                "fsm_status":        val_res.status_text,
                "anomaly_score":     val_res.anomaly_score,
                "fsm_step":          fsm_str,
                "fsm_valid":         val_res.is_valid,
                "fsm_anomaly":       val_res.anomaly_message if not val_res.is_valid else "",
            }

            # ── Render annotated frame ─────────────────────────────────────
            canvas = visualizer.render(
                frame, objects, pose_res, left_hand, right_hand,
                metrics=metrics, interactions=interactions,
            )

            if show_window:
                cv2.imshow("ISRO BAS HAR — Live Inference Demo", canvas)

            # ── Console logging (every 30 frames) ──────────────────────────
            if frames_processed % 30 == 0:
                print(
                    f"[Demo] Frame {frames_processed:05d} | "
                    f"FPS: {fps:4.1f} | "
                    f"HAR: {action_name:<20} ({confidence*100:5.1f}%) | "
                    f"HAR Lat: {har_elapsed_ms:4.2f}ms | "
                    f"Pipe: {total_ms:4.1f}ms | "
                    f"Buf: {har_telem['buffer_fill']:02d}/60 | "
                    f"{fsm_str}"
                )

            # ── Auto-exit ──────────────────────────────────────────────────
            if max_frames > 0 and frames_processed >= max_frames:
                print(f"[Demo] Reached max_frames={max_frames}. Exiting.")
                break

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted by Ctrl+C.")
    finally:
        feed.stop()
        pose_est.close()
        hand_trk.close()
        if show_window:
            cv2.destroyAllWindows()

    # ── Summary Calculation ───────────────────────────────────────────────────
    def safe_mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    summary = {
        "frames_processed":       frames_processed,
        "mean_fps":               round(safe_mean([f for f in fps_acc if f > 0]), 2),
        "mean_total_ms":          round(safe_mean(total_ms_acc), 2),
        "mean_detector_ms":       round(safe_mean(detector_ms_acc), 2),
        "mean_pose_ms":           round(safe_mean(pose_ms_acc), 2),
        "mean_hand_ms":           round(safe_mean(hand_ms_acc), 2),
        "mean_hoi_ms":            round(safe_mean(hoi_ms_acc), 3),
        "mean_har_ms":            round(safe_mean(har_ms_acc), 3),
        "yolo_detections":        yolo_detection_count,
        "pose_detections":        pose_detection_count,
        "hand_detections":        hand_detection_count,
        "hoi_events":             hoi_event_count,
        "predicted_actions_dist": predicted_actions_dist,
        "feature_dim":            FEATURE_DIM,
        "buffer_window":          60,
        "model_loaded":           har_classifier.model_loaded,
        "device":                 exec_device.upper(),
    }

    print("\n" + "=" * 68)
    print("LIVE DEMO SESSION SUMMARY")
    print("=" * 68)
    print(f"  Frames Processed : {frames_processed}")
    print(f"  Mean FPS         : {summary['mean_fps']:.1f} FPS")
    print(f"  Mean Total Ms    : {summary['mean_total_ms']:.2f} ms")
    print(f"  Mean YOLO Ms     : {summary['mean_detector_ms']:.2f} ms")
    print(f"  Mean Pose Ms     : {summary['mean_pose_ms']:.2f} ms")
    print(f"  Mean Hand Ms     : {summary['mean_hand_ms']:.2f} ms")
    print(f"  Mean HOI Ms      : {summary['mean_hoi_ms']:.3f} ms")
    print(f"  Mean HAR Ms      : {summary['mean_har_ms']:.3f} ms")
    print(f"  HAR Model Loaded : {summary['model_loaded']} ({exec_device.upper()})")
    print(f"  Predicted Actions: {predicted_actions_dist}")
    print("=" * 68)

    return summary


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Live Perception + Temporal HAR Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default="synthetic",
                        help="Camera index ('0'), video file path, or 'synthetic'. Default: synthetic")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target acquisition FPS (default: 30)")
    parser.add_argument("--checkpoint", default="models/har/conv1d_bigru_best.pt",
                        help="Path to trained HAR checkpoint .pt file")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                        help="Confidence gate threshold for HAR (default: 0.65)")
    parser.add_argument("--smoothing_window", type=int, default=5,
                        help="Frames window for HAR prediction smoothing (default: 5)")
    parser.add_argument("--protocol", default="config/experiment_protocols.json",
                        help="Path to experiment protocol JSON")
    parser.add_argument("--device", default="", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = unlimited)")
    parser.add_argument("--record", default="", help="Optional path to record feature JSONL")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        source=args.source,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        show_window=True,
        max_frames=args.max_frames,
        record_path=args.record,
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.confidence_threshold,
        smoothing_window=args.smoothing_window,
        protocol_path=args.protocol,
        device=args.device or None,
    )
