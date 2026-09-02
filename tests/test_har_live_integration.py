"""Unit and Integration Tests for Live Temporal HAR Inference Pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.core.models import (
    BoundingBox,
    DetectedObject,
    FramePerceptionResult,
    HandLandmark,
    HandTrackingResult,
    InteractionEvent,
    InteractionState,
    PoseEstimationResult,
    PoseLandmark,
)
from src.activity_engine.har_model import HAR_ACTION_CLASSES, NUM_HAR_CLASSES
from src.activity_engine.temporal_har import PredictionSmoother, TemporalHARClassifier
from scripts.run_perception_demo import run_demo


def create_dummy_perception(frame_id: int = 0) -> FramePerceptionResult:
    """Helper to synthesize dummy perception frame with simulated pose, hands, and objects."""
    pose_lms = [
        PoseLandmark(id=i, name=f"landmark_{i}", x=0.5 + 0.01 * np.sin(frame_id * 0.1), y=0.5, z=0.0, visibility=0.95)
        for i in range(33)
    ]
    pose = PoseEstimationResult(landmarks=pose_lms, torso_angle=0.0, is_detected=True)

    hand_lms = [
        HandLandmark(id=i, x=0.45, y=0.55, z=0.0)
        for i in range(21)
    ]
    left_hand = HandTrackingResult(
        handedness="Left",
        landmarks=hand_lms,
        wrist_pos=(0.45, 0.55),
        is_grasping=False,
        is_detected=True,
    )

    bbox = BoundingBox(xmin=0.40, ymin=0.50, xmax=0.55, ymax=0.70)
    objects = [
        DetectedObject(
            class_id=0,
            class_name="bottle",
            confidence=0.91,
            bbox=bbox,
        )
    ]

    interactions = [
        InteractionEvent(
            hand_side="Left",
            target_object_name="bottle",
            target_object_bbox=bbox,
            state=InteractionState.APPROACHING,
            proximity_distance_px=0.08,
            duration_frames=10,
        )
    ]

    return FramePerceptionResult(
        frame_id=frame_id,
        timestamp_sec=float(frame_id) / 30.0,
        objects=objects,
        body_pose=pose,
        left_hand=left_hand,
        right_hand=None,
        interactions=interactions,
    )


class TestLiveHARIntegration(unittest.TestCase):
    """Test suite for live HAR classifier, smoothing, confidence gating, and demo pipeline."""

    def setUp(self):
        self.checkpoint_path = Path("models/har/conv1d_bigru_best.pt")

    def test_01_checkpoint_loading_and_telemetry(self):
        """Verify TemporalHARClassifier loads trained checkpoint and reports telemetry."""
        if not self.checkpoint_path.exists():
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")

        classifier = TemporalHARClassifier(
            checkpoint_path=self.checkpoint_path,
            confidence_threshold=0.65,
            smoothing_window=5,
            device="cpu",
        )
        self.assertTrue(classifier.model_loaded)
        self.assertIsNotNone(classifier.model)

        telem = classifier.get_telemetry()
        self.assertTrue(telem["model_loaded"])
        self.assertEqual(telem["buffer_capacity"], 60)
        self.assertEqual(telem["device"], "CPU")

    def test_02_rolling_60_frame_buffering(self):
        """Verify rolling buffer stores up to 60 frames and correctly reports fill."""
        classifier = TemporalHARClassifier(
            checkpoint_path=self.checkpoint_path if self.checkpoint_path.exists() else None,
            device="cpu",
        )

        for i in range(75):
            perc = create_dummy_perception(frame_id=i)
            action, conf = classifier.update_and_classify(perc)
            self.assertIsInstance(action, str)
            self.assertIsInstance(conf, float)

        telem = classifier.get_telemetry()
        self.assertEqual(telem["buffer_fill"], 60)
        self.assertTrue(telem["is_full"])

        classifier.reset_buffer()
        telem_after = classifier.get_telemetry()
        self.assertEqual(telem_after["buffer_fill"], 0)
        self.assertFalse(telem_after["is_full"])

    def test_03_har_inference_output_contract(self):
        """Verify HAR inference returns valid action class names and confidence in [0, 1]."""
        if not self.checkpoint_path.exists():
            self.skipTest(f"Checkpoint not found at {self.checkpoint_path}")

        classifier = TemporalHARClassifier(
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )

        for i in range(10):
            perc = create_dummy_perception(frame_id=i)
            action, conf = classifier.update_and_classify(perc)

        telem = classifier.get_telemetry()
        self.assertIn(telem["raw_action"], HAR_ACTION_CLASSES.values())
        self.assertGreaterEqual(telem["raw_confidence"], 0.0)
        self.assertLessEqual(telem["raw_confidence"], 1.0)
        self.assertGreater(telem["latency_ms"], 0.0)

    def test_04_confidence_threshold_behavior(self):
        """Verify low-confidence predictions are flagged as UNCERTAIN."""
        smoother = PredictionSmoother(window_size=3, confidence_threshold=0.80)

        # Feed low confidence uniform distribution (0.1 each for 10 classes)
        low_conf_probs = np.full(NUM_HAR_CLASSES, 0.1, dtype=np.float32)
        out_action, conf, raw_act, raw_conf, is_conf = smoother.update(low_conf_probs)

        self.assertFalse(is_conf)
        self.assertTrue(out_action.startswith("UNCERTAIN"))
        self.assertLess(conf, 0.80)

        # Feed high confidence distribution (0.95 on class 2)
        high_conf_probs = np.zeros(NUM_HAR_CLASSES, dtype=np.float32)
        high_conf_probs[2] = 0.95
        # Feed 3 times to satisfy window
        for _ in range(3):
            out_action, conf, raw_act, raw_conf, is_conf = smoother.update(high_conf_probs)

        self.assertTrue(is_conf)
        self.assertEqual(out_action, "ACTION_HOLD_BOTTLE")
        self.assertGreaterEqual(conf, 0.80)

    def test_05_prediction_smoothing_and_debouncing(self):
        """Verify a single transient noisy frame does not switch the active debounced action."""
        smoother = PredictionSmoother(
            window_size=5,
            confidence_threshold=0.50,
            min_consecutive_switch=3,
        )

        # Settle in state 1 (SANITIZE)
        probs_sanitize = np.zeros(NUM_HAR_CLASSES, dtype=np.float32)
        probs_sanitize[1] = 0.90
        for _ in range(5):
            smoother.update(probs_sanitize)

        self.assertEqual(smoother.current_stable_action, "ACTION_SANITIZE")

        # Ingest 1 transient noisy frame (e.g. class 9 CLOSE_BOX)
        probs_noisy = np.zeros(NUM_HAR_CLASSES, dtype=np.float32)
        probs_noisy[9] = 0.95
        out_action, _, _, _, _ = smoother.update(probs_noisy)

        # Stable action should remain SANITIZE because 1 frame is debounced
        self.assertEqual(smoother.current_stable_action, "ACTION_SANITIZE")

    def test_06_missing_checkpoint_graceful_fallback(self):
        """Verify initializing with non-existent checkpoint falls back to baseline gracefully."""
        classifier = TemporalHARClassifier(
            checkpoint_path="non_existent_model_checkpoint.pt",
            device="cpu",
        )
        self.assertFalse(classifier.model_loaded)
        self.assertIsNone(classifier.model)

        perc = create_dummy_perception(0)
        action, conf = classifier.update_and_classify(perc)
        self.assertEqual(action, "ACTION_IDLE")
        self.assertEqual(conf, 1.0)

    def test_07_end_to_end_synthetic_pipeline(self):
        """Run complete live demo pipeline on synthetic feed for 20 frames."""
        summary = run_demo(
            source="synthetic",
            width=640,
            height=480,
            target_fps=30,
            show_window=False,
            max_frames=20,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path.exists() else "",
            device="cpu",
        )

        self.assertEqual(summary["frames_processed"], 20)
        self.assertGreater(summary["mean_fps"], 0.0)
        self.assertIn("mean_har_ms", summary)
        self.assertIn("predicted_actions_dist", summary)


if __name__ == "__main__":
    unittest.main()
