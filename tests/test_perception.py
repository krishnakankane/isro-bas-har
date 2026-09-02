"""Unit tests for Perception Components: Ingestion, Object Detector, Pose, Hands, and Robustness."""

import unittest
import numpy as np

from src.ingestion.camera_feed import CameraFeed
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.hand_tracker import HandTracker
from src.core.models import (
    DetectedObject,
    BoundingBox,
    PoseEstimationResult,
    HandTrackingResult,
)


class TestCameraFeedIngestion(unittest.TestCase):
    """Test suite for CameraFeed video ingestion."""

    def test_synthetic_feed_ingestion(self):
        """Verify synthetic feed produces valid frame format and timestamps."""
        feed = CameraFeed(source="synthetic", target_fps=30, frame_width=640, frame_height=480)
        feed.start()
        self.assertTrue(feed.is_running())

        # Read frames
        success, frame, ts = feed.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(ts, 0.0)

        feed.stop()
        self.assertFalse(feed.is_running())

    def test_invalid_video_source_graceful_fallback(self):
        """Verify non-existent file source falls back to synthetic feed safely."""
        feed = CameraFeed(source="non_existent_file_xyz_123.mp4", target_fps=30)
        feed.start()
        self.assertTrue(feed.is_running())
        success, frame, ts = feed.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        feed.stop()


class TestPerceptionSchemas(unittest.TestCase):
    """Test output data schemas of Object Detector, Pose Estimator, and Hand Tracker."""

    def setUp(self):
        self.detector = ObjectDetector(model_name_or_path="yolov8n.pt", device="cpu")
        self.pose_estimator = PoseEstimator(model_complexity=0)
        self.hand_tracker = HandTracker()

    def test_detector_schema_and_dummy_input(self):
        """Verify detector handles dummy frame and returns valid DetectedObject instances."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(test_frame)
        self.assertIsInstance(detections, list)

        # If detections occur, verify schema
        for obj in detections:
            self.assertIsInstance(obj, DetectedObject)
            self.assertIsInstance(obj.class_id, int)
            self.assertIsInstance(obj.class_name, str)
            self.assertIsInstance(obj.confidence, float)
            self.assertIsInstance(obj.bbox, BoundingBox)
            self.assertGreaterEqual(obj.bbox.area, 0.0)

    def test_pose_schema_and_dummy_input(self):
        """Verify pose estimator output schema."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose_res = self.pose_estimator.estimate(test_frame)
        self.assertIsInstance(pose_res, PoseEstimationResult)
        self.assertIsInstance(pose_res.landmarks, list)
        self.assertIsInstance(pose_res.torso_angle, float)
        self.assertIsInstance(pose_res.is_detected, bool)

    def test_hand_tracker_schema_and_dummy_input(self):
        """Verify hand tracker output schema."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        left_h, right_h = self.hand_tracker.track(test_frame)
        # On blank image, hands should be None or is_detected False
        if left_h is not None:
            self.assertIsInstance(left_h, HandTrackingResult)
        if right_h is not None:
            self.assertIsInstance(right_h, HandTrackingResult)


class TestPerceptionRobustness(unittest.TestCase):
    """Test perception robustness against corrupt/empty inputs and missing assets."""

    def setUp(self):
        self.detector = ObjectDetector(model_name_or_path="yolov8n.pt", device="cpu")
        self.pose_estimator = PoseEstimator()
        self.hand_tracker = HandTracker()

    def test_none_frame_handling(self):
        """Verify None frames do not raise unhandled exceptions."""
        self.assertEqual(self.detector.detect(None), [])
        pose_res = self.pose_estimator.estimate(None)
        self.assertFalse(pose_res.is_detected)
        left_h, right_h = self.hand_tracker.track(None)
        self.assertIsNone(left_h)
        self.assertIsNone(right_h)

    def test_empty_frame_handling(self):
        """Verify empty numpy arrays do not raise unhandled exceptions."""
        empty_frame = np.array([], dtype=np.uint8)
        self.assertEqual(self.detector.detect(empty_frame), [])
        pose_res = self.pose_estimator.estimate(empty_frame)
        self.assertFalse(pose_res.is_detected)
        left_h, right_h = self.hand_tracker.track(empty_frame)
        self.assertIsNone(left_h)
        self.assertIsNone(right_h)

    def test_missing_model_raises_explicit_error(self):
        """Verify missing model file raises FileNotFoundError instead of silent fallback."""
        with self.assertRaises(FileNotFoundError):
            PoseEstimator(model_path="models/mediapipe/non_existent_pose.task")

        with self.assertRaises(FileNotFoundError):
            HandTracker(model_path="models/mediapipe/non_existent_hand.task")

    def test_pose_estimator_full_variant(self):
        """Verify pose estimator can initialize and run with pose_landmarker_full.task."""
        full_pose = PoseEstimator(model_path="models/mediapipe/pose_landmarker_full.task")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = full_pose.estimate(test_frame)
        self.assertIsInstance(res, PoseEstimationResult)
        self.assertGreaterEqual(full_pose.last_latency_ms, 0.0)
        full_pose.close()


if __name__ == "__main__":
    unittest.main()
