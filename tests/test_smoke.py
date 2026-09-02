"""End-to-End Smoke Test for Phase 2 Perception Pipeline."""

import unittest
import numpy as np

from src.ingestion.camera_feed import CameraFeed
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.hand_tracker import HandTracker
from src.visualization.perception_visualizer import PerceptionVisualizer


class TestPerceptionSmoke(unittest.TestCase):
    """End-to-end smoke test executing ingestion -> detection -> pose -> hands -> visualization."""

    def test_full_pipeline_smoke_run(self):
        """Execute 5 frames through the entire perception stack."""
        feed = CameraFeed(source="synthetic", target_fps=30)
        feed.start()

        detector = ObjectDetector(model_name_or_path="yolov8n.pt", device="cpu")
        pose_estimator = PoseEstimator(model_complexity=0)
        hand_tracker = HandTracker()
        visualizer = PerceptionVisualizer(show_hud=True)

        frames_tested = 0
        for _ in range(5):
            success, frame, ts = feed.read_frame()
            if not success or frame is None:
                continue

            objects = detector.detect(frame)
            pose = pose_estimator.estimate(frame)
            left_h, right_h = hand_tracker.track(frame)

            metrics = {
                "fps_actual": 30.0,
                "detector_ms": detector.last_latency_ms,
                "pose_ms": pose_estimator.last_latency_ms,
                "hand_ms": hand_tracker.last_latency_ms,
                "total_latency_ms": detector.last_latency_ms + pose_estimator.last_latency_ms + hand_tracker.last_latency_ms,
                "resolution": "1280x720",
                "device": "CPU",
            }

            rendered = visualizer.render(frame, objects, pose, left_h, right_h, metrics)
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.shape, frame.shape)
            frames_tested += 1

        feed.stop()
        self.assertGreaterEqual(frames_tested, 1)


if __name__ == "__main__":
    unittest.main()
