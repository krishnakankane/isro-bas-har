"""Smoke test for the run_perception_demo script.

Verifies the demo pipeline runs end-to-end in headless mode on the synthetic feed,
produces annotated frames, and reports valid latency metrics.
"""

import unittest
import sys
from pathlib import Path

# Ensure project root is on path for direct test execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_perception_demo import run_demo


class TestPerceptionDemoSynthetic(unittest.TestCase):
    """Headless smoke test for the full perception demo pipeline."""

    def test_demo_runs_on_synthetic_feed(self):
        """Run demo for 30 synthetic frames and verify valid output metrics."""
        summary = run_demo(
            source="synthetic",
            width=640,
            height=480,
            target_fps=30,
            show_window=False,
            max_frames=30,
            record_path="",
        )

        self.assertGreaterEqual(summary["frames_processed"], 25,
                                "Expected at least 25 frames processed from synthetic feed.")

        # FPS should be non-trivially positive once the rolling window fills
        self.assertGreater(summary["mean_fps"], 0.0)

        # Per-component latencies must be non-negative
        self.assertGreaterEqual(summary["mean_detector_ms"], 0.0)
        self.assertGreaterEqual(summary["mean_pose_ms"], 0.0)
        self.assertGreaterEqual(summary["mean_hand_ms"], 0.0)
        self.assertGreaterEqual(summary["mean_hoi_ms"], 0.0)
        self.assertGreaterEqual(summary["mean_total_ms"], 0.0)

        # Total pipeline latency should be > 0 (models ran)
        self.assertGreater(summary["mean_total_ms"], 0.0,
                           "Total pipeline latency must be > 0 ms — model did not run.")

        # Total ms should be plausibly bounded (< 5 seconds per frame would be absurd)
        self.assertLess(summary["mean_total_ms"], 5000.0)

        # Feature dim must match schema constant
        self.assertEqual(summary["feature_dim"], 154)
        self.assertEqual(summary["buffer_window"], 60)

    def test_demo_summary_keys_present(self):
        """Verify the summary dict always contains all expected keys."""
        summary = run_demo(
            source="synthetic",
            width=320,
            height=240,
            target_fps=30,
            show_window=False,
            max_frames=5,
            record_path="",
        )
        expected_keys = {
            "frames_processed", "mean_fps", "mean_total_ms",
            "mean_detector_ms", "mean_pose_ms", "mean_hand_ms",
            "mean_hoi_ms", "hoi_events", "feature_dim", "buffer_window",
            "yolo_detections", "pose_detections", "hand_detections",
        }
        for key in expected_keys:
            self.assertIn(key, summary, f"Missing key in summary: {key!r}")


if __name__ == "__main__":
    unittest.main()
