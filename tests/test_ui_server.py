"""Unit and API integration tests for Phase 12 Operator UI Server."""

import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from src.ui.server import app, worker
from src.core.state_machine import ProtocolStateMachine


class TestOperatorUIServer(unittest.TestCase):
    """Test suite for Operator Dashboard FastAPI server and backend pipeline worker."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_index_html_endpoint(self):
        """Verify root endpoint returns the operator dashboard HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ISRO BAS-HAR", response.text)
        self.assertIn("LIVE PERCEPTION", response.text)

    def test_02_api_status(self):
        """Verify /api/status returns subsystem models and metadata."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pipeline_status", data)
        self.assertIn("models", data)
        self.assertEqual(data["models"]["yolo_detector"], "READY")
        self.assertEqual(data["models"]["protocol_fsm"], "READY")
        self.assertEqual(data["session"]["protocol_id"], "HOME_DEMO_PROTOCOL_01")
        self.assertEqual(data["session"]["confidence_threshold"], 0.65)

    def test_03_api_protocol(self):
        """Verify /api/protocol returns active protocol steps."""
        response = self.client.get("/api/protocol")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_count"], 10)
        self.assertEqual(len(data["steps"]), 10)
        self.assertEqual(data["steps"][0]["expected_action"], "ACTION_IDLE")
        self.assertEqual(data["steps"][9]["expected_action"], "ACTION_CLOSE_BOX")

    def test_04_api_anomalies(self):
        """Verify /api/anomalies returns anomaly telemetry."""
        response = self.client.get("/api/anomalies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("current_anomaly_score", data)
        self.assertIn("fsm_status", data)

    def test_05_api_performance(self):
        """Verify /api/performance returns latency breakdown structure."""
        response = self.client.get("/api/performance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("breakdown_ms", data)
        self.assertIn("yolo_detector", data["breakdown_ms"])
        self.assertIn("pose_estimator", data["breakdown_ms"])
        self.assertIn("hand_tracker", data["breakdown_ms"])
        self.assertIn("hoi_analysis", data["breakdown_ms"])
        self.assertIn("har_inference", data["breakdown_ms"])

    def test_06_api_reset(self):
        """Verify /api/reset resets session state."""
        response = self.client.post("/api/reset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")

    def test_07_worker_lifecycle_on_video_mode(self):
        """Verify starting worker on demo video runs frames and updates telemetry."""
        video_path = "data/raw/videos/SES_001_NOMINAL.mp4"
        if not Path(video_path).exists():
            self.skipTest("Demo video file not found.")

        # Start worker on video
        worker.start(source=video_path, mode="video")
        self.assertTrue(worker.running)

        # Allow worker to initialize and process frames
        for _ in range(60):
            if worker.latest_jpeg_bytes is not None and worker.latest_metrics.get("status") == "ONLINE":
                break
            time.sleep(0.1)

        self.assertIsNotNone(worker.latest_jpeg_bytes)
        self.assertEqual(worker.latest_metrics["status"], "ONLINE")
        self.assertGreaterEqual(worker.latest_metrics["fps_actual"], 0.0)

        # Stop worker
        worker.stop()
        self.assertFalse(worker.running)
        self.assertEqual(worker.lifecycle_state, "MISSION STOPPED")

    def test_08_api_mission_summary(self):
        """Verify /api/mission_summary returns executive telemetry metrics."""
        response = self.client.get("/api/mission_summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("protocol_id", data)
        self.assertIn("session_id", data)
        self.assertIn("completed_steps", data)
        self.assertIn("total_steps", data)
        self.assertIn("completion_pct", data)
        self.assertIn("anomaly_count", data)
        self.assertIn("har_uncertain_count", data)
        self.assertIn("protocol_anomaly_count", data)
        self.assertIn("max_protocol_anomaly_score", data)
        self.assertIn("avg_confidence", data)
        self.assertIn("avg_fps", data)
        self.assertIn("avg_pipeline_latency_ms", data)
        self.assertIn("mission_status", data)

    def test_09_step_timestamps_in_protocol(self):
        """Verify /api/protocol includes step_timestamps dictionary."""
        response = self.client.get("/api/protocol")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("step_timestamps", data)

    def test_10_status_semantics_waiting_and_timeout(self):
        """Verify status semantics: WAITING on idle dwell, TIMEOUT on expiration, OK on expected."""
        sm = ProtocolStateMachine(protocol_json_path="config/experiment_protocols.json")
        sm.reset()

        # Step 1 expected is IDLE. When executing IDLE -> status is OK
        res = sm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=0.0, frame=0)
        self.assertEqual(res.status_text, "OK")

        # Advance Step 1 by dwelling 5 frames
        for f in range(1, 6):
            res = sm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=float(f)/30.0, frame=f)
        self.assertIn(1, sm.completed_steps)

        # Now Step 2 expected is ACTION_SANITIZE.
        # If user performs neutral ACTION_IDLE (waiting), status must be WAITING
        res_wait = sm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=5.0, frame=150)
        self.assertEqual(res_wait.status_text, "WAITING")
        self.assertTrue(res_wait.is_valid)

        # If user remains IDLE beyond step 2 max_duration_sec (e.g. 35.0s > 30.0s), status must be TIMEOUT
        res_timeout = sm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=40.0, frame=1200)
        self.assertEqual(res_timeout.status_text, "TIMEOUT")
        self.assertFalse(res_timeout.is_valid)

    def test_11_api_readiness(self):
        """Verify /api/readiness returns subsystem checklist status."""
        response = self.client.get("/api/readiness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("lifecycle_state", data)
        self.assertIn("checklist", data)
        self.assertTrue(data["checklist"]["object_detector"])
        self.assertTrue(data["checklist"]["har_model"])
        self.assertTrue(data["checklist"]["protocol_fsm"])
        self.assertTrue(data["all_ready"])

    def test_12_api_inject_anomaly(self):
        """Verify /api/demo/inject_anomaly accepts valid scenarios and rejects unknown ones."""
        # Valid scenario
        response = self.client.post("/api/demo/inject_anomaly", json={"scenario": "OUT_OF_ORDER_ACTION"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["injected_scenario"], "OUT_OF_ORDER_ACTION")
        self.assertEqual(data["mode"], "SIMULATED / INJECTED")

        # Invalid scenario
        bad_resp = self.client.post("/api/demo/inject_anomaly", json={"scenario": "INVALID_UNKNOWN_SCENARIO"})
        self.assertEqual(bad_resp.status_code, 400)

    def test_13_mission_lifecycle_transitions(self):
        """Verify mission lifecycle transitions across start, reset, and stop states."""
        worker.stop()
        self.assertEqual(worker.lifecycle_state, "MISSION STOPPED")
        worker.reset_session()
        self.assertEqual(worker.lifecycle_state, "PROTOCOL ARMED")

    def test_14_error_handling_invalid_source(self):
        """Verify starting pipeline on non-existent source gracefully sets SYSTEM ERROR."""
        with self.assertRaises(RuntimeError):
            worker.start(source="non_existent_file_xyz.mp4", mode="video")
        self.assertEqual(worker.lifecycle_state, "SYSTEM ERROR")
        self.assertIsNotNone(worker.error_message)
        # Recover via reset
        worker.reset_session()
        self.assertEqual(worker.lifecycle_state, "PROTOCOL ARMED")


if __name__ == "__main__":
    unittest.main()
