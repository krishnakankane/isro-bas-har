"""Unit and integration tests for Phase 10 Intelligent Protocol Validation and Anomaly Detection."""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.models import (
    AnomalyType,
    ExperimentStep,
    ProtocolDefinition,
    ValidationResult,
)
from src.core.state_machine import ProtocolStateMachine
from scripts.validate_protocol import validate_from_manifest, validate_from_features


class TestProtocolValidationSuite(unittest.TestCase):
    """Comprehensive test suite for deterministic protocol sequence state machine and anomaly scoring."""

    def setUp(self):
        self.protocol_path = Path("config/experiment_protocols.json")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp_dir.name)

        self.fsm = ProtocolStateMachine(
            protocol_json_path=self.protocol_path,
            min_dwell_frames=3,
            min_confidence=0.65,
            log_dir=self.log_dir,
            session_id="TEST_SES_01",
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_nominal_protocol_advancement_and_completion(self):
        """Verify executing all 10 steps in sequence advances and completes the protocol."""
        actions = [
            "ACTION_IDLE",
            "ACTION_SANITIZE",
            "ACTION_HOLD_BOTTLE",
            "ACTION_PLACE_BOTTLE",
            "ACTION_HOLD_BOX",
            "ACTION_OPEN_BOX",
            "ACTION_PICK_OBJECT",
            "ACTION_TRANSFER_OBJECT",
            "ACTION_RETURN_OBJECT",
            "ACTION_CLOSE_BOX",
        ]

        frame_idx = 0
        for step_idx, act in enumerate(actions, 1):
            # Feed dwell frames
            for _ in range(3):
                frame_idx += 1
                t_sec = float(frame_idx) / 30.0
                res = self.fsm.evaluate_action(act, confidence=0.95, timestamp=t_sec, frame=frame_idx)
                self.assertTrue(res.is_valid)
                self.assertEqual(res.anomaly_score, 0.0)

            self.assertIn(step_idx, self.fsm.completed_steps)

        self.assertTrue(self.fsm.is_completed)
        self.assertEqual(len(self.fsm.completed_steps), 10)
        self.assertEqual(len(self.fsm.anomaly_history), 0)

    def test_02_low_confidence_gating(self):
        """Verify predictions below min_confidence (0.65) do not advance the protocol."""
        # Current expected step is Step 1 (ACTION_IDLE)
        # Feed 10 frames of low confidence
        for f in range(10):
            res = self.fsm.evaluate_action("ACTION_IDLE", confidence=0.40, timestamp=f * 0.1, frame=f)
            self.assertFalse(res.is_valid)
            self.assertEqual(res.anomaly_type, AnomalyType.LOW_CONFIDENCE)
            self.assertEqual(res.status_text, "UNCERTAIN")
            self.assertGreater(res.anomaly_score, 0.0)
            self.assertLessEqual(res.anomaly_score, 0.25)

        # Step 1 should NOT be completed
        self.assertNotIn(1, self.fsm.completed_steps)
        self.assertEqual(self.fsm.current_step_idx, 0)

    def test_03_out_of_order_and_skipped_step(self):
        """Verify jumping ahead (e.g. Step 4 PLACE_BOTTLE when Step 1 IDLE is expected) triggers OUT_OF_ORDER_ACTION."""
        res = self.fsm.evaluate_action("ACTION_PLACE_BOTTLE", confidence=0.90, timestamp=1.0, frame=30)

        self.assertFalse(res.is_valid)
        self.assertEqual(res.anomaly_type, AnomalyType.OUT_OF_ORDER_ACTION)
        self.assertEqual(res.status_text, "ANOMALY")
        self.assertAlmostEqual(res.anomaly_score, 0.70)
        self.assertIn("Out-of-order", res.anomaly_message)
        self.assertEqual(len(self.fsm.anomaly_history), 1)

    def test_04_unexpected_alien_action(self):
        """Verify unrecognized actions not in protocol trigger UNEXPECTED_ACTION anomaly."""
        res = self.fsm.evaluate_action("ACTION_JUMPING_JACKS", confidence=0.95, timestamp=1.0, frame=30)

        self.assertFalse(res.is_valid)
        self.assertEqual(res.anomaly_type, AnomalyType.UNEXPECTED_ACTION)
        self.assertEqual(res.status_text, "ANOMALY")
        self.assertAlmostEqual(res.anomaly_score, 0.85)
        self.assertIn("Unexpected action", res.anomaly_message)

    def test_05_repeated_completed_action(self):
        """Verify repeating an already completed step after transitioning away triggers REPEATED_ACTION."""
        # 1. Complete Step 1 (IDLE)
        for f in range(3):
            self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
        self.assertIn(1, self.fsm.completed_steps)

        # 2. Advance to Step 2 (SANITIZE)
        for f in range(3, 6):
            self.fsm.evaluate_action("ACTION_SANITIZE", confidence=0.95, timestamp=f * 0.1, frame=f)
        self.assertIn(2, self.fsm.completed_steps)

        # 3. Advance to Step 3 (HOLD_BOTTLE)
        for f in range(6, 9):
            self.fsm.evaluate_action("ACTION_HOLD_BOTTLE", confidence=0.95, timestamp=f * 0.1, frame=f)
        self.assertIn(3, self.fsm.completed_steps)

        # 4. Now perform Step 2 (SANITIZE) again while Step 4 (PLACE_BOTTLE) is expected
        res = self.fsm.evaluate_action("ACTION_SANITIZE", confidence=0.95, timestamp=1.0, frame=30)

        self.assertFalse(res.is_valid)
        self.assertEqual(res.anomaly_type, AnomalyType.REPEATED_ACTION)
        self.assertEqual(res.status_text, "ANOMALY")
        self.assertAlmostEqual(res.anomaly_score, 0.30)
        self.assertIn("Repeated step", res.anomaly_message)

    def test_06_completion_idempotency(self):
        """Verify state machine remains in COMPLETED state on subsequent frames without duplicate logging."""
        actions = [
            "ACTION_IDLE", "ACTION_SANITIZE", "ACTION_HOLD_BOTTLE", "ACTION_PLACE_BOTTLE",
            "ACTION_HOLD_BOX", "ACTION_OPEN_BOX", "ACTION_PICK_OBJECT", "ACTION_TRANSFER_OBJECT",
            "ACTION_RETURN_OBJECT", "ACTION_CLOSE_BOX",
        ]
        f_idx = 0
        for act in actions:
            for _ in range(3):
                f_idx += 1
                self.fsm.evaluate_action(act, confidence=0.95, timestamp=f_idx * 0.1, frame=f_idx)

        self.assertTrue(self.fsm.is_completed)
        comp_time = self.fsm.completion_timestamp
        self.assertIsNotNone(comp_time)

        # Feed 10 more frames of CLOSE_BOX or IDLE after completion
        for _ in range(10):
            f_idx += 1
            res = self.fsm.evaluate_action("ACTION_CLOSE_BOX", confidence=0.95, timestamp=f_idx * 0.1, frame=f_idx)
            self.assertTrue(res.is_valid)
            self.assertTrue(res.is_completed)
            self.assertEqual(res.status_text, "COMPLETED")
            self.assertEqual(res.anomaly_score, 0.0)

        # Completion timestamp should remain stable
        self.assertEqual(self.fsm.completion_timestamp, comp_time)

    def test_07_anomaly_score_bounds(self):
        """Verify anomaly score is strictly bounded in [0.0, 1.0] across all conditions."""
        res_ok = self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=0.1, frame=3)
        self.assertEqual(res_ok.anomaly_score, 0.0)

        res_low = self.fsm.evaluate_action("ACTION_IDLE", confidence=0.10, timestamp=0.2, frame=6)
        self.assertTrue(0.0 <= res_low.anomaly_score <= 1.0)

        res_unexp = self.fsm.evaluate_action("UNKNOWN_ACTION", confidence=0.90, timestamp=0.3, frame=9)
        self.assertTrue(0.0 <= res_unexp.anomaly_score <= 1.0)

    def test_08_jsonl_event_audit_logging(self):
        """Verify structured JSONL events are created and written to log file."""
        self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=0.1, frame=3)
        self.fsm.evaluate_action("ACTION_UNKNOWN", confidence=0.90, timestamp=0.2, frame=6)

        log_path = self.fsm.log_file
        self.assertTrue(log_path.exists())

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertGreaterEqual(len(lines), 2)
        event_types = [e.get("event_type") for e in lines]
        self.assertIn("TRANSITION", event_types)
        self.assertIn("ANOMALY", event_types)

    def test_09_reset_and_session_reinitialization(self):
        """Verify reset clears completed steps, histories, and completion status."""
        for f in range(3):
            self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
        self.assertEqual(len(self.fsm.completed_steps), 1)

        self.fsm.reset()
        self.assertEqual(len(self.fsm.completed_steps), 0)
        self.assertEqual(self.fsm.current_step_idx, 0)
        self.assertFalse(self.fsm.is_completed)
        self.assertEqual(len(self.fsm.transition_history), 0)
        self.assertEqual(len(self.fsm.anomaly_history), 0)

    def test_10_validate_protocol_offline_manifest(self):
        """Verify offline validation script executes successfully on real manifest."""
        manifest_p = Path("data/temporal_har/manifests/SES_001_NOMINAL_actions.json")
        if not manifest_p.exists():
            self.skipTest("Real manifest not found.")

        report = validate_from_manifest(
            manifest_path=manifest_p,
            protocol_path=self.protocol_path,
            log_dir=self.log_dir,
        )

        self.assertEqual(report["completed_steps_count"], 10)
        self.assertTrue(report["is_completed"])
        self.assertEqual(report["anomalies_detected_count"], 0)

    def test_11_confidence_boundary_precision(self):
        """Verify fine-grained threshold gating around 0.65 (0.59, 0.60, 0.64, 0.65, 0.66)."""
        below_thresh = [0.59, 0.60, 0.64]
        for val in below_thresh:
            res = self.fsm.evaluate_action("ACTION_IDLE", confidence=val, timestamp=0.1, frame=1)
            self.assertFalse(res.is_valid)
            self.assertEqual(res.anomaly_type, AnomalyType.LOW_CONFIDENCE)
            self.assertEqual(res.status_text, "UNCERTAIN")

        # Threshold and above
        for val in [0.65, 0.66]:
            self.fsm.reset()
            res = self.fsm.evaluate_action("ACTION_IDLE", confidence=val, timestamp=0.1, frame=1)
            self.assertTrue(res.is_valid)
            self.assertEqual(res.anomaly_type, AnomalyType.NONE)
            self.assertEqual(res.status_text, "OK")

    def test_12_premature_completion_rejection(self):
        """Verify attempting ACTION_CLOSE_BOX prematurely is blocked and flagged as OUT_OF_ORDER_ACTION."""
        res = self.fsm.evaluate_action("ACTION_CLOSE_BOX", confidence=0.95, timestamp=1.0, frame=30)
        self.assertFalse(res.is_valid)
        self.assertFalse(self.fsm.is_completed)
        self.assertIn(res.anomaly_type, (AnomalyType.OUT_OF_ORDER_ACTION, AnomalyType.SKIPPED_STEP))
        self.assertEqual(res.status_text, "ANOMALY")
        self.assertAlmostEqual(res.anomaly_score, 0.70)

    def test_13_missing_action_timeout(self):
        """Verify exceeding step max_duration_sec in idle triggers TIMEOUT_EXCEEDED."""
        # Complete Step 1 IDLE
        for f in range(3):
            self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
        self.assertIn(1, self.fsm.completed_steps)

        # Now in Step 2 SANITIZE (max_duration_sec = 30.0)
        # Advance clock to 35.0s in idle
        res = self.fsm.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=36.0, frame=1080)
        self.assertFalse(res.is_valid)
        self.assertIn(res.anomaly_type, (AnomalyType.TIMEOUT_EXCEEDED, AnomalyType.MISSING_EXPECTED_ACTION))
        self.assertEqual(res.status_text, "TIMEOUT")
        self.assertAlmostEqual(res.anomaly_score, 0.75)

    def test_14_deterministic_scoring_reproducibility(self):
        """Verify that identical sequences and inputs produce identical anomaly scores and messages."""
        fsm1 = ProtocolStateMachine(protocol_json_path=self.protocol_path, min_dwell_frames=3, min_confidence=0.65)
        fsm2 = ProtocolStateMachine(protocol_json_path=self.protocol_path, min_dwell_frames=3, min_confidence=0.65)

        res1 = fsm1.evaluate_action("ACTION_TRANSFER_OBJECT", confidence=0.88, timestamp=2.5, frame=75)
        res2 = fsm2.evaluate_action("ACTION_TRANSFER_OBJECT", confidence=0.88, timestamp=2.5, frame=75)

        self.assertEqual(res1.anomaly_type, res2.anomaly_type)
        self.assertEqual(res1.anomaly_score, res2.anomaly_score)
        self.assertEqual(res1.anomaly_message, res2.anomaly_message)
        self.assertEqual(res1.status_text, res2.status_text)

    def test_15_adversarial_battery_runner(self):
        """Verify full adversarial scenario battery executes and all scenarios pass."""
        from scripts.adversarial_protocol_test import run_adversarial_battery
        report_p = self.log_dir / "adv_report.json"
        rep = run_adversarial_battery(
            protocol_path=self.protocol_path,
            output_report_path=report_p,
            min_confidence=0.65,
        )
        self.assertTrue(rep["all_scenarios_passed"])
        self.assertEqual(rep["total_scenarios_passed"], 9)


if __name__ == "__main__":
    unittest.main()
