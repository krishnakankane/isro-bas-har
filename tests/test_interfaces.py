"""Unit tests for Protocol State Machine and Core Data Models."""

import unittest
from pathlib import Path
from src.core.models import (
    BoundingBox,
    DetectedObject,
    AnomalyType,
    ValidationResult,
)
from src.core.state_machine import ProtocolStateMachine
from config.settings import CONFIG_DIR


class TestProtocolStateMachine(unittest.TestCase):
    """Test suite for protocol sequence verification."""

    def setUp(self):
        protocol_path = CONFIG_DIR / "experiment_protocols.json"
        self.fsm = ProtocolStateMachine(str(protocol_path))

    def test_protocol_loaded(self):
        """Verify protocol was loaded with all 10 steps."""
        self.assertIsNotNone(self.fsm.protocol)
        self.assertEqual(len(self.fsm.protocol.steps), 10)
        self.assertEqual(self.fsm.protocol.steps[0].name, "IDLE")

    def test_nominal_step_sequence(self):
        """Verify nominal sequential step execution."""
        expected = self.fsm.get_expected_step()
        self.assertEqual(expected.step_id, 1)

        # Trigger step 1
        res = self.fsm.evaluate_action("ACTION_IDLE", 0.95, 1.0)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.current_step_id, 1)

    def test_out_of_sequence_and_skipped_step(self):
        """Verify that jumping directly to Step 4 triggers an out-of-order / skipped step anomaly."""
        res = self.fsm.evaluate_action("ACTION_PLACE_BOTTLE", 0.92, 1.0)
        self.assertFalse(res.is_valid)
        self.assertIn(res.anomaly_type, (AnomalyType.SKIPPED_STEP, AnomalyType.OUT_OF_ORDER_ACTION))
        self.assertEqual(res.suggested_next_step_id, 1)


class TestBoundingBox(unittest.TestCase):
    """Test geometric bounding box calculations."""

    def test_bbox_center_and_area(self):
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0)
        self.assertEqual(bbox.center, (30.0, 50.0))
        self.assertEqual(bbox.area, 2400.0)


if __name__ == "__main__":
    unittest.main()
