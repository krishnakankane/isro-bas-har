"""Unit and integration tests for HOI Engine, Feature Extractor, and Temporal Buffer.

Covers:
- HOI state transitions (UNKNOWN → APPROACHING → GRASPING → MANIPULATING → RELEASED)
- Distance calculations
- Missing hand / missing object handling
- Temporal persistence and dwell counters
- Feature vector dimensional consistency (154)
- Sliding-window behaviour (fill, eviction, zero-padding)
- Reset behaviour (HOI engine + buffer)
- Feature recording (JSONL output)
"""

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.core.models import (
    BoundingBox,
    DetectedObject,
    FramePerceptionResult,
    HandLandmark,
    HandTrackingResult,
    InteractionState,
    PoseEstimationResult,
    PoseLandmark,
    TemporalFeatureVector,
)
from src.perception.hoi_detector import HOIConfig, HOIDetector, _euclidean_dist_norm
from src.activity_engine.feature_extractor import (
    FEATURE_DIM,
    PerceptionFeatureExtractor,
)
from src.activity_engine.temporal_buffer import TemporalFeatureBuffer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_object(
    name: str = "bottle",
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.1,
    h: float = 0.2,
    conf: float = 0.85,
    cls_id: int = 39,
) -> DetectedObject:
    """Create a DetectedObject at a given normalised centre with given size."""
    return DetectedObject(
        class_id=cls_id,
        class_name=name,
        confidence=conf,
        bbox=BoundingBox(
            xmin=cx - w / 2,
            ymin=cy - h / 2,
            xmax=cx + w / 2,
            ymax=cy + h / 2,
        ),
    )


def _make_hand(
    side: str = "Right",
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    is_detected: bool = True,
    is_grasping: bool = False,
) -> HandTrackingResult:
    """Create a minimal HandTrackingResult with a single wrist landmark."""
    lms = [HandLandmark(id=i, x=wrist_x + i * 0.01, y=wrist_y, z=0.0) for i in range(21)]
    lms[0] = HandLandmark(id=0, x=wrist_x, y=wrist_y, z=0.0)
    return HandTrackingResult(
        handedness=side,
        landmarks=lms,
        wrist_pos=(wrist_x, wrist_y),
        is_grasping=is_grasping,
        is_detected=is_detected,
    )


def _make_pose(detected: bool = True) -> PoseEstimationResult:
    """Create a synthetic 33-landmark pose."""
    lms = [
        PoseLandmark(id=i, name=f"lm_{i}", x=0.5, y=float(i) / 33.0, z=0.0, visibility=0.9)
        for i in range(33)
    ]
    return PoseEstimationResult(landmarks=lms, torso_angle=15.0, is_detected=detected)


def _make_frame_result(
    frame_id: int = 0,
    objects=None,
    pose=None,
    left_hand=None,
    right_hand=None,
    interactions=None,
) -> FramePerceptionResult:
    return FramePerceptionResult(
        frame_id=frame_id,
        timestamp_sec=time.time(),
        objects=objects or [],
        body_pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        interactions=interactions or [],
    )


# ─── HOI Engine Tests ─────────────────────────────────────────────────────────

class TestHOIDistanceCalculation(unittest.TestCase):
    """Unit tests for the geometric distance helper."""

    def test_zero_distance_when_coincident(self):
        bbox = BoundingBox(0.4, 0.4, 0.6, 0.6)  # centre = (0.5, 0.5)
        dist = _euclidean_dist_norm(0.5, 0.5, 0.5, 0.5, bbox)
        self.assertAlmostEqual(dist, 0.0, places=6)

    def test_known_distance(self):
        bbox = BoundingBox(0.4, 0.4, 0.6, 0.6)
        # wrist at (0.2, 0.2) → dist = sqrt((0.3)^2 + (0.3)^2) ≈ 0.4243
        dist = _euclidean_dist_norm(0.2, 0.2, 0.5, 0.5, bbox)
        expected = math.sqrt(0.3 ** 2 + 0.3 ** 2)
        self.assertAlmostEqual(dist, expected, places=5)

    def test_distance_is_always_non_negative(self):
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        for _ in range(20):
            wx, wy = np.random.rand(2)
            cx, cy = np.random.rand(2)
            dist = _euclidean_dist_norm(wx, wy, cx, cy, bbox)
            self.assertGreaterEqual(dist, 0.0)


class TestHOIStateTransitions(unittest.TestCase):
    """Test state machine transitions across simulated frames."""

    def setUp(self):
        cfg = HOIConfig(
            approach_dist_norm=0.25,
            contact_dist_norm=0.12,
            dwell_frames_to_manipulate=5,
            release_frames_threshold=3,
        )
        self.hoi = HOIDetector(config=cfg)
        self.obj = _make_object(cx=0.5, cy=0.5)

    def _run_frames(self, n: int, wrist_x: float, wrist_y: float, grasping: bool):
        """Simulate n identical frames and return last events."""
        hand = _make_hand("Right", wrist_x=wrist_x, wrist_y=wrist_y, is_grasping=grasping)
        events = []
        for _ in range(n):
            events = self.hoi.analyze_interaction([self.obj], None, hand)
        return events

    def test_far_hand_returns_no_events(self):
        """Hand far from object → no interaction events (UNKNOWN state not emitted)."""
        events = self._run_frames(1, wrist_x=0.9, wrist_y=0.9, grasping=False)
        self.assertEqual(events, [])

    def test_approaching_state_on_proximity(self):
        """Hand enters approach zone without grasping → APPROACHING."""
        events = self._run_frames(1, wrist_x=0.62, wrist_y=0.5, grasping=False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, InteractionState.APPROACHING)

    def test_grasping_state_on_contact_with_grasp(self):
        """Hand enters contact zone AND grasping=True → GRASPING."""
        events = self._run_frames(1, wrist_x=0.55, wrist_y=0.5, grasping=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, InteractionState.GRASPING)

    def test_manipulating_after_dwell(self):
        """GRASPING sustained for dwell_frames → MANIPULATING."""
        events = self._run_frames(10, wrist_x=0.54, wrist_y=0.5, grasping=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].state, InteractionState.MANIPULATING)

    def test_released_after_leaving(self):
        """After GRASPING, move hand far away → eventually UNKNOWN / no event emitted."""
        # First establish grasping
        self._run_frames(5, wrist_x=0.54, wrist_y=0.5, grasping=True)
        # Then move far using correct kwarg
        hand_far = _make_hand("Right", wrist_x=0.9, wrist_y=0.9, is_grasping=False)
        for _ in range(4):  # > release_frames_threshold=3
            events = self.hoi.analyze_interaction([self.obj], None, hand_far)
        # Hand is far — no event emitted for UNKNOWN state
        self.assertEqual(events, [])

    def test_reset_clears_all_states(self):
        """Reset wipes all persistent pair states."""
        self._run_frames(10, wrist_x=0.54, wrist_y=0.5, grasping=True)
        self.hoi.reset()
        self.assertEqual(len(self.hoi.get_active_pairs()), 0)

    def test_duration_frames_increases_with_dwell(self):
        """duration_frames should be positive and generally increasing during GRASPING/MANIPULATING."""
        hand = _make_hand("Right", wrist_x=0.54, wrist_y=0.5, is_grasping=True)
        durations = []
        for _ in range(15):
            evts = self.hoi.analyze_interaction([self.obj], None, hand)
            if evts:
                durations.append(evts[0].duration_frames)
        # Must have observed some frames, and the final dwell should be > 1
        self.assertGreater(len(durations), 0, "Expected at least 1 event with duration_frames")
        self.assertGreater(durations[-1], 1, "Expected duration to grow beyond 1 frame")


class TestHOIMissingInputs(unittest.TestCase):
    """HOI engine should handle missing hands/objects gracefully."""

    def setUp(self):
        self.hoi = HOIDetector()

    def test_no_hands_returns_empty(self):
        obj = _make_object()
        events = self.hoi.analyze_interaction([obj], None, None)
        self.assertEqual(events, [])

    def test_no_objects_returns_empty(self):
        hand = _make_hand()
        events = self.hoi.analyze_interaction([], hand, None)
        self.assertEqual(events, [])

    def test_both_none_returns_empty(self):
        events = self.hoi.analyze_interaction([], None, None)
        self.assertEqual(events, [])

    def test_undetected_hand_returns_empty(self):
        obj = _make_object()
        hand = _make_hand(is_detected=False)
        events = self.hoi.analyze_interaction([obj], hand, None)
        self.assertEqual(events, [])

    def test_dual_hand_tracking(self):
        """Both hands can independently track the same object."""
        obj = _make_object(cx=0.5, cy=0.5)
        left  = _make_hand("Left",  wrist_x=0.58, wrist_y=0.5)
        right = _make_hand("Right", wrist_x=0.62, wrist_y=0.5)
        events = self.hoi.analyze_interaction([obj], left, right)
        hand_sides = {e.hand_side for e in events}
        self.assertIn("Left", hand_sides)
        self.assertIn("Right", hand_sides)


# ─── Feature Extractor Tests ──────────────────────────────────────────────────

class TestPerceptionFeatureExtractor(unittest.TestCase):
    """Test feature vector dimension and slot correctness."""

    def setUp(self):
        self.ext = PerceptionFeatureExtractor()

    def test_feature_dim_constant(self):
        self.assertEqual(FEATURE_DIM, 154)

    def test_output_shape_empty_perception(self):
        """Empty perception (no detections) → zeros-filled vector of correct dim."""
        fr = _make_frame_result()
        fv = self.ext.extract(fr)
        self.assertIsInstance(fv, TemporalFeatureVector)
        self.assertEqual(fv.data.shape, (FEATURE_DIM,))
        self.assertEqual(fv.data.dtype, np.float32)

    def test_pose_slots_filled(self):
        """Slot 0–98 (x,y,z) and 99–131 (visibility) should be non-zero for a detected pose."""
        pose = _make_pose(detected=True)
        fr = _make_frame_result(pose=pose)
        fv = self.ext.extract(fr)
        # At least some pose slots must be non-zero
        self.assertTrue(np.any(fv.data[0:99] != 0.0))
        self.assertTrue(np.any(fv.data[99:132] != 0.0))

    def test_no_pose_detection_gives_zero_slots(self):
        """Non-detected pose → slots 0–131 must be zero."""
        pose = _make_pose(detected=False)
        fr = _make_frame_result(pose=pose)
        fv = self.ext.extract(fr)
        self.assertTrue(np.all(fv.data[0:132] == 0.0))

    def test_presence_flags_correct(self):
        """Slots 138-140 indicate pose/left/right detection presence."""
        pose  = _make_pose(detected=True)
        left  = _make_hand("Left",  is_detected=True)
        right = _make_hand("Right", is_detected=False)
        fr = _make_frame_result(pose=pose, left_hand=left, right_hand=right)
        fv = self.ext.extract(fr)
        self.assertAlmostEqual(fv.data[138], 1.0)   # pose detected
        self.assertAlmostEqual(fv.data[139], 1.0)   # left detected
        self.assertAlmostEqual(fv.data[140], 0.0)   # right NOT detected

    def test_hoi_state_one_hot_approaching(self):
        """APPROACHING state → one-hot slot 150 (index 2 in HOI block starting at 148)."""
        from src.core.models import InteractionEvent
        evt = InteractionEvent(
            hand_side="Right",
            target_object_name="bottle",
            target_object_bbox=BoundingBox(0.4, 0.4, 0.6, 0.6),
            state=InteractionState.APPROACHING,
            proximity_distance_px=0.15,
            duration_frames=3,
        )
        fr = _make_frame_result(interactions=[evt])
        fv = self.ext.extract(fr)
        # APPROACHING is index 2 in _HOI_STATE_ORDER → slot 148+2 = 150
        self.assertAlmostEqual(fv.data[150], 1.0)
        # Slots 148 (NONE) and 149 (UNKNOWN) must be zero
        self.assertAlmostEqual(fv.data[148], 0.0, places=5)
        self.assertAlmostEqual(fv.data[149], 0.0, places=5)
        # Slot 151 (GRASPING) and 152 (MANIPULATING) must be zero
        self.assertAlmostEqual(fv.data[151], 0.0, places=5)
        self.assertAlmostEqual(fv.data[152], 0.0, places=5)
        # Slot 153 is dwell (shared), not tested for zero here — see FEATURE_SCHEMA.md note

    def test_no_objects_sentinel_distance(self):
        """Without objects, wrist-to-object distances default to 1.0 (sentinel)."""
        left = _make_hand("Left", wrist_x=0.3, wrist_y=0.3, is_detected=True)
        fr = _make_frame_result(left_hand=left)
        fv = self.ext.extract(fr)
        self.assertAlmostEqual(fv.data[146], 1.0)  # Left dist sentinel

    def test_feature_vector_is_all_finite(self):
        """No NaN or Inf values should appear in the feature vector."""
        pose  = _make_pose()
        left  = _make_hand("Left",  wrist_x=0.3, wrist_y=0.6, is_detected=True, is_grasping=True)
        right = _make_hand("Right", wrist_x=0.7, wrist_y=0.6, is_detected=True)
        obj   = _make_object(cx=0.5, cy=0.5)
        fr = _make_frame_result(pose=pose, left_hand=left, right_hand=right, objects=[obj])
        fv = self.ext.extract(fr)
        self.assertTrue(np.all(np.isfinite(fv.data)))

    def test_invalid_feature_dim_raises(self):
        with self.assertRaises(ValueError):
            PerceptionFeatureExtractor(feature_dim=99)

    def test_to_numpy_returns_copy(self):
        """to_numpy() should return an independent copy, not a view."""
        fr = _make_frame_result(pose=_make_pose())
        fv = self.ext.extract(fr)
        arr = fv.to_numpy()
        arr[0] = 999.0
        self.assertNotAlmostEqual(fv.data[0], 999.0)


# ─── Temporal Buffer Tests ────────────────────────────────────────────────────

class TestTemporalFeatureBuffer(unittest.TestCase):
    """Tests for the sliding-window temporal buffer."""

    def _make_fv(self, frame_id: int = 0, val: float = 1.0) -> TemporalFeatureVector:
        data = np.full(FEATURE_DIM, val, dtype=np.float32)
        return TemporalFeatureVector(data=data, frame_id=frame_id, timestamp_sec=time.time())

    def test_initial_state_empty(self):
        buf = TemporalFeatureBuffer(window_size=60)
        self.assertEqual(buf.current_length(), 0)
        self.assertFalse(buf.is_full())

    def test_push_and_length(self):
        buf = TemporalFeatureBuffer(window_size=60)
        for i in range(10):
            buf.push(self._make_fv(i))
        self.assertEqual(buf.current_length(), 10)

    def test_is_full_after_window_size_pushes(self):
        buf = TemporalFeatureBuffer(window_size=10)
        for i in range(10):
            buf.push(self._make_fv(i))
        self.assertTrue(buf.is_full())
        self.assertEqual(buf.current_length(), 10)

    def test_eviction_maintains_max_length(self):
        """Pushing beyond window_size should evict oldest."""
        buf = TemporalFeatureBuffer(window_size=5)
        for i in range(10):
            buf.push(self._make_fv(i))
        self.assertEqual(buf.current_length(), 5)

    def test_get_window_shape(self):
        buf = TemporalFeatureBuffer(window_size=60)
        for i in range(30):
            buf.push(self._make_fv(i))
        window = buf.get_window()
        self.assertEqual(window.shape, (60, FEATURE_DIM))
        self.assertEqual(window.dtype, np.float32)

    def test_zero_padding_for_incomplete_window(self):
        """Leading rows of an incomplete window should be zeros."""
        buf = TemporalFeatureBuffer(window_size=60)
        buf.push(self._make_fv(0, val=7.0))
        window = buf.get_window()
        # First 59 rows should be zero-padded
        self.assertTrue(np.all(window[:59, :] == 0.0))
        # Last row should contain the pushed vector
        self.assertTrue(np.all(window[59, :] == 7.0))

    def test_push_empty_adds_zero_vector(self):
        buf = TemporalFeatureBuffer(window_size=5)
        buf.push_empty()
        self.assertEqual(buf.current_length(), 1)
        window = buf.get_window()
        self.assertTrue(np.all(window[4, :] == 0.0))

    def test_reset_clears_buffer(self):
        buf = TemporalFeatureBuffer(window_size=60)
        for i in range(30):
            buf.push(self._make_fv(i))
        buf.reset()
        self.assertEqual(buf.current_length(), 0)
        window = buf.get_window()
        self.assertTrue(np.all(window == 0.0))

    def test_get_window_returns_independent_copy(self):
        """Modifying the returned window should not affect the buffer."""
        buf = TemporalFeatureBuffer(window_size=5)
        buf.push(self._make_fv(0, val=5.0))
        w1 = buf.get_window()
        w1[:] = 999.0
        w2 = buf.get_window()
        self.assertFalse(np.all(w2 == 999.0))

    def test_recording_writes_jsonl(self):
        """JSONL recording should produce parseable lines with correct keys."""
        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "features.jsonl"
            buf = TemporalFeatureBuffer(window_size=5, record_path=rec_path)
            for i in range(3):
                buf.push(self._make_fv(frame_id=i, val=float(i)))
            buf.disable_recording()

            lines = rec_path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 3)
            for i, line in enumerate(lines):
                rec = json.loads(line)
                self.assertIn("frame_id", rec)
                self.assertIn("timestamp_sec", rec)
                self.assertIn("data", rec)
                self.assertEqual(len(rec["data"]), FEATURE_DIM)
                self.assertEqual(rec["frame_id"], i)

    def test_dimension_mismatch_is_handled(self):
        """A vector with wrong dim should be zero-padded/truncated, not raise."""
        buf = TemporalFeatureBuffer(window_size=5)
        bad_fv = TemporalFeatureVector(
            data=np.ones(50, dtype=np.float32),
            frame_id=0,
            timestamp_sec=time.time(),
        )
        buf.push(bad_fv)  # Must not raise
        self.assertEqual(buf.current_length(), 1)


# ─── Integration: Demo pipeline imports cleanly ────────────────────────────────

class TestDemoImport(unittest.TestCase):
    def test_demo_module_importable(self):
        """The demo module must be importable without side effects."""
        import importlib
        mod = importlib.import_module("scripts.run_perception_demo")
        self.assertTrue(hasattr(mod, "run_demo"))


if __name__ == "__main__":
    unittest.main()
