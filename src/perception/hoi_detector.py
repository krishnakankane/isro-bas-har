"""Hand-Object Interaction (HOI) Engine for ISRO BAS HAR.

Computes spatial relationships between detected hands and objects across frames,
maintaining temporal persistence to classify interaction states.

State Transitions
-----------------
UNKNOWN  ──► APPROACHING  (hand enters proximity zone)
APPROACHING ──► GRASPING  (hand is_grasping AND within contact threshold)
GRASPING ──► MANIPULATING (grasp sustained for dwell_frames_threshold frames)
MANIPULATING ──► RELEASED  (hand leaves object or grasp ends)
RELEASED ──► UNKNOWN       (object lost or hand leaves scene)

Design Notes
------------
- Bounding-box proximity NEVER claims physical contact. The APPROACHING state
  means the hand wrist is within `approach_dist_norm` normalized units of the
  object center. Physical contact is NOT inferred.
- All thresholds are in normalized [0, 1] image coordinates so the engine
  remains resolution-agnostic.
- Temporal persistence (dwell) is tracked per (hand_side, object_name) pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.interfaces import HOIDetectorInterface
from ..core.models import (
    BoundingBox,
    DetectedObject,
    HandTrackingResult,
    InteractionEvent,
    InteractionState,
)


# ─── Configurable HOI Parameters ─────────────────────────────────────────────

@dataclass
class HOIConfig:
    """All thresholds and knobs for HOI state-machine logic.

    All spatial thresholds are in normalized image coordinates [0.0, 1.0].
    This makes the engine independent of camera resolution.
    """
    # Hand wrist is within this distance of object center → APPROACHING
    approach_dist_norm: float = 0.25

    # Hand wrist is within this distance AND is_grasping → GRASPING
    contact_dist_norm: float = 0.12

    # Number of consecutive GRASPING frames before → MANIPULATING
    dwell_frames_to_manipulate: int = 8

    # Number of consecutive non-grasping frames before → RELEASED from MANIPULATING
    release_frames_threshold: int = 4

    # Maximum number of tracked (hand, object) pairs (guards memory)
    max_tracked_pairs: int = 16


# ─── Persistent State Tracker ────────────────────────────────────────────────

@dataclass
class _PairState:
    """Internal per-(hand_side, object_name) temporal state record."""
    current_state: InteractionState = InteractionState.UNKNOWN
    grasping_dwell: int = 0          # consecutive frames in GRASPING
    release_dwell: int = 0           # consecutive frames without grasp
    total_dwell_frames: int = 0      # total frames in current state
    last_proximity_dist: float = 1.0 # last measured wrist-to-center dist (norm)


# ─── HOI Detector ────────────────────────────────────────────────────────────

class HOIDetector(HOIDetectorInterface):
    """Geometric spatial proximity and temporal interaction-state detector.

    Uses normalized wrist-to-object-center distance as the spatial signal.
    Maintains per-(hand, object) state machines across frames to produce
    temporally stable InteractionEvent outputs.

    The detector is deliberately conservative:
      - APPROACHING means the hand is *near* the object.
      - GRASPING means the hand is *very near* AND the hand tracker reports
        a pinch/grasp gesture. It does NOT mean physical contact is confirmed.
      - MANIPULATING means GRASPING has been sustained for `dwell_frames_to_manipulate`
        consecutive frames, indicating intentional interaction.
    """

    def __init__(self, config: Optional[HOIConfig] = None):
        """Initialise HOI Detector.

        Args:
            config: HOIConfig instance. Uses defaults if None.
        """
        self.config = config or HOIConfig()
        # Keyed by (hand_side: str, object_name: str)
        self._pair_states: Dict[Tuple[str, str], _PairState] = {}

    def reset(self) -> None:
        """Clear all persistent interaction state (call between experiments)."""
        self._pair_states.clear()

    def analyze_interaction(
        self,
        objects: List[DetectedObject],
        left_hand: Optional[HandTrackingResult],
        right_hand: Optional[HandTrackingResult],
    ) -> List[InteractionEvent]:
        """Compute HOI events for all hand-object pairs this frame.

        Args:
            objects:    Detected objects from YOLO (normalized bbox expected).
            left_hand:  Left hand tracking result (or None).
            right_hand: Right hand tracking result (or None).

        Returns:
            List of InteractionEvent, one per active hand-object pair.
            Pairs with state UNKNOWN and distance > approach_dist are omitted.
        """
        hands: List[HandTrackingResult] = []
        if left_hand and left_hand.is_detected:
            hands.append(left_hand)
        if right_hand and right_hand.is_detected:
            hands.append(right_hand)

        active_keys: set = set()
        events: List[InteractionEvent] = []

        for hand in hands:
            wrist_x, wrist_y = hand.wrist_pos  # normalized [0, 1]

            for obj in objects:
                key = (hand.handedness, obj.class_name)
                active_keys.add(key)

                # Guard maximum tracked pairs
                if key not in self._pair_states:
                    if len(self._pair_states) >= self.config.max_tracked_pairs:
                        continue
                    self._pair_states[key] = _PairState()

                pair = self._pair_states[key]

                # Compute normalised wrist-to-object-center distance
                cx, cy = obj.bbox.center
                # bbox.center is in pixel space; normalise using 1.0 (already
                # normalised if YOLO returns [0,1]) — but Ultralytics returns
                # pixel values. We convert using a unit-agnostic ratio.
                dist = _euclidean_dist_norm(
                    wrist_x, wrist_y, cx, cy, obj.bbox
                )
                pair.last_proximity_dist = dist

                # ── State machine ─────────────────────────────────────────
                new_state = self._next_state(pair, dist, hand.is_grasping)

                if new_state != pair.current_state:
                    pair.total_dwell_frames = 0
                pair.current_state = new_state
                pair.total_dwell_frames += 1

                # Emit event only for non-UNKNOWN active interactions
                if new_state != InteractionState.UNKNOWN:
                    events.append(
                        InteractionEvent(
                            hand_side=hand.handedness,
                            target_object_name=obj.class_name,
                            target_object_bbox=obj.bbox,
                            state=new_state,
                            proximity_distance_px=dist,  # stored as norm dist
                            duration_frames=pair.total_dwell_frames,
                        )
                    )

        # ── Age-out inactive pairs ────────────────────────────────────────
        # Pairs not seen this frame get their state reset after release threshold
        inactive_keys = set(self._pair_states.keys()) - active_keys
        for key in inactive_keys:
            pair = self._pair_states[key]
            pair.release_dwell += 1
            if pair.release_dwell >= self.config.release_frames_threshold:
                pair.current_state = InteractionState.UNKNOWN
                pair.grasping_dwell = 0
                pair.total_dwell_frames = 0

        return events

    # ── Internal helpers ──────────────────────────────────────────────────

    def _next_state(
        self,
        pair: _PairState,
        dist: float,
        is_grasping: bool,
    ) -> InteractionState:
        """Compute the next interaction state for a single hand-object pair."""
        cfg = self.config
        current = pair.current_state

        # Reset release counter when hand is active near the object
        if dist < cfg.approach_dist_norm:
            pair.release_dwell = 0
        else:
            pair.release_dwell += 1

        # ── Transition logic ──────────────────────────────────────────────

        if dist >= cfg.approach_dist_norm:
            # Too far — check if we should release from sustained states
            if current in (InteractionState.GRASPING, InteractionState.MANIPULATING):
                if pair.release_dwell >= cfg.release_frames_threshold:
                    pair.grasping_dwell = 0
                    return InteractionState.RELEASED
                return current  # Hold state briefly during occlusion
            if current == InteractionState.RELEASED:
                return InteractionState.UNKNOWN
            return InteractionState.UNKNOWN

        # Within approach zone
        if dist < cfg.contact_dist_norm and is_grasping:
            pair.grasping_dwell += 1
            if pair.grasping_dwell >= cfg.dwell_frames_to_manipulate:
                return InteractionState.MANIPULATING
            return InteractionState.GRASPING
        else:
            # Not grasping or not close enough for contact
            pair.grasping_dwell = max(0, pair.grasping_dwell - 1)
            if current == InteractionState.MANIPULATING:
                # Transition out on sustained non-grasp
                if pair.release_dwell >= cfg.release_frames_threshold:
                    return InteractionState.RELEASED
                return InteractionState.MANIPULATING
            if current == InteractionState.GRASPING:
                return InteractionState.APPROACHING
            return InteractionState.APPROACHING

    def get_active_pairs(self) -> Dict[Tuple[str, str], _PairState]:
        """Return a snapshot of all currently tracked (hand, object) states."""
        return dict(self._pair_states)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _euclidean_dist_norm(
    wx: float, wy: float,
    cx: float, cy: float,
    bbox: BoundingBox,
) -> float:
    """Return Euclidean distance in normalised bbox-diagonal units.

    Converts the pixel-space object centre into normalized coordinates using
    the object's own bounding-box diagonal as a scale reference, making the
    distance invariant to frame resolution and object size.

    - wx, wy : wrist position, already in [0, 1] normalised coords
    - cx, cy : object bbox centre, in the same coordinate space as wrist
      (both come from the same normalised detection pipeline)

    Returns a value in [0, ∞) where < 0.12 ≈ very close (contact zone),
    < 0.25 ≈ approaching, > 0.25 ≈ far.
    """
    dx = wx - cx
    dy = wy - cy
    return math.sqrt(dx * dx + dy * dy)
