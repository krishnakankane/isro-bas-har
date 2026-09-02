"""Structured Temporal Feature Extractor for ISRO BAS HAR.

Converts one frame's full perception output (pose + hands + objects + HOI)
into a fixed-dimension numpy feature vector suitable for temporal classification.

Feature Vector Schema (FEATURE_DIM = 154)
==========================================
The schema is deliberately fixed and documented so that the future temporal
HAR model always receives the same-shaped input regardless of how many objects
or hands are detected in a given frame. Missing detections are encoded as zeros,
making the absence of evidence itself an informative signal.

Slot Layout
-----------
[  0 –  98]  Body pose landmarks: 33 × (x, y, z) — normalised [0, 1]
[ 99 – 131]  Body pose visibility: 33 × visibility — normalised [0, 1]
[132 – 133]  Left wrist (x, y) — normalised [0, 1]; zeros if no left hand
[134 – 135]  Right wrist (x, y) — normalised [0, 1]; zeros if no right hand
[136]         Left hand is_grasping — float (0.0 or 1.0)
[137]         Right hand is_grasping — float (0.0 or 1.0)
[138]         Pose detected — float (0.0 or 1.0)
[139]         Left hand detected — float (0.0 or 1.0)
[140]         Right hand detected — float (0.0 or 1.0)
[141]         Torso angle — normalised to [-1, 1] via angle / 180.0
[142]         Best object detection confidence — [0, 1]; 0 if none
[143 – 144]  Best object centre (cx, cy) — normalised [0, 1]; 0 if none
[145]         Best object normalised area — [0, 1]; 0 if none
[146]         Left wrist-to-best-object distance — normalised [0, 1]; 1 if none
[147]         Right wrist-to-best-object distance — normalised [0, 1]; 1 if none
[148 – 153]  HOI state one-hot: [NONE, UNKNOWN, APPROACHING, GRASPING, MANIPULATING, RELEASED]
[153]         HOI interaction dwell — normalised [0, 1] via dwell / 120.0

Total: 154 dimensions

Notes
-----
- "Best object" = highest-confidence detection this frame. If the custom
  apparatus detector is used in a future phase, it will replace YOLO's COCO
  classes seamlessly.
- Pose landmarks are always in the order defined by MediaPipe (0 = NOSE, …,
  32 = RIGHT_FOOT_INDEX). If fewer than 33 landmarks are returned (abnormal),
  the remainder is zero-padded.
- All values are float32.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..core.models import (
    DetectedObject,
    FramePerceptionResult,
    HandTrackingResult,
    InteractionEvent,
    InteractionState,
    PoseEstimationResult,
    TemporalFeatureVector,
)

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_DIM: int = 154
N_POSE_LANDMARKS: int = 33
MAX_TORSO_ANGLE_DEG: float = 180.0
MAX_DWELL_FRAMES: float = 120.0

# HOI state one-hot order (must stay fixed — determines slot indices 148-153)
_HOI_STATE_ORDER = [
    InteractionState.NONE,
    InteractionState.UNKNOWN,
    InteractionState.APPROACHING,
    InteractionState.GRASPING,
    InteractionState.MANIPULATING,
    InteractionState.RELEASED,
]
_HOI_STATE_TO_IDX = {s: i for i, s in enumerate(_HOI_STATE_ORDER)}


# ─── Feature Extractor ────────────────────────────────────────────────────────

class PerceptionFeatureExtractor:
    """Converts a FramePerceptionResult into a fixed-dimension TemporalFeatureVector.

    Usage
    -----
        extractor = PerceptionFeatureExtractor()
        fv = extractor.extract(frame_result)      # TemporalFeatureVector
        arr = fv.to_numpy()                        # np.ndarray, shape (154,)
    """

    def __init__(self, feature_dim: int = FEATURE_DIM):
        if feature_dim != FEATURE_DIM:
            raise ValueError(
                f"feature_dim must be {FEATURE_DIM}. Got {feature_dim}. "
                "Changing the dimension breaks the temporal model's input contract."
            )
        self.feature_dim = FEATURE_DIM

    def extract(self, perception: FramePerceptionResult) -> TemporalFeatureVector:
        """Extract the feature vector from one frame's perception output.

        Args:
            perception: A fully populated FramePerceptionResult.

        Returns:
            TemporalFeatureVector with data shaped (FEATURE_DIM,).
        """
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── Slots 0–131: Body Pose ───────────────────────────────────────
        self._fill_pose(vec, perception.body_pose)

        # ── Slots 132–137: Hand Positions & Grasps ───────────────────────
        self._fill_hands(vec, perception.left_hand, perception.right_hand)

        # ── Slots 138–141: Detection Presence Flags & Torso Angle ────────
        vec[138] = 1.0 if (perception.body_pose and perception.body_pose.is_detected) else 0.0
        vec[139] = 1.0 if (perception.left_hand and perception.left_hand.is_detected) else 0.0
        vec[140] = 1.0 if (perception.right_hand and perception.right_hand.is_detected) else 0.0
        if perception.body_pose:
            vec[141] = float(perception.body_pose.torso_angle) / MAX_TORSO_ANGLE_DEG

        # ── Slots 142–147: Best Object Geometry & Wrist Distances ────────
        self._fill_objects(
            vec,
            perception.objects,
            perception.left_hand,
            perception.right_hand,
        )

        # ── Slots 148–153: HOI State One-Hot ─────────────────────────────
        self._fill_hoi(vec, perception.interactions)

        return TemporalFeatureVector(
            data=vec,
            frame_id=perception.frame_id,
            timestamp_sec=perception.timestamp_sec,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _fill_pose(self, vec: np.ndarray, pose: Optional[PoseEstimationResult]) -> None:
        """Fill slots 0–131 with normalised pose landmark data."""
        if pose is None or not pose.is_detected or not pose.landmarks:
            return  # Leave as zeros

        for lm in pose.landmarks:
            if 0 <= lm.id < N_POSE_LANDMARKS:
                base = lm.id * 3
                vec[base]     = float(lm.x)
                vec[base + 1] = float(lm.y)
                vec[base + 2] = float(lm.z)
                vec[99 + lm.id] = float(max(0.0, min(1.0, lm.visibility)))

    def _fill_hands(
        self,
        vec: np.ndarray,
        left: Optional[HandTrackingResult],
        right: Optional[HandTrackingResult],
    ) -> None:
        """Fill slots 132–137 with hand wrist positions and grasp flags."""
        if left and left.is_detected:
            vec[132] = float(left.wrist_pos[0])
            vec[133] = float(left.wrist_pos[1])
            vec[136] = 1.0 if left.is_grasping else 0.0
        if right and right.is_detected:
            vec[134] = float(right.wrist_pos[0])
            vec[135] = float(right.wrist_pos[1])
            vec[137] = 1.0 if right.is_grasping else 0.0

    def _fill_objects(
        self,
        vec: np.ndarray,
        objects: List[DetectedObject],
        left: Optional[HandTrackingResult],
        right: Optional[HandTrackingResult],
    ) -> None:
        """Fill slots 142–147 with best-object geometry and wrist-to-object distances."""
        if not objects:
            vec[146] = 1.0  # Max distance when no objects
            vec[147] = 1.0
            return

        # Pick highest-confidence detection as "best object"
        best = max(objects, key=lambda o: o.confidence)
        cx, cy = best.bbox.center
        diag = math.sqrt(
            (best.bbox.xmax - best.bbox.xmin) ** 2 +
            (best.bbox.ymax - best.bbox.ymin) ** 2
        )

        vec[142] = float(best.confidence)
        vec[143] = float(cx)
        vec[144] = float(cy)
        # Normalise area: cap at 1.0 (in normalised coords, area can't exceed 1)
        vec[145] = float(min(1.0, best.bbox.area))

        # Wrist-to-object-centre distances (normalised by image diagonal ≈ 1.41)
        lx, ly = (left.wrist_pos if left and left.is_detected else (None, None))
        rx, ry = (right.wrist_pos if right and right.is_detected else (None, None))

        if lx is not None:
            vec[146] = float(min(1.0, math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)))
        else:
            vec[146] = 1.0  # Sentinel: no left hand

        if rx is not None:
            vec[147] = float(min(1.0, math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)))
        else:
            vec[147] = 1.0  # Sentinel: no right hand

    def _fill_hoi(
        self,
        vec: np.ndarray,
        interactions: List[InteractionEvent],
    ) -> None:
        """Fill slots 148–153 with HOI state one-hot and dwell."""
        if not interactions:
            # Default to NONE one-hot
            vec[148] = 1.0
            return

        # Pick the highest-priority interaction: MANIPULATING > GRASPING > APPROACHING > RELEASED > UNKNOWN
        priority = {
            InteractionState.MANIPULATING: 5,
            InteractionState.GRASPING: 4,
            InteractionState.APPROACHING: 3,
            InteractionState.RELEASED: 2,
            InteractionState.UNKNOWN: 1,
            InteractionState.NONE: 0,
        }
        best_event = max(interactions, key=lambda e: priority.get(e.state, 0))

        state_idx = _HOI_STATE_TO_IDX.get(best_event.state, 0)
        vec[148 + state_idx] = 1.0

        # Dwell normalised to [0, 1] — clip at MAX_DWELL_FRAMES
        vec[153] = float(min(1.0, best_event.duration_frames / MAX_DWELL_FRAMES))
