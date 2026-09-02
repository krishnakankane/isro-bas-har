"""Data structures and transfer objects for perception, HAR, and validation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class AnomalyType(Enum):
    NONE = auto()
    OUT_OF_ORDER_ACTION = auto()
    OUT_OF_SEQUENCE = auto()  # Alias for OUT_OF_ORDER_ACTION
    UNEXPECTED_ACTION = auto()
    REPEATED_ACTION = auto()
    MISSING_EXPECTED_ACTION = auto()
    SKIPPED_STEP = auto()     # Alias for MISSING_EXPECTED_ACTION
    LOW_CONFIDENCE = auto()
    INVALID_TRANSITION = auto()
    TIMEOUT_EXCEEDED = auto()
    MISSING_REQUIRED_OBJECT = auto()
    UNKNOWN_ANOMALY = auto()


class InteractionState(Enum):
    NONE = auto()       # No interaction data
    UNKNOWN = auto()    # Hands/objects present but no interaction detected
    APPROACHING = auto() # Hand wrist within approach zone of object
    GRASPING = auto()   # Hand is_grasping AND within contact zone (NOT confirmed physical contact)
    MANIPULATING = auto() # GRASPING sustained for dwell_frames_to_manipulate frames
    RELEASED = auto()   # Interaction ended; hand departed or grasp released


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass
class PoseLandmark:
    id: int
    name: str
    x: float  # Normalized [0, 1] or pixel
    y: float
    z: float  # Metric depth or relative scale
    visibility: float = 1.0


@dataclass
class PoseEstimationResult:
    landmarks: List[PoseLandmark] = field(default_factory=list)
    torso_angle: float = 0.0  # Angle in degrees relative to vertical axis (zero-g posture)
    is_detected: bool = False


@dataclass
class HandLandmark:
    id: int
    x: float
    y: float
    z: float


@dataclass
class HandTrackingResult:
    handedness: str  # 'Left' or 'Right'
    landmarks: List[HandLandmark] = field(default_factory=list)
    wrist_pos: Tuple[float, float] = (0.0, 0.0)
    is_grasping: bool = False
    is_detected: bool = False


@dataclass
class InteractionEvent:
    hand_side: str  # 'Left' or 'Right'
    target_object_name: str
    target_object_bbox: BoundingBox
    state: InteractionState = InteractionState.NONE
    proximity_distance_px: float = 0.0  # Stored as normalised [0,1] distance
    duration_frames: int = 0
    confidence: float = 1.0  # Reserved for future probabilistic HOI models


@dataclass
class TemporalFeatureVector:
    """Fixed-dimension feature vector for one frame, fed into the sliding window.

    Dimension breakdown (total = 154):
    ─────────────────────────────────
    Body pose landmarks  : 33 × 3 (x, y, z)      = 99
    Pose visibility      : 33 × 1                  = 33
    Left wrist position  : 2 (x, y)               =  2
    Right wrist position : 2 (x, y)               =  2
    Left is_grasping     : 1 (bool as float)       =  1
    Right is_grasping    : 1 (bool as float)       =  1
    Pose detected        : 1 (bool as float)       =  1
    Left hand detected   : 1 (bool as float)       =  1
    Right hand detected  : 1 (bool as float)       =  1
    Torso angle          : 1 (degrees, normalised) =  1
    Best object conf     : 1                       =  1
    Best object cx, cy   : 2 (normalised)          =  2
    Best object area     : 1 (normalised)          =  1
    Left-to-obj dist     : 1 (normalised)          =  1
    Right-to-obj dist    : 1 (normalised)          =  1
    HOI state one-hot    : 6 (one per state)       =  6
    HOI dwell frames     : 1 (normalised 0-1)      =  1
    ──────────────────────────────────────────────────
    TOTAL                :                         154
    """
    FEATURE_DIM: int = field(default=154, init=False, repr=False)
    data: np.ndarray = field(default_factory=lambda: np.zeros(154, dtype=np.float32))
    frame_id: int = 0
    timestamp_sec: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Return the flat float32 feature array."""
        return self.data.copy()


@dataclass
class FramePerceptionResult:
    frame_id: int
    timestamp_sec: float
    objects: List[DetectedObject] = field(default_factory=list)
    body_pose: Optional[PoseEstimationResult] = None
    left_hand: Optional[HandTrackingResult] = None
    right_hand: Optional[HandTrackingResult] = None
    interactions: List[InteractionEvent] = field(default_factory=list)
    raw_frame: Optional[np.ndarray] = None


@dataclass
class ExperimentStep:
    step_id: int
    name: str
    description: str
    required_objects: List[str]
    expected_action: str
    preconditions: List[int]
    min_duration_sec: float
    max_duration_sec: float
    voice_prompt_on_start: str
    voice_prompt_on_complete: str


@dataclass
class ProtocolDefinition:
    protocol_id: str
    protocol_name: str
    description: str
    version: str
    steps: List[ExperimentStep] = field(default_factory=list)


@dataclass
class AnomalyEvent:
    """Structured record of a detected experiment protocol anomaly."""
    anomaly_type: AnomalyType
    current_step_id: int
    current_step_name: str
    expected_action: str
    observed_action: str
    confidence: float
    timestamp_sec: float
    frame: int = 0
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    explanation: str = ""


@dataclass
class TransitionEvent:
    """Structured record of an action transition in the protocol validator."""
    timestamp_sec: float
    frame: int
    previous_action: str
    current_action: str
    expected_action: str
    step_id: int
    valid: bool
    anomaly_type: Optional[str] = None
    anomaly_message: str = ""


@dataclass
class ValidationResult:
    is_valid: bool
    current_step_id: int
    current_step_name: str
    completed_step_ids: List[int]
    suggested_next_step_id: int
    suggested_next_step_name: str
    anomaly_type: AnomalyType = AnomalyType.NONE
    anomaly_message: str = ""
    confidence: float = 1.0
    is_completed: bool = False
    anomaly_score: float = 0.0
    expected_action: str = ""
    observed_action: str = ""
    status_text: str = "OK"  # "OK", "WAITING", "UNCERTAIN", "ANOMALY", "TIMEOUT", "COMPLETED"
    step_timestamps: Dict[int, float] = field(default_factory=dict)
