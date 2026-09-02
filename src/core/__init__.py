"""Core data structures, interfaces, and protocol state machine."""

from .models import (
    BoundingBox,
    DetectedObject,
    PoseLandmark,
    PoseEstimationResult,
    HandLandmark,
    HandTrackingResult,
    InteractionEvent,
    FramePerceptionResult,
    ExperimentStep,
    ProtocolDefinition,
    ValidationResult,
    AnomalyType,
)
from .interfaces import (
    CameraFeedInterface,
    ObjectDetectorInterface,
    PoseEstimatorInterface,
    HandTrackerInterface,
    HOIDetectorInterface,
    TemporalHARInterface,
    ProtocolValidatorInterface,
    VoiceAlertInterface,
    TelemetryLoggerInterface,
    StreamerInterface,
)
