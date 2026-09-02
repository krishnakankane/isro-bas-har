"""Abstract Base Interfaces for all system modules.

Enables dependency inversion and model hot-swapping without affecting application logic.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any, Tuple
import numpy as np

from .models import (
    FramePerceptionResult,
    DetectedObject,
    PoseEstimationResult,
    HandTrackingResult,
    InteractionEvent,
    ValidationResult,
    ProtocolDefinition,
)


class CameraFeedInterface(ABC):
    """Abstract interface for video ingestion."""

    @abstractmethod
    def start(self) -> None:
        """Start the video capture thread."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[Tuple[bool, np.ndarray, float]]:
        """Read the latest frame with timestamp."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and release resources."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if video ingestion is active."""
        pass


class ObjectDetectorInterface(ABC):
    """Abstract interface for laboratory apparatus detection."""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load object detection weights."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Infer bounding boxes and classes from frame."""
        pass


class PoseEstimatorInterface(ABC):
    """Abstract interface for astronaut 2D/3D skeletal pose estimation."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> PoseEstimationResult:
        """Extract body keypoints and zero-g torso orientation."""
        pass


class HandTrackerInterface(ABC):
    """Abstract interface for 21-landmark hand tracking."""

    @abstractmethod
    def track(self, frame: np.ndarray) -> Tuple[Optional[HandTrackingResult], Optional[HandTrackingResult]]:
        """Track (left_hand, right_hand) landmarks and grasp states."""
        pass


class HOIDetectorInterface(ABC):
    """Abstract interface for Hand-Object Interaction spatial detection."""

    @abstractmethod
    def analyze_interaction(
        self,
        objects: List[DetectedObject],
        left_hand: Optional[HandTrackingResult],
        right_hand: Optional[HandTrackingResult],
    ) -> List[InteractionEvent]:
        """Compute spatial proximity, contact, and interaction states."""
        pass


class TemporalHARInterface(ABC):
    """Abstract interface for temporal human activity recognition."""

    @abstractmethod
    def update_and_classify(self, perception: FramePerceptionResult) -> Tuple[str, float]:
        """Update temporal sliding window and predict (action_name, confidence)."""
        pass

    @abstractmethod
    def reset_buffer(self) -> None:
        """Clear temporal feature buffer."""
        pass


class ProtocolValidatorInterface(ABC):
    """Abstract interface for experiment protocol validation and FSM state machine."""

    @abstractmethod
    def load_protocol(self, protocol_def: ProtocolDefinition) -> None:
        """Load experiment protocol specification."""
        pass

    @abstractmethod
    def evaluate_action(self, action_name: str, confidence: float, timestamp: float) -> ValidationResult:
        """Evaluate action against sequence, check preconditions, return validation state."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state machine to initial step."""
        pass


class VoiceAlertInterface(ABC):
    """Abstract interface for offline voice guidance."""

    @abstractmethod
    def speak(self, text: str, priority: int = 1) -> None:
        """Queue voice message for non-blocking speech synthesis."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Cancel speech and clear queue."""
        pass


class TelemetryLoggerInterface(ABC):
    """Abstract interface for structured experiment telemetry and audit logs."""

    @abstractmethod
    def log_step_event(self, frame_id: int, timestamp: float, validation: ValidationResult) -> None:
        """Log timestamped validation record."""
        pass

    @abstractmethod
    def export_summary(self, destination_path: str) -> None:
        """Export session summary report to CSV/JSON."""
        pass


class StreamerInterface(ABC):
    """Abstract interface for IP video streaming."""

    @abstractmethod
    def broadcast_frame(self, frame: np.ndarray) -> None:
        """Send annotated frame to active IP stream subscribers."""
        pass

    @abstractmethod
    def start_server(self, host: str, port: int) -> None:
        """Start local streaming HTTP/RTSP server."""
        pass

    @abstractmethod
    def stop_server(self) -> None:
        """Stop streaming server."""
        pass
