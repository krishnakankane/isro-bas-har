"""Human Pose Estimator implementation using Google MediaPipe Tasks PoseLandmarker.

Standardizes 33 3D skeletal body keypoints and computes torso axis angles.
Note: Full microgravity/zero-g orientation invariance is treated as an active research & validation task.
"""

import time
import math
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

from ..core.interfaces import PoseEstimatorInterface
from ..core.models import PoseEstimationResult, PoseLandmark


# MediaPipe 33 Landmark Names Mapping
POSE_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


class PoseEstimator(PoseEstimatorInterface):
    """MediaPipe Tasks 3D Skeletal Pose Landmarker."""

    def __init__(
        self,
        model_path: str = "models/mediapipe/pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: Optional[int] = None,
    ):
        """Initialize Pose Estimator using MediaPipe Tasks Vision API.

        Args:
            model_path: Path to MediaPipe .task model bundle.
            min_detection_confidence: Detection threshold.
            min_tracking_confidence: Tracking threshold.
            model_complexity: Kept for API backwards compatibility.
        """
        self.model_path = model_path
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence
        self.model_complexity = model_complexity

        self._detector = None
        self.last_latency_ms: float = 0.0
        self._init_mediapipe()

    def _resolve_model_path(self, path_str: str) -> Path:
        """Resolve model path against working directory and project root."""
        candidate = Path(path_str)
        if candidate.is_file():
            return candidate

        # Check relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate_project = project_root / path_str
        if candidate_project.is_file():
            return candidate_project

        # Check in models/mediapipe/
        candidate_mediapipe = project_root / "models" / "mediapipe" / candidate.name
        if candidate_mediapipe.is_file():
            return candidate_mediapipe

        # Check in models/
        candidate_models = project_root / "models" / candidate.name
        if candidate_models.is_file():
            return candidate_models

        return candidate

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Tasks Vision PoseLandmarker."""
        resolved_path = self._resolve_model_path(self.model_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"[PoseEstimator] Required model asset not found at '{self.model_path}' "
                f"(resolved: '{resolved_path}'). Ensure task models are present in 'models/mediapipe/'."
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(resolved_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.PoseLandmarker.create_from_options(options)
            print(f"[PoseEstimator] MediaPipe Tasks PoseLandmarker initialized successfully ({resolved_path.name}).")
        except Exception as e:
            raise RuntimeError(f"[PoseEstimator] Failed to initialize MediaPipe Tasks PoseLandmarker: {e}") from e

    def estimate(self, frame: np.ndarray) -> PoseEstimationResult:
        """Estimate 33 3D skeletal landmarks from BGR video frame.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            PoseEstimationResult containing standardized 3D landmarks and torso orientation.
        """
        if frame is None or frame.size == 0 or self._detector is None:
            self.last_latency_ms = 0.0
            return PoseEstimationResult(landmarks=[], torso_angle=0.0, is_detected=False)

        start_t = time.perf_counter()

        try:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = self._detector.detect(mp_image)

            if not results.pose_landmarks or len(results.pose_landmarks) == 0:
                self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
                return PoseEstimationResult(landmarks=[], torso_angle=0.0, is_detected=False)

            # Extract first detected person
            person_landmarks = results.pose_landmarks[0]
            landmarks: List[PoseLandmark] = []
            for idx, lm in enumerate(person_landmarks):
                name = POSE_LANDMARK_NAMES[idx] if idx < len(POSE_LANDMARK_NAMES) else f"LANDMARK_{idx}"
                landmarks.append(
                    PoseLandmark(
                        id=idx,
                        name=name,
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        visibility=float(getattr(lm, "visibility", 1.0)),
                    )
                )

            # Compute torso tilt angle in 2D plane (Shoulder to Hip vector)
            torso_angle = self._calculate_torso_angle(landmarks)

            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return PoseEstimationResult(
                landmarks=landmarks,
                torso_angle=torso_angle,
                is_detected=True,
            )

        except Exception as e:
            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            raise RuntimeError(f"[PoseEstimator] Inference error during pose estimation: {e}") from e

    def _calculate_torso_angle(self, landmarks: List[PoseLandmark]) -> float:
        """Compute torso vector angle relative to image vertical axis (in degrees)."""
        if len(landmarks) < 25:
            return 0.0

        l_sh = landmarks[11]
        r_sh = landmarks[12]
        sh_mid = ((l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0)

        l_hip = landmarks[23]
        r_hip = landmarks[24]
        hip_mid = ((l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0)

        dx = hip_mid[0] - sh_mid[0]
        dy = hip_mid[1] - sh_mid[1]

        angle_rad = math.atan2(dx, dy)
        return math.degrees(angle_rad)

    def close(self) -> None:
        """Release detector resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
