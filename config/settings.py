"""Global system configuration settings for ISRO BAS HAR system."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


@dataclass
class IngestionConfig:
    source_type: str = "webcam"  # 'webcam', 'file', 'rtsp'
    source_path: str = "0"        # Device index or file/stream path
    target_fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720
    buffer_size: int = 5
    rotation_angle: int = 0      # 0, 90, 180, 270 degrees (for zero-g camera mounts)


@dataclass
class PerceptionConfig:
    object_model_path: str = str(MODELS_DIR / "yolov8n_bas_apparatus.pt")
    object_conf_thresh: float = 0.45
    object_iou_thresh: float = 0.45
    pose_min_detection_conf: float = 0.5
    pose_min_tracking_conf: float = 0.5
    hand_min_detection_conf: float = 0.5
    hand_min_tracking_conf: float = 0.5
    hoi_proximity_threshold_px: float = 80.0
    device: str = "cuda"  # 'cuda' or 'cpu'


# Authoritative System-Wide Confidence Threshold
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.65


@dataclass
class ActivityConfig:
    temporal_window_frames: int = 60  # ~2.0 seconds at 30 FPS
    feature_dimension: int = 128
    har_model_path: str = str(MODELS_DIR / "temporal_har_lstm.pt")
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class AlertingConfig:
    enable_voice_alerts: bool = True
    tts_engine: str = "pyttsx3"  # 'pyttsx3' or 'piper'
    voice_rate: int = 175
    voice_volume: float = 1.0
    cooldown_seconds: float = 3.0


@dataclass
class TelemetryConfig:
    log_dir: Path = LOGS_DIR
    enable_jsonl: bool = True
    enable_sqlite: bool = True
    enable_error_snapshots: bool = True


@dataclass
class StreamingConfig:
    enable_stream: bool = True
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    enable_local_recording: bool = True
    record_format: str = "mp4v"
    circular_buffer_hours: int = 4


@dataclass
class SystemSettings:
    project_name: str = "ISRO-BAS-HAR"
    version: str = "1.0.0"
    active_protocol_file: Path = CONFIG_DIR / "experiment_protocols.json"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


# Default settings instance
settings = SystemSettings()
