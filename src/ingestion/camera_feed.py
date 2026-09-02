"""Video Ingestion Subsystem for ISRO BAS HAR.

Supports webcam capture, local video files, and synthetic video generation for testing.
Designed with abstract interfaces so RTSP streaming can be added seamlessly in future phases.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2

from ..core.interfaces import CameraFeedInterface


class CameraFeed(CameraFeedInterface):
    """Threaded Video Capture Feed supporting Webcam, File, and Synthetic inputs."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_fps: int = 30,
        loop_video: bool = True,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ):
        """Initialize Camera Feed.

        Args:
            source: Integer camera index (0, 1) or filepath string ('video.mp4') or 'synthetic'.
            target_fps: Target acquisition framerate.
            loop_video: Whether to loop playback for video files upon reaching EOF.
            frame_width: Optional target width resize.
            frame_height: Optional target height resize.
        """
        self.source = source
        self.target_fps = target_fps
        self.loop_video = loop_video
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._is_synthetic = (str(source).lower() == "synthetic")
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._fps_actual: float = 0.0
        self._last_frame_time: float = 0.0

    def start(self) -> None:
        """Start the video ingestion reader thread."""
        if self._is_running:
            return

        if not self._is_synthetic:
            # Parse integer index if string digit
            if isinstance(self.source, str) and self.source.isdigit():
                cap_source = int(self.source)
            else:
                cap_source = self.source

            self._cap = cv2.VideoCapture(cap_source)
            if not self._cap.isOpened():
                # Fallback to synthetic mode if camera cannot be opened
                print(f"[CameraFeed] Warning: Could not open source '{self.source}'. Falling back to synthetic test feed.")
                self._is_synthetic = True
            else:
                if self.frame_width and self.frame_height:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Pre-seed initial frame so read_frame() never returns None on startup
        if self._is_synthetic:
            self._latest_frame = self._generate_synthetic_frame()
            self._latest_timestamp = time.time()
        else:
            success, frame = self._cap.read()
            if success and frame is not None:
                if self.frame_width and self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self._latest_frame = frame
                self._latest_timestamp = time.time()
            else:
                self._latest_frame = self._generate_synthetic_frame()
                self._latest_timestamp = time.time()

        self._is_running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        """Background frame reader loop."""
        target_interval = 1.0 / max(1, self.target_fps)

        while self._is_running:
            loop_start = time.time()

            if self._is_synthetic:
                frame = self._generate_synthetic_frame()
                success = True
            else:
                success, frame = self._cap.read()
                if not success:
                    if self.loop_video and isinstance(self.source, str) and Path(self.source).exists():
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, frame = self._cap.read()
                    else:
                        frame = None

            now = time.time()
            if success and frame is not None:
                if self.frame_width and self.frame_height:
                    if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_count += 1
                    if self._last_frame_time > 0:
                        delta = now - self._last_frame_time
                        if delta > 0:
                            self._fps_actual = 0.9 * self._fps_actual + 0.1 * (1.0 / delta)
                    self._last_frame_time = now

            # Sleep to match target frame rate
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a realistic test frame with moving patterns and simulation elements."""
        w, h = self.frame_width or 1280, self.frame_height or 720
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Spacecraft payload background grid
        cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (30, 35, 45), -1)
        for x in range(40, w - 40, 100):
            cv2.line(frame, (x, 40), (x, h - 40), (45, 50, 65), 1)
        for y in range(40, h - 40, 100):
            cv2.line(frame, (40, y), (w - 40, y), (45, 50, 65), 1)

        # Moving geometric target simulating human hand/apparatus interaction
        t = time.time()
        cx = int(w / 2 + (w / 4) * np.sin(t * 1.5))
        cy = int(h / 2 + (h / 6) * np.cos(t * 1.5))

        # Simulated apparatus rack
        cv2.rectangle(frame, (cx - 60, cy - 40), (cx + 60, cy + 40), (0, 165, 255), 2)
        cv2.putText(frame, "SIMULATED PAYLOAD TARGET", (cx - 100, cy - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Watermark
        cv2.putText(frame, "ISRO BAS HAR -- SYNTHETIC CAMERA FEED", (60, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')} | Frame: {self._frame_count}", (60, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame

    def read_frame(self, timeout_sec: float = 0.5) -> Optional[Tuple[bool, np.ndarray, float]]:
        """Read the latest frame, return (success, frame, timestamp)."""
        deadline = time.time() + timeout_sec
        while time.time() <= deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_timestamp
            time.sleep(0.01)

        with self._lock:
            if not self._is_running or self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._latest_timestamp

    def stop(self) -> None:
        """Stop reader thread and release video capture."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap and not self._is_synthetic:
            self._cap.release()
            self._cap = None

    def is_running(self) -> bool:
        """Return running status."""
        return self._is_running

    @property
    def fps_actual(self) -> float:
        """Return observed acquisition FPS."""
        return self._fps_actual
