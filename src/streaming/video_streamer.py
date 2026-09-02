"""IP Video Streaming placeholder interface."""

import numpy as np
from ..core.interfaces import StreamerInterface


class VideoStreamer(StreamerInterface):
    """Local network video streaming server (placeholder)."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._is_active = False

    def broadcast_frame(self, frame: np.ndarray) -> None:
        pass

    def start_server(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._is_active = True

    def stop_server(self) -> None:
        self._is_active = False
