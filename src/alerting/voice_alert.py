"""Offline Voice Alert Dispatcher placeholder interface."""

import queue
import time
from typing import Optional
from ..core.interfaces import VoiceAlertInterface


class VoiceAlertDispatcher(VoiceAlertInterface):
    """Asynchronous offline text-to-speech voice alert dispatcher (placeholder)."""

    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self.message_queue = queue.Queue(maxsize=10)
        self.last_spoken_time: float = 0.0
        self.last_spoken_text: str = ""

    def speak(self, text: str, priority: int = 1) -> None:
        """Queue voice message for synthesis (placeholder)."""
        now = time.time()
        if text == self.last_spoken_text and (now - self.last_spoken_time) < self.cooldown_seconds:
            return  # Suppress duplicate alerts within cooldown

        self.last_spoken_text = text
        self.last_spoken_time = now
        # Placeholder non-blocking queue put
        if not self.message_queue.full():
            self.message_queue.put((priority, text))

    def stop(self) -> None:
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break
