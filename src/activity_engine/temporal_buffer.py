"""Sliding-Window Temporal Feature Buffer for ISRO BAS HAR.

Accumulates fixed-dimension feature vectors into a rolling window for
consumption by the temporal HAR model (to be trained in a later phase).

Key design properties
---------------------
- Fixed FEATURE_DIM × window_size shape contract — the model always receives
  the same array dimensions.
- Missing-frame robustness — a zero vector is inserted for frames where
  extraction fails, preserving time alignment.
- Thread-safe snapshot — `get_window()` returns a copy.
- Reset method supports experiment transitions without creating a new instance.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..core.models import TemporalFeatureVector
from .feature_extractor import FEATURE_DIM


class TemporalFeatureBuffer:
    """Rolling fixed-length buffer of TemporalFeatureVectors.

    The buffer always holds exactly `window_size` frames once full;
    older frames are evicted automatically (deque with maxlen).

    Args:
        window_size:  Number of frames in the sliding window (default 60).
        feature_dim:  Expected feature vector dimension (must match FEATURE_DIM).
        record_path:  Optional Path to a JSONL file for recording.
                      If given, every pushed vector is also appended to disk.
    """

    def __init__(
        self,
        window_size: int = 60,
        feature_dim: int = FEATURE_DIM,
        record_path: Optional[Path] = None,
    ):
        if feature_dim != FEATURE_DIM:
            raise ValueError(
                f"feature_dim must be {FEATURE_DIM}; got {feature_dim}. "
                "The temporal model expects a fixed-dimension input."
            )
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buffer: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        self._record_path: Optional[Path] = None
        self._record_file = None
        if record_path is not None:
            self.enable_recording(record_path)

    # ── Core API ──────────────────────────────────────────────────────────

    def push(self, fv: TemporalFeatureVector) -> None:
        """Append one feature vector to the buffer.

        If the vector dimension mismatches, a zero-padded/truncated version
        is stored to maintain dimensional consistency.

        Args:
            fv: TemporalFeatureVector produced by PerceptionFeatureExtractor.
        """
        data = fv.data
        if data.shape[0] != self.feature_dim:
            corrected = np.zeros(self.feature_dim, dtype=np.float32)
            n = min(data.shape[0], self.feature_dim)
            corrected[:n] = data[:n]
            data = corrected

        with self._lock:
            self._buffer.append(data.copy())

        if self._record_file is not None:
            self._write_record(fv)

    def push_empty(self) -> None:
        """Push a zero vector for a frame where extraction failed.

        Preserves temporal alignment in the window.
        """
        empty = TemporalFeatureVector(
            data=np.zeros(self.feature_dim, dtype=np.float32),
            frame_id=-1,
            timestamp_sec=time.time(),
        )
        self.push(empty)

    def get_window(self) -> np.ndarray:
        """Return the current window as a (window_size, feature_dim) float32 array.

        If the buffer is not yet full, the leading rows are zero-padded.

        Returns:
            np.ndarray of shape (window_size, feature_dim), dtype float32.
        """
        with self._lock:
            current = list(self._buffer)

        window = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        if current:
            n = len(current)
            window[-n:] = np.stack(current, axis=0)
        return window

    def is_full(self) -> bool:
        """Return True when the buffer has accumulated window_size frames."""
        with self._lock:
            return len(self._buffer) == self.window_size

    def current_length(self) -> int:
        """Return the number of frames currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    def reset(self) -> None:
        """Clear the buffer. Call between experiment sessions."""
        with self._lock:
            self._buffer.clear()

    # ── Recording API ─────────────────────────────────────────────────────

    def enable_recording(self, path: Path) -> None:
        """Open a JSONL file for feature recording.

        Each line is one JSON object with keys:
            frame_id, timestamp_sec, data (list of floats)

        Args:
            path: Destination JSONL file path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._record_path = path
        self._record_file = open(path, "a", encoding="utf-8")  # append mode

    def disable_recording(self) -> None:
        """Flush and close the recording file."""
        if self._record_file is not None:
            self._record_file.flush()
            self._record_file.close()
            self._record_file = None

    def _write_record(self, fv: TemporalFeatureVector) -> None:
        """Serialise one feature vector as a JSONL line."""
        try:
            record = {
                "frame_id": fv.frame_id,
                "timestamp_sec": round(fv.timestamp_sec, 6),
                "data": fv.data.tolist(),
            }
            self._record_file.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[TemporalFeatureBuffer] Recording write error: {exc}")

    def __del__(self):
        self.disable_recording()
