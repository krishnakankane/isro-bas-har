"""Object Detector implementation using Ultralytics YOLO.

Operates with pretrained base weights (yolov8n.pt) for Phase 2 perception validation.
Maintains modular interface so fine-tuned space apparatus models can be plugged in seamlessly.
"""

import time
from typing import List, Optional
import numpy as np
import torch

from ..core.interfaces import ObjectDetectorInterface
from ..core.models import DetectedObject, BoundingBox


class ObjectDetector(ObjectDetectorInterface):
    """YOLO-based Object Detector for apparatus and experiment elements."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initialize Object Detector.

        Args:
            model_name_or_path: Name of pretrained weights or path to custom .pt/.onnx model.
            conf_thresh: Confidence detection threshold.
            iou_thresh: Non-maximum suppression IoU threshold.
            device: 'cuda' or 'cpu' (defaults to CUDA if GPU available).
        """
        self.model_path = model_name_or_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.last_latency_ms: float = 0.0
        self.load_model(self.model_path)

    def load_model(self, model_path: str) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # Warm up model if on GPU
            if self.device == "cuda" and torch.cuda.is_available():
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model.predict(dummy, device=self.device, verbose=False)
            print(f"[ObjectDetector] Successfully loaded '{model_path}' on device '{self.device}'.")
        except Exception as e:
            print(f"[ObjectDetector] Error loading model '{model_path}': {e}. Running in dummy mode.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Perform object detection on input BGR frame.

        Args:
            frame: (H, W, 3) uint8 numpy array.

        Returns:
            List of DetectedObject transfer objects.
        """
        if frame is None or frame.size == 0 or self.model is None:
            self.last_latency_ms = 0.0
            return []

        start_t = time.perf_counter()

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False,
                )

            detections: List[DetectedObject] = []
            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxyn.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    names = res.names

                    for i in range(len(xyxy)):
                        box = xyxy[i]
                        cls_id = int(cls_ids[i])
                        name = names.get(cls_id, f"class_{cls_id}")
                        conf = float(confs[i])

                        bbox = BoundingBox(
                            xmin=float(box[0]),
                            ymin=float(box[1]),
                            xmax=float(box[2]),
                            ymax=float(box[3]),
                        )
                        detections.append(
                            DetectedObject(
                                class_id=cls_id,
                                class_name=name,
                                confidence=conf,
                                bbox=bbox,
                            )
                        )

            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return detections

        except Exception as e:
            print(f"[ObjectDetector] Detection error: {e}")
            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return []
