# src/detector.py
from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from .config import YoloDetectorConfig
from .models import Detection   # or whatever you named the types file


class ObjectDetector:
    """Abstract base class for object detectors."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class YoloDetector(ObjectDetector):
    """
    YOLO11-based object detector.

    Expects frames as BGR uint8 (standard OpenCV images).
    """

    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        # Use provided config or default
        self.config = config or YoloDetectorConfig()

        model_source = str(self.config.model_path)
        print(f"[YoloDetector] Loading model from: {model_source}")
        self._model = YOLO(model_source)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            max_det=self.config.max_det,
            classes=self.config.classes,
            verbose=False,
        )[0]

        detections: List[Detection] = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            cls_name = self._model.names[cls_id]
            score = float(box.conf[0].item())

            detections.append(
                Detection(
                    cls_name=cls_name,
                    score=score,
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections
