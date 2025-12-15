"""
YOLO-based object detector implementation.

Uses Ultralytics YOLO11 for real-time object detection.
Supports both PyTorch (.pt) and ONNX (.onnx) model formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ultralytics import YOLO

from aura.config.settings import DetectorConfig
from aura.core.schemas import BoundingBox, Detection
from aura.detection.base import Detector

if TYPE_CHECKING:
    from aura.core.types import FrameArray

logger = logging.getLogger(__name__)


class YoloDetector(Detector):
    """
    YOLO11-based object detector using Ultralytics.

    This detector wraps the Ultralytics YOLO model for inference.
    It accepts BGR uint8 images (standard OpenCV format) and returns
    a list of Detection objects.

    Attributes:
        config: Detector configuration settings.

    Example:
        >>> from aura.config import DetectorConfig
        >>> config = DetectorConfig(confidence_threshold=0.3)
        >>> detector = YoloDetector(config)
        >>> detections = detector.detect(frame)
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """
        Initialize the YOLO detector.

        Args:
            config: Detector configuration. Uses defaults if None.

        Raises:
            FileNotFoundError: If the model weights file doesn't exist.
            RuntimeError: If the model fails to load.
        """
        self.config = config or DetectorConfig()

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                f"Please ensure the weights file exists or update the config."
            )

        logger.info(f"Loading YOLO model from: {model_path}")
        self._model = YOLO(str(model_path))
        logger.info(
            f"Model loaded successfully. Classes: {len(self._model.names)}"
        )

    def detect(self, frame: FrameArray) -> list[Detection]:
        """
        Run object detection on a single frame.

        Args:
            frame: Input image as BGR uint8 numpy array (H, W, 3).

        Returns:
            List of Detection objects found in the frame.
        """
        # Run inference with configured parameters
        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            max_det=self.config.max_detections,
            classes=self.config.classes,
            verbose=False,
        )[0]

        # Convert YOLO results to Detection objects
        detections: list[Detection] = []

        for box in results.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )

            # Extract class info and confidence
            class_id = int(box.cls[0].item())
            class_name = self._model.names[class_id]
            confidence = float(box.conf[0].item())

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    box=bbox,
                )
            )

        return detections

    @property
    def class_names(self) -> dict[int, str]:
        """
        Get mapping of class IDs to class names.

        Returns:
            Dictionary mapping class ID (int) to class name (str).
        """
        return dict(self._model.names)

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes the detector can recognize.

        Returns:
            Number of detection classes.
        """
        return len(self._model.names)

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"YoloDetector("
            f"model={self.config.model_path.name}, "
            f"num_classes={self.num_classes}, "
            f"device={self.config.device})"
        )




