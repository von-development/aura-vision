"""
Abstract base class for object detectors.

Defines the interface that all detector implementations must follow,
enabling easy swapping of detection backends (YOLO, ONNX, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.core.schemas import Detection
    from aura.core.types import FrameArray


class Detector(ABC):
    """
    Abstract base class for object detectors.

    All detector implementations must inherit from this class and
    implement the `detect` method. This allows the rest of the
    pipeline to work with any detector backend.

    Example:
        >>> class MyDetector(Detector):
        ...     def detect(self, frame: FrameArray) -> list[Detection]:
        ...         # Custom detection logic
        ...         return detections
    """

    @abstractmethod
    def detect(self, frame: FrameArray) -> list[Detection]:
        """
        Run object detection on a single frame.

        Args:
            frame: Input image as BGR uint8 numpy array (H, W, 3).

        Returns:
            List of Detection objects found in the frame.
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """
        Get mapping of class IDs to class names.

        Returns:
            Dictionary mapping class ID (int) to class name (str).
        """
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """
        Get the number of classes the detector can recognize.

        Returns:
            Number of detection classes.
        """
        ...

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}(num_classes={self.num_classes})"




