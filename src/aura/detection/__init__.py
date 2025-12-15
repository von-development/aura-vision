"""
Detection module: Object detection implementations.

Provides abstract detector protocol and concrete implementations (YOLO).
"""

from aura.detection.base import Detector
from aura.detection.yolo import YoloDetector

__all__ = [
    "Detector",
    "YoloDetector",
]




