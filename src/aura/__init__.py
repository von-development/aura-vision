"""
Aura-Vision: Lightweight computer vision module for real-time object detection.

This package provides:
- Real-time object detection using YOLO11
- Camera capture abstraction
- Scene perception and serialization
- Configurable runtime profiles

Example usage:
    >>> from aura import YoloDetector, CameraSource, FrameRenderer
    >>> from aura.config import load_profile
    >>>
    >>> profile = load_profile("laptop")
    >>> detector = YoloDetector(profile.detector)
    >>> camera = CameraSource(index=0)
    >>>
    >>> for frame in camera:
    ...     detections = detector.detect(frame)
    ...     # Process detections...
"""

from aura.core.schemas import BoundingBox, Detection, MeasuredObject, Scene
from aura.detection.yolo import YoloDetector
from aura.vision.camera import CameraSource
from aura.vision.renderer import FrameRenderer
from aura.perception.scene import SceneBuilder
from aura.config import load_profile, DetectorConfig, RuntimeProfile

__all__ = [
    # Core schemas
    "BoundingBox",
    "Detection",
    "MeasuredObject",
    "Scene",
    # Detection
    "YoloDetector",
    # Vision
    "CameraSource",
    "FrameRenderer",
    # Perception
    "SceneBuilder",
    # Config
    "load_profile",
    "DetectorConfig",
    "RuntimeProfile",
]

__version__ = "0.2.0"

