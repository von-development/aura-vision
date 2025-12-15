"""
Aura-Vision: Lightweight computer vision module for real-time object detection.

This package provides:
- Real-time object detection using YOLO11
- Monocular depth estimation (known object size method)
- Size estimation from depth
- Camera capture abstraction
- Scene perception and serialization
- Configurable runtime profiles

Example usage:
    >>> from aura import YoloDetector, CameraSource, FrameRenderer, SceneBuilder
    >>> from aura.config import load_profile
    >>> from aura.depth import KnownSizeDepthEstimator
    >>>
    >>> profile = load_profile("laptop")
    >>> detector = YoloDetector(profile.detector)
    >>> depth_estimator = KnownSizeDepthEstimator()
    >>> scene_builder = SceneBuilder(depth_estimator=depth_estimator)
    >>> camera = CameraSource(profile.camera)
    >>>
    >>> with camera:
    ...     for frame in camera:
    ...         detections = detector.detect(frame)
    ...         scene = scene_builder.build(detections, frame)
    ...         # scene.objects now have depth_m and size info!
"""

from aura.core.schemas import BoundingBox, Detection, MeasuredObject, Scene
from aura.detection.yolo import YoloDetector
from aura.vision.camera import CameraSource
from aura.vision.renderer import FrameRenderer
from aura.perception.scene import SceneBuilder
from aura.config import load_profile, DetectorConfig, RuntimeProfile
from aura.depth import KnownSizeDepthEstimator, CameraCalibration, CalibrationData
from aura.measurement import SizeEstimator

__all__ = [
    # Core schemas
    "BoundingBox",
    "Detection",
    "MeasuredObject",
    "Scene",
    # Detection
    "YoloDetector",
    # Depth estimation
    "KnownSizeDepthEstimator",
    "CameraCalibration",
    "CalibrationData",
    # Measurement
    "SizeEstimator",
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

__version__ = "0.3.0"
