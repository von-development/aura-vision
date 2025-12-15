"""
Depth estimation module for Aura-Vision.

Provides monocular depth estimation methods for single-camera setups.
Supports both known-object-size estimation and ML-based depth prediction.
"""

from aura.depth.base import DepthEstimator, DepthResult
from aura.depth.known_size import KnownSizeDepthEstimator
from aura.depth.calibration import CameraCalibration, CalibrationData

__all__ = [
    "DepthEstimator",
    "DepthResult",
    "KnownSizeDepthEstimator",
    "CameraCalibration",
    "CalibrationData",
]

