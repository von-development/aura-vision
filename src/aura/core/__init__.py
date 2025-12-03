"""
Core module: Data models, types, and protocols.

This module contains the fundamental data structures used throughout Aura-Vision.
"""

from aura.core.schemas import BoundingBox, Detection, MeasuredObject, Scene
from aura.core.types import FrameArray, DetectionList

__all__ = [
    "BoundingBox",
    "Detection",
    "MeasuredObject",
    "Scene",
    "FrameArray",
    "DetectionList",
]

