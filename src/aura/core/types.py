"""
Type aliases and type definitions for Aura-Vision.

This module defines common type aliases used throughout the codebase
to improve readability and provide better IDE support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from aura.core.schemas import Detection

# Type alias for OpenCV-compatible BGR image frames (H, W, 3) uint8
FrameArray = NDArray[np.uint8]

# Type alias for a list of detections
DetectionList = list["Detection"]

# Bounding box as (x1, y1, x2, y2) tuple
BoxCoordinates = tuple[int, int, int, int]

# Normalized bounding box (0-1 range) as (x_center, y_center, width, height)
NormalizedBox = tuple[float, float, float, float]

# RGB color tuple for drawing
ColorRGB = tuple[int, int, int]

# BGR color tuple (OpenCV format)
ColorBGR = tuple[int, int, int]




