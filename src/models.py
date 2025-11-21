# src/types.py
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


@dataclass
class Detection:
    "Single object detection results"
    cls_name: str
    score: float
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class MeasuredObject:
    """
    Detection plus geometric info (depth, size, bearing).

    For now depth_m and size_cm are None.
    Later, depth from stereo and size from intrinsics will fill these.
    """
    detection: Detection
    depth_m: Optional[float] = None
    size_cm: Optional[Dict[str, float]] = None  # {"width": ..., "height": ...}
    bearing_rad: Optional[float] = None         # angle from camera center (later)


@dataclass
class Scene:
    """Full perception snapshot for one frame."""
    timestamp: float
    objects: List[MeasuredObject]
