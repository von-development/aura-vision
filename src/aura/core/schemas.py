"""
Core data schemas for Aura-Vision.

This module defines the fundamental data structures used throughout the
detection and perception pipeline. All classes are immutable dataclasses
with helper methods for common operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from aura.core.types import BoxCoordinates


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    Stores coordinates in (x1, y1, x2, y2) format where:
    - (x1, y1) is the top-left corner
    - (x2, y2) is the bottom-right corner

    Attributes:
        x1: Left edge x-coordinate.
        y1: Top edge y-coordinate.
        x2: Right edge x-coordinate.
        y2: Bottom edge y-coordinate.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_xyxy(cls, coords: BoxCoordinates) -> BoundingBox:
        """
        Create a BoundingBox from (x1, y1, x2, y2) tuple.

        Args:
            coords: Tuple of (x1, y1, x2, y2) coordinates.

        Returns:
            New BoundingBox instance.
        """
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (cx, cy) of the bounding box."""
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return (cx, cy)

    @property
    def xyxy(self) -> BoxCoordinates:
        """Return coordinates as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary representation."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True, slots=True)
class Detection:
    """
    Single object detection result from the detector.

    Represents one detected object with its class, confidence score,
    and bounding box location.

    Attributes:
        class_id: Integer class ID from the model.
        class_name: Human-readable class label.
        confidence: Detection confidence score (0.0 to 1.0).
        box: Bounding box of the detected object.
    """

    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox

    @property
    def label(self) -> str:
        """Formatted label string with class name and confidence."""
        return f"{self.class_name} {self.confidence:.2f}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "box_xyxy": self.box.xyxy,
        }


@dataclass(frozen=True, slots=True)
class MeasuredObject:
    """
    Detection enriched with geometric measurements.

    Extends a raw detection with depth, physical size, and bearing
    information. These measurements come from stereo vision or other
    depth estimation methods.

    Attributes:
        detection: The underlying detection result.
        depth_m: Estimated depth in meters (None if unavailable).
        width_cm: Estimated physical width in centimeters.
        height_cm: Estimated physical height in centimeters.
        bearing_rad: Angle from camera center in radians.
    """

    detection: Detection
    depth_m: float | None = None
    width_cm: float | None = None
    height_cm: float | None = None
    bearing_rad: float | None = None

    @property
    def class_name(self) -> str:
        """Shortcut to access the class name."""
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        """Shortcut to access the confidence score."""
        return self.detection.confidence

    @property
    def box(self) -> BoundingBox:
        """Shortcut to access the bounding box."""
        return self.detection.box

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "detection": self.detection.to_dict(),
            "depth_m": self.depth_m,
            "width_cm": self.width_cm,
            "height_cm": self.height_cm,
            "bearing_rad": self.bearing_rad,
        }


@dataclass(slots=True)
class Scene:
    """
    Complete perception snapshot for a single frame.

    Represents all detected and measured objects in one camera frame,
    along with metadata like timestamp and frame ID.

    Attributes:
        timestamp: Unix timestamp when the frame was captured.
        frame_id: Sequential frame number.
        objects: List of measured objects in the scene.
    """

    timestamp: float
    frame_id: int
    objects: list[MeasuredObject] = field(default_factory=list)

    @property
    def num_objects(self) -> int:
        """Number of objects detected in the scene."""
        return len(self.objects)

    @property
    def is_empty(self) -> bool:
        """Check if no objects were detected."""
        return len(self.objects) == 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "num_objects": self.num_objects,
            "objects": [obj.to_dict() for obj in self.objects],
        }




