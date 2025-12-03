"""
Scene building and perception utilities.

Converts raw detections into structured Scene objects with
optional depth and size measurements.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from aura.core.schemas import Detection, MeasuredObject, Scene

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SceneBuilder:
    """
    Builds Scene objects from raw detections.

    Handles the conversion of detection results into structured
    perception data, including optional depth and size estimation.

    Attributes:
        frame_counter: Running count of processed frames.

    Example:
        >>> builder = SceneBuilder()
        >>> scene = builder.build(detections)
        >>> print(scene.to_dict())
    """

    def __init__(self) -> None:
        """Initialize the scene builder."""
        self._frame_counter: int = 0

    @property
    def frame_counter(self) -> int:
        """Current frame count."""
        return self._frame_counter

    def reset(self) -> None:
        """Reset the frame counter."""
        self._frame_counter = 0

    def build(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> Scene:
        """
        Build a Scene from a list of detections.

        Args:
            detections: List of Detection objects from the detector.
            timestamp: Unix timestamp for the scene. Uses current time if None.

        Returns:
            Scene object containing all detections as MeasuredObjects.
        """
        self._frame_counter += 1
        ts = timestamp if timestamp is not None else time.time()

        # Convert detections to MeasuredObjects
        # Note: depth/size fields are None until stereo vision is implemented
        objects = [
            MeasuredObject(detection=det)
            for det in detections
        ]

        return Scene(
            timestamp=ts,
            frame_id=self._frame_counter,
            objects=objects,
        )

    def build_with_measurements(
        self,
        detections: list[Detection],
        depths: list[float | None] | None = None,
        timestamp: float | None = None,
    ) -> Scene:
        """
        Build a Scene with depth measurements.

        Args:
            detections: List of Detection objects.
            depths: List of depth values (meters) for each detection.
                    Must match length of detections if provided.
            timestamp: Unix timestamp for the scene.

        Returns:
            Scene object with depth measurements populated.

        Raises:
            ValueError: If depths list length doesn't match detections.
        """
        if depths is not None and len(depths) != len(detections):
            raise ValueError(
                f"Depths list length ({len(depths)}) must match "
                f"detections length ({len(detections)})"
            )

        self._frame_counter += 1
        ts = timestamp if timestamp is not None else time.time()

        objects: list[MeasuredObject] = []
        for i, det in enumerate(detections):
            depth = depths[i] if depths is not None else None
            objects.append(MeasuredObject(detection=det, depth_m=depth))

        return Scene(
            timestamp=ts,
            frame_id=self._frame_counter,
            objects=objects,
        )


def scene_to_json(scene: Scene, indent: int | None = 2) -> str:
    """
    Serialize a Scene to JSON string.

    Args:
        scene: Scene object to serialize.
        indent: JSON indentation level (None for compact).

    Returns:
        JSON string representation of the scene.
    """
    return json.dumps(scene.to_dict(), indent=indent)


def log_scene(scene: Scene, level: int = logging.INFO) -> None:
    """
    Log scene information.

    Args:
        scene: Scene to log.
        level: Logging level to use.
    """
    if scene.is_empty:
        logger.log(level, f"Frame {scene.frame_id}: No objects detected")
    else:
        obj_names = [obj.class_name for obj in scene.objects]
        logger.log(
            level,
            f"Frame {scene.frame_id}: {scene.num_objects} objects - {obj_names}"
        )

