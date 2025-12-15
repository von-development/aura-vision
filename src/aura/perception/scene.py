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
from aura.depth.base import DepthEstimator
from aura.depth.calibration import CalibrationData, CameraCalibration
from aura.measurement.size import SizeEstimator

if TYPE_CHECKING:
    from aura.core.types import FrameArray

logger = logging.getLogger(__name__)


class SceneBuilder:
    """
    Builds Scene objects from raw detections.

    Handles the conversion of detection results into structured
    perception data, including optional depth and size estimation.

    Attributes:
        frame_counter: Running count of processed frames.
        depth_estimator: Optional depth estimation method.
        size_estimator: Optional size estimation calculator.

    Example:
        >>> from aura.depth import KnownSizeDepthEstimator
        >>> depth_est = KnownSizeDepthEstimator()
        >>> builder = SceneBuilder(depth_estimator=depth_est)
        >>> scene = builder.build(detections, frame)
        >>> for obj in scene.objects:
        ...     print(f"{obj.class_name}: {obj.depth_m:.2f}m")
    """

    def __init__(
        self,
        depth_estimator: DepthEstimator | None = None,
        calibration: CalibrationData | None = None,
        enable_size_estimation: bool = True,
    ) -> None:
        """
        Initialize the scene builder.

        Args:
            depth_estimator: Depth estimation method to use.
            calibration: Camera calibration for size estimation.
            enable_size_estimation: Whether to estimate physical sizes.
        """
        self._frame_counter: int = 0
        self.depth_estimator = depth_estimator
        self.enable_size_estimation = enable_size_estimation

        # Initialize size estimator if we have calibration or depth estimator
        self.size_estimator: SizeEstimator | None = None
        if enable_size_estimation:
            if calibration is not None:
                self.size_estimator = SizeEstimator(calibration)
            elif depth_estimator is not None:
                # Try to get calibration from depth estimator
                if hasattr(depth_estimator, 'calibration'):
                    self.size_estimator = SizeEstimator(depth_estimator.calibration)
                else:
                    self.size_estimator = SizeEstimator()
            else:
                self.size_estimator = SizeEstimator()

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
        frame: FrameArray | None = None,
        timestamp: float | None = None,
    ) -> Scene:
        """
        Build a Scene from a list of detections.

        If a depth estimator is configured, estimates depth and size
        for each detection automatically.

        Args:
            detections: List of Detection objects from the detector.
            frame: Optional frame for ML-based depth estimation.
            timestamp: Unix timestamp for the scene. Uses current time if None.

        Returns:
            Scene object containing all detections as MeasuredObjects.
        """
        self._frame_counter += 1
        ts = timestamp if timestamp is not None else time.time()

        objects: list[MeasuredObject] = []

        for det in detections:
            depth_m: float | None = None
            width_cm: float | None = None
            height_cm: float | None = None

            # Estimate depth if estimator is available
            if self.depth_estimator is not None:
                depth_result = self.depth_estimator.estimate_depth(det, frame)
                if depth_result.is_valid:
                    depth_m = depth_result.depth_m

                    # Estimate size if we have depth
                    if self.size_estimator is not None and depth_m is not None:
                        size_result = self.size_estimator.estimate_from_detection(
                            det, depth_m
                        )
                        if size_result.is_valid:
                            width_cm = size_result.width_cm
                            height_cm = size_result.height_cm

            objects.append(
                MeasuredObject(
                    detection=det,
                    depth_m=depth_m,
                    width_cm=width_cm,
                    height_cm=height_cm,
                )
            )

        return Scene(
            timestamp=ts,
            frame_id=self._frame_counter,
            objects=objects,
        )

    def build_without_estimation(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> Scene:
        """
        Build a Scene without depth/size estimation.

        Faster method when measurements are not needed.

        Args:
            detections: List of Detection objects from the detector.
            timestamp: Unix timestamp for the scene.

        Returns:
            Scene object with detections only (no measurements).
        """
        self._frame_counter += 1
        ts = timestamp if timestamp is not None else time.time()

        objects = [MeasuredObject(detection=det) for det in detections]

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
        Build a Scene with explicit depth measurements.

        Args:
            detections: List of Detection objects.
            depths: List of depth values (meters) for each detection.
                    Must match length of detections if provided.
            timestamp: Unix timestamp for the scene.

        Returns:
            Scene object with depth and size measurements populated.

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
            depth_m = depths[i] if depths is not None else None
            width_cm: float | None = None
            height_cm: float | None = None

            # Estimate size if we have depth and size estimator
            if depth_m is not None and self.size_estimator is not None:
                size_result = self.size_estimator.estimate_from_detection(det, depth_m)
                if size_result.is_valid:
                    width_cm = size_result.width_cm
                    height_cm = size_result.height_cm

            objects.append(
                MeasuredObject(
                    detection=det,
                    depth_m=depth_m,
                    width_cm=width_cm,
                    height_cm=height_cm,
                )
            )

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
        obj_info = []
        for obj in scene.objects:
            info = obj.class_name
            if obj.depth_m is not None:
                info += f" @{obj.depth_m:.2f}m"
            obj_info.append(info)
        logger.log(
            level,
            f"Frame {scene.frame_id}: {scene.num_objects} objects - {obj_info}"
        )
