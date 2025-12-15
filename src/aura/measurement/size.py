"""
Size estimation from depth and camera intrinsics.

Converts pixel dimensions to real-world measurements (centimeters)
using depth information and the pinhole camera model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aura.depth.calibration import CalibrationData, CameraCalibration

if TYPE_CHECKING:
    from aura.core.schemas import BoundingBox, Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SizeResult:
    """
    Result of size estimation.

    Attributes:
        width_cm: Estimated width in centimeters.
        height_cm: Estimated height in centimeters.
        depth_m: Depth used for estimation in meters.
        confidence: Confidence in the estimate (0.0 to 1.0).
    """

    width_cm: float | None
    height_cm: float | None
    depth_m: float | None
    confidence: float

    @property
    def is_valid(self) -> bool:
        """Check if size estimation was successful."""
        return (
            self.width_cm is not None
            and self.height_cm is not None
            and self.width_cm > 0
            and self.height_cm > 0
        )

    @property
    def area_cm2(self) -> float | None:
        """Estimated area in square centimeters."""
        if self.is_valid:
            return self.width_cm * self.height_cm  # type: ignore
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "width_cm": self.width_cm,
            "height_cm": self.height_cm,
            "depth_m": self.depth_m,
            "confidence": self.confidence,
        }


class SizeEstimator:
    """
    Estimates real-world size from pixel dimensions and depth.

    Uses the pinhole camera model to convert pixel measurements
    to real-world dimensions:

        real_size = (pixel_size * depth) / focal_length

    Attributes:
        calibration: Camera calibration data.

    Example:
        >>> calibration = CameraCalibration.get_default_calibration()
        >>> estimator = SizeEstimator(calibration)
        >>> size = estimator.estimate_size(detection.box, depth_m=1.5)
        >>> print(f"Size: {size.width_cm:.1f}cm x {size.height_cm:.1f}cm")
    """

    def __init__(self, calibration: CalibrationData | None = None) -> None:
        """
        Initialize the size estimator.

        Args:
            calibration: Camera calibration data. Uses default if None.
        """
        self.calibration = calibration or CameraCalibration.get_default_calibration()

        logger.info(
            f"SizeEstimator initialized: "
            f"f={self.calibration.focal_length_px:.1f}px"
        )

    def estimate_size(
        self,
        box: BoundingBox,
        depth_m: float,
        confidence: float = 1.0,
    ) -> SizeResult:
        """
        Estimate real-world size from bounding box and depth.

        Args:
            box: Bounding box in pixels.
            depth_m: Distance to object in meters.
            confidence: Detection confidence to pass through.

        Returns:
            SizeResult with estimated dimensions in centimeters.
        """
        if depth_m <= 0:
            return SizeResult(
                width_cm=None,
                height_cm=None,
                depth_m=depth_m,
                confidence=0.0,
            )

        f = self.calibration.focal_length_px

        # Convert pixel dimensions to meters: W = (w * Z) / f
        width_m = (box.width * depth_m) / f
        height_m = (box.height * depth_m) / f

        # Convert to centimeters
        width_cm = width_m * 100
        height_cm = height_m * 100

        return SizeResult(
            width_cm=width_cm,
            height_cm=height_cm,
            depth_m=depth_m,
            confidence=confidence,
        )

    def estimate_from_detection(
        self,
        detection: Detection,
        depth_m: float,
    ) -> SizeResult:
        """
        Estimate size from a detection object.

        Args:
            detection: Detection with bounding box.
            depth_m: Distance to object in meters.

        Returns:
            SizeResult with estimated dimensions.
        """
        return self.estimate_size(
            box=detection.box,
            depth_m=depth_m,
            confidence=detection.confidence,
        )

    def pixel_to_meters(self, pixel_size: float, depth_m: float) -> float:
        """
        Convert pixel dimension to meters at given depth.

        Args:
            pixel_size: Size in pixels.
            depth_m: Distance in meters.

        Returns:
            Size in meters.
        """
        return (pixel_size * depth_m) / self.calibration.focal_length_px

    def meters_to_pixels(self, real_size_m: float, depth_m: float) -> float:
        """
        Convert real-world dimension to pixels at given depth.

        Args:
            real_size_m: Size in meters.
            depth_m: Distance in meters.

        Returns:
            Size in pixels.
        """
        return (real_size_m * self.calibration.focal_length_px) / depth_m

    def __repr__(self) -> str:
        """String representation."""
        return f"SizeEstimator(focal_length={self.calibration.focal_length_px:.1f}px)"




