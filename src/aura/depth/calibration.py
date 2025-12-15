"""
Camera calibration utilities for depth estimation.

Provides simple calibration methods that don't require a chessboard pattern.
Uses reference objects (credit card, A4 paper, etc.) for focal length estimation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default calibration file location
DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "camera_calibration.yaml"

# Common reference objects with known sizes (width, height in meters)
REFERENCE_OBJECTS: dict[str, tuple[float, float]] = {
    "credit_card": (0.0856, 0.0539),      # ISO/IEC 7810 ID-1
    "a4_paper": (0.210, 0.297),            # A4 sheet
    "us_letter": (0.2159, 0.2794),         # US Letter
    "smartphone_avg": (0.070, 0.150),      # Average smartphone
    "dollar_bill": (0.1561, 0.0663),       # US dollar bill
}

# Default focal length for typical webcams (640x480, ~60° horizontal FOV)
DEFAULT_FOCAL_LENGTH = 600.0


@dataclass
class CalibrationData:
    """
    Camera calibration data.

    Stores the camera's intrinsic parameters needed for depth and size estimation.

    Attributes:
        focal_length_px: Focal length in pixels.
        image_width: Calibration image width in pixels.
        image_height: Calibration image height in pixels.
        principal_point: Principal point (cx, cy) in pixels.
        calibration_method: How the calibration was obtained.
        reference_object: Name of reference object used (if applicable).
    """

    focal_length_px: float
    image_width: int = 640
    image_height: int = 480
    principal_point: tuple[float, float] | None = None
    calibration_method: str = "default"
    reference_object: str | None = None

    def __post_init__(self) -> None:
        """Set default principal point if not provided."""
        if self.principal_point is None:
            # Default to image center
            self.principal_point = (self.image_width / 2, self.image_height / 2)

    @property
    def cx(self) -> float:
        """Principal point x-coordinate."""
        return self.principal_point[0]  # type: ignore

    @property
    def cy(self) -> float:
        """Principal point y-coordinate."""
        return self.principal_point[1]  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "focal_length_px": self.focal_length_px,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "principal_point": list(self.principal_point),  # type: ignore
            "calibration_method": self.calibration_method,
            "reference_object": self.reference_object,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationData:
        """Create from dictionary."""
        if "principal_point" in data and data["principal_point"] is not None:
            data["principal_point"] = tuple(data["principal_point"])
        return cls(**data)

    def save(self, path: Path | str) -> None:
        """
        Save calibration data to YAML file.

        Args:
            path: Path to save the calibration file.
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved calibration to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> CalibrationData:
        """
        Load calibration data from YAML file.

        Args:
            path: Path to the calibration file.

        Returns:
            Loaded CalibrationData instance.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded calibration from: {path}")
        return cls.from_dict(data)


class CameraCalibration:
    """
    Simple camera calibration using reference objects.

    This class provides methods to calibrate a camera's focal length
    without requiring a chessboard pattern. Instead, it uses known
    reference objects (credit card, A4 paper, etc.) placed at a
    measured distance from the camera.

    Example:
        >>> calibrator = CameraCalibration()
        >>> # User measures a credit card appearing as 102 pixels wide at 50cm
        >>> calibration = calibrator.calibrate_from_reference(
        ...     reference_object="credit_card",
        ...     pixel_width=102,
        ...     distance_m=0.5,
        ... )
        >>> print(f"Focal length: {calibration.focal_length_px:.1f} pixels")
    """

    def __init__(self, image_width: int = 640, image_height: int = 480) -> None:
        """
        Initialize the calibrator.

        Args:
            image_width: Expected image width in pixels.
            image_height: Expected image height in pixels.
        """
        self.image_width = image_width
        self.image_height = image_height

    def calibrate_from_reference(
        self,
        reference_object: str,
        pixel_width: float,
        distance_m: float,
    ) -> CalibrationData:
        """
        Calculate focal length using a reference object of known size.

        Uses the pinhole camera model: f = (pixel_width * distance) / real_width

        Args:
            reference_object: Name of reference object (e.g., "credit_card").
            pixel_width: Measured width of object in pixels.
            distance_m: Distance from camera to object in meters.

        Returns:
            CalibrationData with calculated focal length.

        Raises:
            ValueError: If reference object is unknown.
        """
        if reference_object not in REFERENCE_OBJECTS:
            available = list(REFERENCE_OBJECTS.keys())
            raise ValueError(
                f"Unknown reference object: {reference_object}. "
                f"Available: {available}"
            )

        real_width_m, real_height_m = REFERENCE_OBJECTS[reference_object]

        # Calculate focal length: f = (w_pixels * Z) / W_real
        focal_length = (pixel_width * distance_m) / real_width_m

        logger.info(
            f"Calibrated focal length: {focal_length:.1f} px "
            f"(using {reference_object} at {distance_m}m)"
        )

        return CalibrationData(
            focal_length_px=focal_length,
            image_width=self.image_width,
            image_height=self.image_height,
            calibration_method="reference_object",
            reference_object=reference_object,
        )

    def calibrate_from_measurement(
        self,
        real_width_m: float,
        pixel_width: float,
        distance_m: float,
    ) -> CalibrationData:
        """
        Calculate focal length using custom measurements.

        Args:
            real_width_m: Real-world width of object in meters.
            pixel_width: Measured width of object in pixels.
            distance_m: Distance from camera to object in meters.

        Returns:
            CalibrationData with calculated focal length.
        """
        focal_length = (pixel_width * distance_m) / real_width_m

        logger.info(
            f"Calibrated focal length: {focal_length:.1f} px "
            f"(custom object: {real_width_m*100:.1f}cm at {distance_m}m)"
        )

        return CalibrationData(
            focal_length_px=focal_length,
            image_width=self.image_width,
            image_height=self.image_height,
            calibration_method="custom_measurement",
        )

    @staticmethod
    def get_default_calibration(
        image_width: int = 640,
        image_height: int = 480,
    ) -> CalibrationData:
        """
        Get default calibration for typical webcams.

        Uses an approximate focal length based on typical webcam specifications
        (~60-70° horizontal field of view).

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            Default CalibrationData with estimated focal length.
        """
        # Estimate focal length from typical FOV (~62° horizontal)
        # f = (image_width / 2) / tan(FOV/2)
        import math
        fov_horizontal_deg = 62.0
        focal_length = (image_width / 2) / math.tan(math.radians(fov_horizontal_deg / 2))

        logger.info(
            f"Using default calibration: f={focal_length:.1f} px "
            f"(estimated from {fov_horizontal_deg}° FOV)"
        )

        return CalibrationData(
            focal_length_px=focal_length,
            image_width=image_width,
            image_height=image_height,
            calibration_method="default_estimate",
        )

    @staticmethod
    def list_reference_objects() -> dict[str, tuple[float, float]]:
        """Get available reference objects and their sizes."""
        return REFERENCE_OBJECTS.copy()




