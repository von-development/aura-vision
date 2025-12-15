"""
Depth estimation using known object sizes.

This method estimates depth by comparing the pixel size of detected objects
to their known real-world dimensions. Fast and works on any hardware.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from aura.depth.base import DepthEstimator, DepthResult
from aura.depth.calibration import CalibrationData, CameraCalibration

if TYPE_CHECKING:
    from aura.core.schemas import Detection
    from aura.core.types import FrameArray

logger = logging.getLogger(__name__)

# Path to the object sizes database
OBJECT_SIZES_PATH = Path(__file__).parent / "object_sizes.yaml"


def load_object_sizes(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """
    Load known object sizes from YAML file.

    Args:
        path: Path to object sizes YAML, or None for default.

    Returns:
        Dictionary mapping class names to (width_m, height_m) tuples.
    """
    path = path or OBJECT_SIZES_PATH

    if not path.exists():
        logger.warning(f"Object sizes file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Flatten the nested structure (coco, electrocom -> single dict)
    sizes: dict[str, tuple[float, float]] = {}

    for category in data.values():
        if isinstance(category, dict):
            for class_name, dimensions in category.items():
                sizes[class_name] = tuple(dimensions)  # type: ignore

    logger.info(f"Loaded {len(sizes)} known object sizes")
    return sizes


class KnownSizeDepthEstimator(DepthEstimator):
    """
    Depth estimator using known object sizes.

    Estimates depth by comparing the pixel size of detected objects
    to their known real-world dimensions using the pinhole camera model:

        depth = (focal_length * real_size) / pixel_size

    This method is:
    - Fast (no ML inference needed)
    - Accurate for known object classes
    - Works on any hardware (including Raspberry Pi)

    Attributes:
        calibration: Camera calibration data.
        object_sizes: Dictionary of known object sizes.

    Example:
        >>> calibration = CameraCalibration.get_default_calibration()
        >>> estimator = KnownSizeDepthEstimator(calibration)
        >>> result = estimator.estimate_depth(detection)
        >>> print(f"Depth: {result.depth_m:.2f}m")
    """

    def __init__(
        self,
        calibration: CalibrationData | None = None,
        object_sizes_path: Path | None = None,
        use_height: bool = True,
    ) -> None:
        """
        Initialize the known size depth estimator.

        Args:
            calibration: Camera calibration data. Uses default if None.
            object_sizes_path: Path to object sizes YAML. Uses default if None.
            use_height: If True, use height for estimation (usually more reliable).
                       If False, use width.
        """
        self.calibration = calibration or CameraCalibration.get_default_calibration()
        self.object_sizes = load_object_sizes(object_sizes_path)
        self.use_height = use_height

        logger.info(
            f"KnownSizeDepthEstimator initialized: "
            f"f={self.calibration.focal_length_px:.1f}px, "
            f"{len(self.object_sizes)} known objects"
        )

    @property
    def method_name(self) -> str:
        """Name of this depth estimation method."""
        return "known_size"

    def has_known_size(self, class_name: str) -> bool:
        """
        Check if object class has known dimensions.

        Args:
            class_name: Name of the object class.

        Returns:
            True if size is known, False otherwise.
        """
        return class_name in self.object_sizes

    def get_known_size(self, class_name: str) -> tuple[float, float] | None:
        """
        Get known size for an object class.

        Args:
            class_name: Name of the object class.

        Returns:
            Tuple of (width_m, height_m) or None if unknown.
        """
        return self.object_sizes.get(class_name)

    def estimate_depth(
        self,
        detection: Detection,
        frame: FrameArray | None = None,
    ) -> DepthResult:
        """
        Estimate depth for a detection using known object size.

        Args:
            detection: The detected object.
            frame: Not used for this method (can be None).

        Returns:
            DepthResult with estimated depth.
        """
        class_name = detection.class_name

        # Check if we know the size of this object
        if class_name not in self.object_sizes:
            return DepthResult(
                depth_m=None,
                confidence=0.0,
                method=self.method_name,
            )

        real_width_m, real_height_m = self.object_sizes[class_name]
        pixel_width = detection.box.width
        pixel_height = detection.box.height

        # Use height or width for estimation
        if self.use_height and pixel_height > 0:
            real_size = real_height_m
            pixel_size = pixel_height
        elif pixel_width > 0:
            real_size = real_width_m
            pixel_size = pixel_width
        else:
            return DepthResult(
                depth_m=None,
                confidence=0.0,
                method=self.method_name,
            )

        # Calculate depth: Z = (f * W_real) / w_pixels
        depth_m = (self.calibration.focal_length_px * real_size) / pixel_size

        # Estimate confidence based on detection confidence and box size
        # Larger boxes (closer objects) tend to be more accurate
        size_factor = min(1.0, pixel_size / 100.0)  # Higher for larger boxes
        confidence = detection.confidence * 0.7 + size_factor * 0.3

        return DepthResult(
            depth_m=depth_m,
            confidence=confidence,
            method=self.method_name,
        )

    def estimate_depth_both_dimensions(
        self,
        detection: Detection,
    ) -> tuple[DepthResult, DepthResult]:
        """
        Estimate depth using both width and height.

        Useful for comparing estimates and detecting unusual aspect ratios.

        Args:
            detection: The detected object.

        Returns:
            Tuple of (depth_from_width, depth_from_height).
        """
        class_name = detection.class_name

        if class_name not in self.object_sizes:
            no_result = DepthResult(depth_m=None, confidence=0.0, method=self.method_name)
            return (no_result, no_result)

        real_width_m, real_height_m = self.object_sizes[class_name]
        f = self.calibration.focal_length_px

        depth_from_width = None
        depth_from_height = None

        if detection.box.width > 0:
            depth_from_width = (f * real_width_m) / detection.box.width

        if detection.box.height > 0:
            depth_from_height = (f * real_height_m) / detection.box.height

        return (
            DepthResult(
                depth_m=depth_from_width,
                confidence=detection.confidence,
                method="known_size_width",
            ),
            DepthResult(
                depth_m=depth_from_height,
                confidence=detection.confidence,
                method="known_size_height",
            ),
        )

    @property
    def known_classes(self) -> list[str]:
        """Get list of classes with known sizes."""
        return list(self.object_sizes.keys())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KnownSizeDepthEstimator("
            f"focal_length={self.calibration.focal_length_px:.1f}px, "
            f"known_objects={len(self.object_sizes)})"
        )




