"""
Configuration settings for Aura-Vision.

This module defines validated configuration classes for the detector,
camera, and runtime profiles. Settings can be loaded from YAML files
or constructed programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _get_default_weights_path() -> Path:
    """Get the default path to YOLO weights directory."""
    # Resolve relative to the repository root
    module_path = Path(__file__).resolve()
    repo_root = module_path.parent.parent.parent.parent  # src/aura/config -> repo root
    return repo_root / "yolo" / "best.pt"


@dataclass
class DetectorConfig:
    """
    Configuration for the object detector.

    Controls model loading, inference parameters, and NMS settings.

    Attributes:
        model_path: Path to the YOLO weights file (.pt).
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0).
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        max_detections: Maximum number of detections per frame.
        image_size: Input image size for inference (square dimension).
        device: Compute device ("cpu", "cuda", "0", "1", etc.).
        classes: List of class IDs to detect, or None for all classes.
    """

    model_path: Path = field(default_factory=_get_default_weights_path)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 50
    image_size: int = 640
    device: str = "cpu"
    classes: list[int] | None = None

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        # Convert string path to Path object
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        # Validate IoU threshold
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(
                f"iou_threshold must be between 0.0 and 1.0, "
                f"got {self.iou_threshold}"
            )

        # Validate max detections
        if self.max_detections < 1:
            raise ValueError(
                f"max_detections must be at least 1, got {self.max_detections}"
            )

        # Validate image size
        if self.image_size < 32:
            raise ValueError(
                f"image_size must be at least 32, got {self.image_size}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectorConfig:
        """
        Create DetectorConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New DetectorConfig instance.
        """
        # Handle model_path specially to resolve paths
        if "model_path" in data and isinstance(data["model_path"], str):
            data = data.copy()
            data["model_path"] = Path(data["model_path"])
        return cls(**data)


@dataclass
class CameraConfig:
    """
    Configuration for camera capture.

    Attributes:
        source: Camera index (int) or video file path (str).
        width: Desired frame width (None for default).
        height: Desired frame height (None for default).
        fps: Desired frame rate (None for default).
        buffer_size: Camera buffer size (lower = less latency).
    """

    source: int | str = 0
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    buffer_size: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.width is not None and self.width < 1:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height is not None and self.height < 1:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.fps is not None and self.fps < 1:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """
        Create CameraConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New CameraConfig instance.
        """
        return cls(**data)


@dataclass
class RuntimeProfile:
    """
    Complete runtime configuration profile.

    Bundles detector and camera settings along with runtime parameters
    like frame skipping for performance tuning.

    Attributes:
        name: Profile identifier (e.g., "laptop", "raspberry_pi").
        detector: Detector configuration.
        camera: Camera configuration.
        frame_skip: Run detection every N frames (1 = every frame).
        show_fps: Whether to display FPS overlay.
        log_detections: Whether to log detections to console.
    """

    name: str
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    frame_skip: int = 1
    show_fps: bool = True
    log_detections: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.frame_skip < 1:
            raise ValueError(
                f"frame_skip must be at least 1, got {self.frame_skip}"
            )

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> RuntimeProfile:
        """
        Create RuntimeProfile from a dictionary.

        Args:
            name: Profile name.
            data: Dictionary with configuration values.

        Returns:
            New RuntimeProfile instance.
        """
        detector_data = data.get("detector", {})
        camera_data = data.get("camera", {})

        return cls(
            name=name,
            detector=DetectorConfig.from_dict(detector_data),
            camera=CameraConfig.from_dict(camera_data),
            frame_skip=data.get("frame_skip", 1),
            show_fps=data.get("show_fps", True),
            log_detections=data.get("log_detections", False),
        )




