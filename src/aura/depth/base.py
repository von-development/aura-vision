"""
Abstract base class for depth estimators.

Defines the interface that all depth estimation methods must follow,
enabling easy swapping between different approaches (known size, MiDaS, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.core.schemas import Detection
    from aura.core.types import FrameArray


@dataclass(frozen=True, slots=True)
class DepthResult:
    """
    Result of depth estimation for a single detection.

    Attributes:
        depth_m: Estimated depth in meters (None if estimation failed).
        confidence: Confidence in the estimate (0.0 to 1.0).
        method: Name of the method used for estimation.
    """

    depth_m: float | None
    confidence: float
    method: str

    @property
    def is_valid(self) -> bool:
        """Check if depth estimation was successful."""
        return self.depth_m is not None and self.depth_m > 0


class DepthEstimator(ABC):
    """
    Abstract base class for depth estimators.

    All depth estimation implementations must inherit from this class
    and implement the `estimate_depth` method.

    Example:
        >>> class MyDepthEstimator(DepthEstimator):
        ...     def estimate_depth(self, detection, frame):
        ...         # Custom depth logic
        ...         return DepthResult(depth_m=1.5, confidence=0.8, method="custom")
    """

    @abstractmethod
    def estimate_depth(
        self,
        detection: Detection,
        frame: FrameArray | None = None,
    ) -> DepthResult:
        """
        Estimate depth for a single detection.

        Args:
            detection: The detected object to estimate depth for.
            frame: Optional frame for ML-based methods that need full image.

        Returns:
            DepthResult with estimated depth and confidence.
        """
        ...

    def estimate_depths(
        self,
        detections: list[Detection],
        frame: FrameArray | None = None,
    ) -> list[DepthResult]:
        """
        Estimate depth for multiple detections.

        Args:
            detections: List of detected objects.
            frame: Optional frame for ML-based methods.

        Returns:
            List of DepthResult objects, one per detection.
        """
        return [self.estimate_depth(det, frame) for det in detections]

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of this depth estimation method."""
        ...

    def __repr__(self) -> str:
        """String representation of the estimator."""
        return f"{self.__class__.__name__}(method={self.method_name})"




