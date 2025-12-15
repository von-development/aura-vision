"""
Camera capture abstraction for Aura-Vision.

Provides a unified interface for capturing frames from webcams,
video files, or image streams.
"""

from __future__ import annotations

import logging
from typing import Iterator, TYPE_CHECKING

import cv2

from aura.config.settings import CameraConfig

if TYPE_CHECKING:
    from aura.core.types import FrameArray

logger = logging.getLogger(__name__)


class CameraSource:
    """
    Camera capture source with iterator interface.

    Wraps OpenCV VideoCapture with automatic configuration and
    resource management. Supports webcams, video files, and RTSP streams.

    Attributes:
        config: Camera configuration settings.
        is_open: Whether the camera is currently open.

    Example:
        >>> config = CameraConfig(source=0, width=640, height=480)
        >>> camera = CameraSource(config)
        >>>
        >>> # Use as context manager
        >>> with camera:
        ...     for frame in camera:
        ...         process(frame)
        ...         if should_stop:
        ...             break
        >>>
        >>> # Or manually manage
        >>> camera.open()
        >>> frame = camera.read()
        >>> camera.close()
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """
        Initialize the camera source.

        Args:
            config: Camera configuration. Uses defaults if None.
        """
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None
        self._frame_count: int = 0

    @property
    def is_open(self) -> bool:
        """Check if the camera is currently open and ready."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def frame_count(self) -> int:
        """Number of frames read since opening."""
        return self._frame_count

    def open(self) -> None:
        """
        Open the camera source.

        Configures the capture with the specified resolution and FPS.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        if self.is_open:
            logger.warning("Camera is already open")
            return

        source = self.config.source
        logger.info(f"Opening camera source: {source}")

        self._capture = cv2.VideoCapture(source)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {source}\n"
                f"For webcam, check if device is connected and not in use.\n"
                f"For video file, check if path exists and format is supported."
            )

        # Configure capture properties
        if self.config.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Set buffer size for lower latency
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Log actual capture properties
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )

        self._frame_count = 0

    def close(self) -> None:
        """Close the camera source and release resources."""
        if self._capture is not None:
            logger.info(f"Closing camera (read {self._frame_count} frames)")
            self._capture.release()
            self._capture = None

    def read(self) -> FrameArray | None:
        """
        Read a single frame from the camera.

        Returns:
            BGR image as numpy array, or None if read failed.

        Raises:
            RuntimeError: If the camera is not open.
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open. Call open() first.")

        success, frame = self._capture.read()  # type: ignore

        if not success:
            return None

        self._frame_count += 1
        return frame

    def __iter__(self) -> Iterator[FrameArray]:
        """
        Iterate over frames from the camera.

        Automatically opens the camera if not already open.

        Yields:
            BGR image frames as numpy arrays.
        """
        if not self.is_open:
            self.open()

        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> CameraSource:
        """Context manager entry - open camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close camera."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the camera source."""
        status = "open" if self.is_open else "closed"
        return (
            f"CameraSource(source={self.config.source}, "
            f"status={status}, frames={self._frame_count})"
        )




