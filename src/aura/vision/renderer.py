"""
Frame rendering and visualization for Aura-Vision.

Provides utilities for drawing detection results, overlays,
and debug information on video frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2

from aura.core.types import ColorBGR

if TYPE_CHECKING:
    from aura.core.schemas import Detection, MeasuredObject, Scene
    from aura.core.types import FrameArray


# Color palette for different classes (BGR format)
DEFAULT_COLORS: list[ColorBGR] = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (128, 255, 0),    # Light green
    (255, 128, 0),    # Light blue
    (0, 128, 255),    # Orange
    (128, 0, 255),    # Pink
]


@dataclass
class RenderStyle:
    """
    Styling options for detection rendering.

    Attributes:
        box_thickness: Line thickness for bounding boxes.
        font_scale: Scale factor for text.
        font_thickness: Thickness of text strokes.
        show_confidence: Whether to show confidence scores.
        show_class_id: Whether to show class IDs.
        show_depth: Whether to show depth measurements.
        show_size: Whether to show size measurements.
        label_background: Whether to draw background behind labels.
    """

    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    show_confidence: bool = True
    show_class_id: bool = False
    show_depth: bool = True
    show_size: bool = True
    label_background: bool = True


class FrameRenderer:
    """
    Renders detection results and overlays onto video frames.

    Handles drawing bounding boxes, labels, FPS counters, and other
    visual elements for the detection visualization.

    Attributes:
        style: Rendering style configuration.
        colors: Color palette for different classes.

    Example:
        >>> renderer = FrameRenderer()
        >>> annotated = renderer.draw_detections(frame, detections)
        >>> annotated = renderer.draw_fps(annotated, fps=25.3)
        >>> cv2.imshow("Output", annotated)
    """

    def __init__(
        self,
        style: RenderStyle | None = None,
        colors: list[ColorBGR] | None = None,
    ) -> None:
        """
        Initialize the frame renderer.

        Args:
            style: Rendering style options. Uses defaults if None.
            colors: Color palette for classes. Uses defaults if None.
        """
        self.style = style or RenderStyle()
        self.colors = colors or DEFAULT_COLORS

    def _get_color(self, class_id: int) -> ColorBGR:
        """Get color for a given class ID (cycles through palette)."""
        return self.colors[class_id % len(self.colors)]

    def draw_detection(
        self,
        frame: FrameArray,
        detection: Detection,
        color: ColorBGR | None = None,
    ) -> FrameArray:
        """
        Draw a single detection on the frame.

        Args:
            frame: Input frame (modified in place).
            detection: Detection to draw.
            color: Override color, or None to use class-based color.

        Returns:
            Frame with detection drawn (same as input, modified).
        """
        box = detection.box
        color = color or self._get_color(detection.class_id)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            self.style.box_thickness,
        )

        # Build label text
        label_parts = []
        if self.style.show_class_id:
            label_parts.append(f"[{detection.class_id}]")
        label_parts.append(detection.class_name)
        if self.style.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        label = " ".join(label_parts)

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.font_scale,
            self.style.font_thickness,
        )

        # Label position (above the box)
        label_y = max(box.y1 - 8, text_height + 4)
        label_x = box.x1

        # Draw label background
        if self.style.label_background:
            cv2.rectangle(
                frame,
                (label_x, label_y - text_height - 4),
                (label_x + text_width + 4, label_y + 4),
                color,
                cv2.FILLED,
            )
            text_color: ColorBGR = (0, 0, 0)  # Black text on colored background
        else:
            text_color = color

        # Draw label text
        cv2.putText(
            frame,
            label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.font_scale,
            text_color,
            self.style.font_thickness,
            cv2.LINE_AA,
        )

        return frame

    def draw_detections(
        self,
        frame: FrameArray,
        detections: list[Detection],
    ) -> FrameArray:
        """
        Draw all detections on the frame.

        Args:
            frame: Input frame (modified in place).
            detections: List of detections to draw.

        Returns:
            Frame with all detections drawn.
        """
        for detection in detections:
            self.draw_detection(frame, detection)
        return frame

    def draw_measured_object(
        self,
        frame: FrameArray,
        obj: MeasuredObject,
        color: ColorBGR | None = None,
    ) -> FrameArray:
        """
        Draw a measured object with depth and size information.

        Args:
            frame: Input frame (modified in place).
            obj: MeasuredObject with detection and measurements.
            color: Override color, or None to use class-based color.

        Returns:
            Frame with measured object drawn.
        """
        detection = obj.detection
        box = detection.box
        color = color or self._get_color(detection.class_id)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            self.style.box_thickness,
        )

        # Build main label text (class + confidence)
        label_parts = []
        if self.style.show_class_id:
            label_parts.append(f"[{detection.class_id}]")
        label_parts.append(detection.class_name)
        if self.style.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        main_label = " ".join(label_parts)

        # Build measurement label (depth + size)
        measurement_parts = []
        if self.style.show_depth and obj.depth_m is not None:
            measurement_parts.append(f"{obj.depth_m:.2f}m")
        if self.style.show_size and obj.width_cm is not None and obj.height_cm is not None:
            measurement_parts.append(f"{obj.width_cm:.0f}x{obj.height_cm:.0f}cm")
        measurement_label = " | ".join(measurement_parts) if measurement_parts else None

        # Calculate text sizes
        (main_width, main_height), _ = cv2.getTextSize(
            main_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.font_scale,
            self.style.font_thickness,
        )

        # Position for main label (above box)
        label_y = max(box.y1 - 8, main_height + 4)
        label_x = box.x1

        # Draw main label background
        if self.style.label_background:
            cv2.rectangle(
                frame,
                (label_x, label_y - main_height - 4),
                (label_x + main_width + 4, label_y + 4),
                color,
                cv2.FILLED,
            )
            text_color: ColorBGR = (0, 0, 0)
        else:
            text_color = color

        # Draw main label
        cv2.putText(
            frame,
            main_label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.font_scale,
            text_color,
            self.style.font_thickness,
            cv2.LINE_AA,
        )

        # Draw measurement label below the box
        if measurement_label:
            meas_scale = self.style.font_scale * 0.85
            (meas_width, meas_height), _ = cv2.getTextSize(
                measurement_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                meas_scale,
                1,
            )

            meas_y = box.y2 + meas_height + 4
            meas_x = box.x1

            # Draw measurement background
            if self.style.label_background:
                cv2.rectangle(
                    frame,
                    (meas_x, meas_y - meas_height - 2),
                    (meas_x + meas_width + 4, meas_y + 4),
                    (50, 50, 50),  # Dark gray background
                    cv2.FILLED,
                )

            # Draw measurement text (cyan color for visibility)
            cv2.putText(
                frame,
                measurement_label,
                (meas_x + 2, meas_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                meas_scale,
                (255, 255, 0),  # Cyan
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_scene(self, frame: FrameArray, scene: Scene) -> FrameArray:
        """
        Draw all objects from a scene on the frame.

        Automatically uses measured object rendering if depth/size data is available.

        Args:
            frame: Input frame (modified in place).
            scene: Scene containing measured objects.

        Returns:
            Frame with scene objects drawn.
        """
        for obj in scene.objects:
            # Use measured object rendering if we have measurements
            if obj.depth_m is not None or obj.width_cm is not None:
                self.draw_measured_object(frame, obj)
            else:
                self.draw_detection(frame, obj.detection)
        return frame

    def draw_fps(
        self,
        frame: FrameArray,
        fps: float,
        position: tuple[int, int] = (10, 30),
        color: ColorBGR = (0, 255, 255),
    ) -> FrameArray:
        """
        Draw FPS counter on the frame.

        Args:
            frame: Input frame (modified in place).
            fps: Frames per second to display.
            position: (x, y) position for the text.
            color: Text color in BGR format.

        Returns:
            Frame with FPS counter drawn.
        """
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_info(
        self,
        frame: FrameArray,
        text: str,
        position: tuple[int, int] = (10, 60),
        color: ColorBGR = (255, 255, 255),
    ) -> FrameArray:
        """
        Draw informational text on the frame.

        Args:
            frame: Input frame (modified in place).
            text: Text to display.
            position: (x, y) position for the text.
            color: Text color in BGR format.

        Returns:
            Frame with info text drawn.
        """
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        return frame

    def draw_detection_count(
        self,
        frame: FrameArray,
        count: int,
        position: tuple[int, int] = (10, 60),
        color: ColorBGR = (255, 255, 255),
    ) -> FrameArray:
        """
        Draw detection count on the frame.

        Args:
            frame: Input frame (modified in place).
            count: Number of detections.
            position: (x, y) position for the text.
            color: Text color in BGR format.

        Returns:
            Frame with detection count drawn.
        """
        text = f"Objects: {count}"
        return self.draw_info(frame, text, position, color)

