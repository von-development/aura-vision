"""
Command-line interface for Aura-Vision.

Provides entry points for running the webcam demo and other
utilities from the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from aura.config import load_profile, get_available_profiles
from aura.config.loader import create_default_profile
from aura.config.settings import CameraConfig
from aura.detection.yolo import YoloDetector
from aura.perception.scene import SceneBuilder, scene_to_json, log_scene
from aura.vision.camera import CameraSource
from aura.vision.renderer import FrameRenderer


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="aura",
        description="Aura-Vision: Real-time object detection with YOLO11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aura webcam                     # Run with default profile
  aura webcam --profile laptop    # Use laptop profile
  aura webcam --camera 1          # Use camera index 1
  aura webcam --model yolo/custom.pt  # Use custom model
  aura webcam --verbose           # Enable debug logging

Available profiles: laptop, raspberry_pi, gpu_fast, debug
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Webcam subcommand
    webcam_parser = subparsers.add_parser(
        "webcam",
        help="Run real-time detection on webcam feed",
    )
    webcam_parser.add_argument(
        "--profile", "-p",
        type=str,
        default="laptop",
        help="Runtime profile to use (default: laptop)",
    )
    webcam_parser.add_argument(
        "--camera", "-c",
        type=int,
        default=None,
        help="Camera index to use (overrides profile)",
    )
    webcam_parser.add_argument(
        "--model", "-m",
        type=Path,
        default=None,
        help="Path to YOLO model weights (overrides profile)",
    )
    webcam_parser.add_argument(
        "--confidence", "-t",
        type=float,
        default=None,
        help="Confidence threshold (overrides profile)",
    )
    webcam_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    webcam_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI display (headless mode)",
    )

    # List profiles subcommand
    subparsers.add_parser(
        "profiles",
        help="List available runtime profiles",
    )

    return parser


def run_webcam(args: argparse.Namespace) -> int:
    """
    Run the webcam detection demo.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    logger = logging.getLogger("aura.webcam")

    # Load profile
    try:
        profile = load_profile(args.profile, model_path=args.model)
        logger.info(f"Loaded profile: {profile.name}")
    except (KeyError, FileNotFoundError) as e:
        logger.error(f"Failed to load profile: {e}")
        logger.info("Using default profile instead")
        profile = create_default_profile()

    # Override camera source if specified
    if args.camera is not None:
        profile = type(profile)(
            name=profile.name,
            detector=profile.detector,
            camera=CameraConfig(
                source=args.camera,
                width=profile.camera.width,
                height=profile.camera.height,
                fps=profile.camera.fps,
                buffer_size=profile.camera.buffer_size,
            ),
            frame_skip=profile.frame_skip,
            show_fps=profile.show_fps,
            log_detections=profile.log_detections,
        )

    # Override confidence if specified
    if args.confidence is not None:
        from aura.config.settings import DetectorConfig
        profile = type(profile)(
            name=profile.name,
            detector=DetectorConfig(
                model_path=profile.detector.model_path,
                confidence_threshold=args.confidence,
                iou_threshold=profile.detector.iou_threshold,
                max_detections=profile.detector.max_detections,
                image_size=profile.detector.image_size,
                device=profile.detector.device,
                classes=profile.detector.classes,
            ),
            camera=profile.camera,
            frame_skip=profile.frame_skip,
            show_fps=profile.show_fps,
            log_detections=profile.log_detections,
        )

    # Initialize components
    try:
        detector = YoloDetector(profile.detector)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return 1

    camera = CameraSource(profile.camera)
    renderer = FrameRenderer()
    scene_builder = SceneBuilder()

    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    # Detection caching for frame skip
    cached_detections: list = []
    frame_idx = 0

    logger.info("Starting webcam detection. Press 'q' to quit.")

    try:
        with camera:
            for frame in camera:
                frame_idx += 1
                fps_frame_count += 1

                # Run detection (with frame skipping)
                if frame_idx % profile.frame_skip == 0:
                    detections = detector.detect(frame)
                    cached_detections = detections
                else:
                    detections = cached_detections

                # Build scene
                scene = scene_builder.build(detections)

                # Log detections if enabled
                if profile.log_detections and detections:
                    log_scene(scene)
                    if args.verbose:
                        print(scene_to_json(scene))

                # Display unless headless mode
                if not args.no_display:
                    # Draw detections
                    renderer.draw_detections(frame, detections)

                    # Calculate and draw FPS
                    elapsed = time.time() - fps_start_time
                    if elapsed >= 1.0:
                        current_fps = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_start_time = time.time()

                    if profile.show_fps:
                        renderer.draw_fps(frame, current_fps)

                    # Draw detection count
                    renderer.draw_detection_count(
                        frame, len(detections), position=(10, 60)
                    )

                    # Show frame
                    window_name = f"Aura-Vision [{profile.name}]"
                    cv2.imshow(window_name, frame)

                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested")
                        break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

    logger.info(f"Processed {scene_builder.frame_counter} frames")
    return 0


def run_list_profiles(args: argparse.Namespace) -> int:
    """
    List available runtime profiles.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    profiles = get_available_profiles()
    print("Available profiles:")
    for name in profiles:
        print(f"  - {name}")
    return 0


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    verbose = getattr(args, "verbose", False)
    setup_logging(verbose)

    # Route to appropriate command
    if args.command == "webcam":
        return run_webcam(args)
    elif args.command == "profiles":
        return run_list_profiles(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

