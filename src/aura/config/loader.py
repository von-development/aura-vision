"""
Configuration loader for Aura-Vision.

Handles loading runtime profiles from YAML configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from aura.config.settings import RuntimeProfile, DetectorConfig, CameraConfig

logger = logging.getLogger(__name__)

# Default profiles configuration path
_DEFAULT_PROFILES_PATH = Path(__file__).parent / "profiles.yaml"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file is invalid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_available_profiles(config_path: Path | None = None) -> list[str]:
    """
    Get list of available profile names.

    Args:
        config_path: Path to profiles YAML file, or None for default.

    Returns:
        List of profile names defined in the configuration.
    """
    path = config_path or _DEFAULT_PROFILES_PATH
    try:
        profiles = _load_yaml_file(path)
        return list(profiles.keys())
    except FileNotFoundError:
        logger.warning(f"Profiles file not found: {path}")
        return []


def load_profile(
    name: str,
    config_path: Path | None = None,
    model_path: Path | None = None,
) -> RuntimeProfile:
    """
    Load a runtime profile by name.

    Args:
        name: Profile name (e.g., "laptop", "raspberry_pi").
        config_path: Path to profiles YAML file, or None for default.
        model_path: Override model path (optional).

    Returns:
        Loaded RuntimeProfile instance.

    Raises:
        KeyError: If the profile name is not found.
        FileNotFoundError: If the config file doesn't exist.
    """
    path = config_path or _DEFAULT_PROFILES_PATH
    profiles = _load_yaml_file(path)

    if name not in profiles:
        available = list(profiles.keys())
        raise KeyError(
            f"Profile '{name}' not found. Available profiles: {available}"
        )

    profile_data = profiles[name]
    logger.info(f"Loading profile: {name}")

    # Create profile from YAML data
    profile = RuntimeProfile.from_dict(name, profile_data)

    # Override model path if provided
    if model_path is not None:
        # Create new detector config with overridden path
        profile = RuntimeProfile(
            name=profile.name,
            detector=DetectorConfig(
                model_path=model_path,
                confidence_threshold=profile.detector.confidence_threshold,
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

    return profile


def create_default_profile() -> RuntimeProfile:
    """
    Create a default runtime profile without loading from file.

    Useful for quick testing or when no config file is available.

    Returns:
        Default RuntimeProfile with sensible defaults.
    """
    return RuntimeProfile(
        name="default",
        detector=DetectorConfig(),
        camera=CameraConfig(),
        frame_skip=1,
        show_fps=True,
        log_detections=False,
    )

