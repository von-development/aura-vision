"""
Configuration module: Settings, profiles, and configuration loading.

Provides validated configuration classes and profile management.
"""

from aura.config.settings import DetectorConfig, RuntimeProfile, CameraConfig
from aura.config.loader import load_profile, get_available_profiles

__all__ = [
    "DetectorConfig",
    "RuntimeProfile",
    "CameraConfig",
    "load_profile",
    "get_available_profiles",
]

