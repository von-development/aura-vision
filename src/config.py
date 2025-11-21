# src/config.py
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS_DIR = REPO_ROOT / "yolo"


@dataclass
class YoloDetectorConfig:
    """
    Config for yolo models (detection)
    Adjust model_path if you move the weights.
    """
    model_path: Path = YOLO_WEIGHTS_DIR / "yolo11n.pt"

    # Inference / NMS settings
    conf_threshold: float = 0.4   # minimum confidence
    iou_threshold: float = 0.45   # IoU for NMS
    max_det: int = 50             # max detections per image

    # Input / runtime
    imgsz: int = 416              # inference size (square)
    device: str = "cpu"           # "cpu" or "0" for first GPU
    classes: list[int] | None = None  # filter by class IDs, or None = all


@dataclass
class RuntimeProfile:
    name: str
    yolo: YoloDetectorConfig
    frame_skip: int = 1  # run detection every N frames


def laptop_profile() -> RuntimeProfile:
    """Profile for your laptop webcam."""
    return RuntimeProfile(
        name="laptop",
        yolo=YoloDetectorConfig(
            imgsz=416,
            device="cpu",      # change to "0" if you have GPU
            max_det=50,
            conf_threshold=0.4,
        ),
        frame_skip=2,          # try 2 if you want more speed
    )


def pi_profile() -> RuntimeProfile:
    """Profile for Raspberry Pi."""
    return RuntimeProfile(
        name="raspberry_pi",
        yolo=YoloDetectorConfig(
            imgsz=320,
            device="cpu",
            max_det=30,
            conf_threshold=0.3,
        ),
        frame_skip=2,          # detect every 2 frames
    )
