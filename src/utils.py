import time
from typing import List

from .models import Detection, MeasuredObject, Scene



## Percepetion Layer
def build_scene_from_detections(detections: List[Detection]) -> Scene:
    """Create a Scene snapshot from raw detections (no depth yet)."""
    objects = [MeasuredObject(detection=det) for det in detections]
    return Scene(timestamp=time.time(), objects=objects)


## UTILS
def scene_to_dict(scene) -> dict:
    """Convert Scene dataclass to a JSON-friendly dict."""
    return {
        "timestamp": scene.timestamp,
        "objects": [
            {
                "class": obj.detection.cls_name,
                "score": obj.detection.score,
                "box_xyxy": obj.detection.box_xyxy,
                "depth_m": obj.depth_m,
                "size_cm": obj.size_cm,
                "bearing_rad": obj.bearing_rad,
            }
            for obj in scene.objects
        ],
    }

