# src/main.py
import time
import json

import cv2

from .config import laptop_profile
from .detector import YoloDetector

from .utils import build_scene_from_detections, scene_to_dict


def main():
    profile = laptop_profile()
    print(f"[Profile] Using runtime profile: {profile.name}")
    print(f"[Webcam] Using YOLO weights from: {profile.yolo.model_path}")

    detector = YoloDetector(config=profile.yolo)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    print("Press 'q' to quit.")
    start_time = time.time()
    frame_count = 0
    frame_idx = 0
    last_detections = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame from webcam.")
                break

            frame_count += 1
            frame_idx += 1

            # Run YOLO only every `frame_skip` frames
            if frame_idx % profile.frame_skip == 0:
                detections = detector.detect(frame)
                last_detections = detections
            else:
                detections = last_detections

            # Build Scene object
            scene = build_scene_from_detections(detections)

            # (Optional) debug-print JSON every 30 frames
            if frame_idx % 30 == 0:
                scene_dict = scene_to_dict(scene)
                print(json.dumps(scene_dict, indent=2))

            # Draw detections
            for obj in scene.objects:
                det = obj.detection
                x1, y1, x2, y2 = det.box_xyxy
                label = f"{det.cls_name} {det.score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # FPS overlay
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("YOLO11 Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()