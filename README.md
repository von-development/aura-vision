# Aura Vision

Develop a lightweight computer-vision module for real-time object detection, classification, and distance/size estimation.


## Scope

The goal is a real-time interface returning bounding boxes, class labels, distance, and size.

The final outcome is to have a working demo on a laptop webcam.

Additionally, the project aims to integrate the same pipeline with a Raspberry Pi for the **AURA** project developed in the **Advanced Topics in Robotics** course.

## Features

- Real-time webcam capture
- Object detection with bounding boxes, labels, and confidence scores
- Structured outputs (usable for logging/integration)
- Multiple detector profiles (swap model weights without changing the pipeline)
- Depth and size estimation modules (in development)

## Quickstart

### 1. Install dependencies
This repo uses `uv`.

```bash
uv sync
```


### 2. Run the webcam demo (default profile)

```bash
uv run aura webcam
```



```bash
uv run aura webcam --model yolo/electrocom61_best.pt
```


THIS IS A WIP

