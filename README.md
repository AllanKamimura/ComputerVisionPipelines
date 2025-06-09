# Computer Vision Pipelines

A comprehensive framework for deploying optimized computer vision models on embedded systems using TensorFlow Lite and GStreamer.

## Overview

This project provides a collection of computer vision pipelines for real-time object detection, depth estimation, and image classification on resource-constrained devices. It leverages TensorFlow Lite for efficient model inference and GStreamer for video capture and streaming.

Key features:
- Hardware-accelerated inference on embedded devices
- Support for multiple platforms (Toradex, Jetson, Raspberry Pi)
- Real-time object detection using YOLO models
- Depth estimation using MiDaS models
- RTSP streaming capabilities
- Configurable via environment variables

## Academic Reference

This project is based on the following academic publication:

> Kamimura, A. H. (2024). Study and implementation of an embedded image processing system in autonomous inspection robots. University of São Paulo.
> Available at: [https://bdta.abcd.usp.br/item/003240088](https://bdta.abcd.usp.br/item/003240088)

## Supported Models

### Object Detection
- YOLOv11n (640x640)
- YOLOv11s (320x320)
- YOLOv5n (320x320)
- YOLOv5nu (640x640)
- YOLOv5su (320x320)
- SSD MobileNet v1/v2

### Depth Estimation
- MiDaS v2.1 Small
- FastDepth

## Requirements

### Software Dependencies
- Python 3
- OpenCV
- TensorFlow Lite Runtime
- GStreamer (with Python bindings)
- AI Edge Lite Runtime

### Hardware
The project is optimized for the following platforms:
- Toradex Verdin i.MX8M Plus
- NVIDIA Jetson
- Raspberry Pi

## Installation

### Using Docker (Recommended)

The project includes a Dockerfile that sets up all necessary dependencies:

```bash
# Build the Docker image
docker build -t cv-pipelines .

# Run the container
docker run --device=/dev/video0:/dev/video0 -it cv-pipelines
```

## Usage

### Environment Variables

The pipelines can be configured using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| USE_HW_ACCELERATED_INFERENCE | Enable hardware acceleration | 1 (True) |
| MINIMUM_SCORE | Minimum confidence score for detections | 0.55 |
| CAPTURE_DEVICE | Video capture device | /dev/video2 |
| CAPTURE_RESOLUTION_X | Capture width | 640 |
| CAPTURE_RESOLUTION_Y | Capture height | 480 |
| CAPTURE_FRAMERATE | Capture framerate | 30 |
| STREAM_BITRATE | RTSP stream bitrate | 2048 |

### Running Object Detection

```bash
# Run YOLO object detection
python src/yolo11n_640.py

# Run SSD MobileNet object detection
python src/ssd-mobilenet-v2.py
```

### Running Depth Estimation

```bash
# Run MiDaS depth estimation
python src/midas-tflite-v2-1-small-lite-v1.py

# Run FastDepth
python src/fastdepth.py
```

## Platform-Specific Implementations

The repository includes platform-specific implementations for different hardware:

- Standard: `*.py` (e.g., `yolo11n_640.py`)
- Jetson: `*-jetson.py` (e.g., `yolo11n_640-jetson.py`)
- Raspberry Pi: `*-raspi.py` (e.g., `yolo11n_640-raspi.py`)

Choose the appropriate implementation based on your target hardware.

## Building TensorFlow Lite with Hardware Acceleration

The `recipes` directory contains scripts for building TensorFlow Lite with hardware acceleration support:

- `nn-imx_1.3.0.sh`: Neural Network Acceleration for i.MX processors
- `tim-vx.sh`: TIM-VX acceleration library
- `tensorflow-lite_2.9.1.sh`: TensorFlow Lite build script
- `tflite-vx-delegate.sh`: VX delegate for TensorFlow Lite

## License

See the [LICENSE](LICENSE) file for details.
