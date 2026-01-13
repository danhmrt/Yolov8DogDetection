# YOLOv8 Dog Detection Relay Trigger (Raspberry Pi 5)

![AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

This project runs **YOLOv8 object detection** on a **USB camera** connected to a **Raspberry Pi 5** and **triggers an HTTP-controlled relay** whenever a **dog** is detected in the camera feed.

- Camera: USB (UVC compliant) OR RTSP
- Model: YOLOv8 (COCO pretrained)
- Trigger: HTTP relay (`/relay/on` and `/relay/off`)
- Platform: Raspberry Pi 5 (64-bit Raspberry Pi OS)

---

## Features

- Real-time USB camera feed OR RTSP
- YOLOv8 COCO detection (includes `dog`)
- Relay turns **ON** when a dog is detected
- Relay turns **OFF** after a configurable timeout
- Overlayed bounding boxes and relay status
- Clean shutdown with relay reset

---

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB recommended)
- USB Camera (UVC compatible) OR POE RTSP camera
- Network-accessible HTTP relay
- Stable power supply (≥5V 5A recommended)

---

## Operating System

- **Raspberry Pi OS (64-bit)** – Bookworm or newer
- Python **3.9+**

Verify Python version:
```bash
python3 --version


## Installation

sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  python3-pip \
  python3-venv \
  build-essential \
  libatlas-base-dev \
  libopenblas-dev \
  liblapack-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libopencv-dev \
  v4l-utils \
  ffmpeg \
  curl
### Verify camera is detected
ls /dev/video*

### Expected output:
/dev/video0

### List camera details
v4l2-ctl --list-devices

ffplay /dev/video0

### Python Install
python3 -m venv venv
source venv/bin/activate

### Pip Install
pip install --upgrade pip setuptools wheel

### Ultralytics Install
pip install ultralytics opencv-python requests numpy


## Running application
Enable source
In bash:
- source venv/bin/activate
- python dog_relay_yolov8.py






