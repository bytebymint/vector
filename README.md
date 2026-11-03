# VECTOR - Real-Time 4K Video Intelligence & Face Recognition Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![OpenCV: 4.13+](https://img.shields.io/badge/OpenCV-4.13%2B-red.svg)](https://opencv.org)
[![Platform: Cross-Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

> High-performance edge video intelligence system featuring real-time 4K/60fps video ingestion, lock-on Kalman face tracking, dual-model neural verification, ESPCN-x4 super-resolution enhancement, and live anonymous person profile generation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [User Interface & Controls](#user-interface--controls)
  - [Split-Screen Studio Mode](#split-screen-studio-mode)
  - [Interactive Playback Controls](#interactive-playback-controls)
  - [Profile Inspector Modal](#profile-inspector-modal)
- [API Reference](#api-reference)
  - [Video Control Endpoints](#video-control-endpoints)
  - [People & Facial Analysis Endpoints](#people--facial-analysis-endpoints)
- [Running Automated Tests](#running-automated-tests)
- [Folder Structure](#folder-structure)
- [License & Authors](#license--authors)

---

## Overview

**VECTOR** is an end-to-end edge AI perception engine engineered for low-latency visual intelligence. Built on modern asynchronous Python (Flask, Flask-SocketIO, PyAV, OpenCV) and vanilla WebComponents, it bridges the gap between ultra-high-definition master video acquisition (up to 4K/8K 60fps) and lightweight, zero-lag browser playback.

The system simultaneously demuxes high-resolution source streams for pixel-perfect AI face crops while piping optimized native MP4/H.264 renditions directly to the browser with zero client decode latency. As subjects move through the scene, a multi-stage pipeline tracks faces continuously until exit, filters false positives using dual-model agreement, super-resolves crops using deep neural upscaling, and maintains dynamic Anonymous Person Profiles in a synchronized dashboard.

---

## Key Features

- **Universal Video Ingestion:**
  - YouTube (VOD & Live) with automatic resolution hierarchy (8K, 4K, 1440p, 1080p).
  - HLS manifests (`.m3u8`), MPEG-DASH, RTSP (TCP), RTMP, and local camera feeds.
  - Automatic format ranking prioritizing stable HLS streams to bypass YouTube HTTP 403 Forbidden token expiration.
- **Hardware-Accelerated Dual-Rendition Pipeline:**
  - Decodes 4K VP9/AV1 master video using hardware acceleration (D3D11VA on Windows, CUDA/VAAPI on Linux).
  - Multiplexes a lightweight 1080p/60fps preview directly into fragmented MP4 chunks (`/api/video/live`) without server re-encoding overhead.
  - Sub-100ms end-to-end latency and frame age.
- **Precision Lock-On Face Tracking:**
  - OpenCV YuNet ONNX runtime with dynamic scale-aware tile scanning.
  - Continuous Kalman filter bounding box projection and Hungarian IoU spatial association.
  - Optical flow forward/backward motion hints to maintain tracking lock across fast camera motion and pans.
  - Bounding boxes lock onto faces continuously until they exit the camera frame entirely.
- **Dual-Model Face Verification:**
  - Two-stage verification pairing YuNet with Caffe Res10 SSD (`face_ssd_fp16.caffemodel`).
  - Spatial IoU cross-validation and probability thresholding eliminating background false alarms.
- **AI Face Enhancement & Super-Resolution:**
  - Integrated ESPCN-x4 deep neural super-resolution model blended with Lanczos-4 resampling.
  - Contrast-Limited Adaptive Histogram Equalization (CLAHE) in CIE-LAB color space for balanced shadow and highlight recovery.
  - Unsharp luminance micro-sharpening eliminating halo artifacts.
- **Anonymous Person Profiles Gallery:**
  - Dynamic gallery displaying verified pedestrian observations with multi-angle thumbnails.
  - Side-by-side inspection of 4K original native crops vs. AI-enhanced super-resolved crops.
  - Interactive comparison modal featuring Side-by-Side, Instant Toggle, and Draggable Split-Slider modes.
  - Quality metrics, pose yaw/pitch estimation, and observation duration tracking.
- **Interactive Playback Controls:**
  - Pause, Resume, Skip +30 seconds, Back 30 seconds, and scrub timeline slider.
  - Drift-compensated clock pacing preserving playback position across seeks and network reconnections.
- **Traffic Intelligence Mode:**
  - Multi-class vehicle detection with ByteTrack and BoT-SORT trackers.
  - Interactive polygon lane and zone drawing editor with perspective-calibrated speed estimation (km/h).

---

## System Architecture

```text
                               +-----------------------------+
                               |   Video Source (YouTube/    |
                               |   HLS/RTSP/4K Camera)       |
                               +--------------+--------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
            [Master 4K Video Stream]                      [1080p Preview Stream]
                       |                                             |
          +------------v------------+                   +------------v------------+
          | Hardware Decoder        |                   | Native Muxer            |
          | (D3D11VA / CUDA / CPU)  |                   | (Fragmented MP4)        |
          +------------+------------+                   +------------+------------+
                       |                                             |
          +------------v------------+                                |
          | FramePacket Publisher   |                                |
          | (Full 4K Image Buffer)  |                                |
          +------------+------------+                                |
                       |                                             |
         +-------------+-------------+                               |
         |                           |                               |
+--------v--------+         +--------v--------+                      |
| Vehicle Tracker |         | Face Engine     |                      |
| (YOLO26 / Byte) |         | (YuNet + SSD)   |                      |
+--------+--------+         +--------+--------+                      |
         |                           |                               |
         |                  +--------v--------+                      |
         |                  | Face Enhancer   |                      |
         |                  | (ESPCN + CLAHE) |                      |
         |                  +--------+--------+                      |
         |                           |                               |
+--------v---------------------------v--------+         +------------v------------+
|  Flask-SocketIO & REST State Dispatcher     |         |  HTTP /api/video/live   |
+---------------------+-----------------------+         +------------+------------+
                      |                                              |
                      +----------------------+-----------------------+
                                             |
                                  +----------v----------+
                                  | Browser Dashboard   |
                                  | - 62% Video HUD     |
                                  | - 38% Profile Feed  |
                                  +---------------------+
```

---

## Prerequisites & Requirements

- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS 13+
- **Python:** 3.10, 3.11, or 3.12 (Python 3.12 recommended)
- **Node.js (Optional):** Node.js 18+ for running frontend headless unit tests
- **Hardware Acceleration (Optional):**
  - NVIDIA GPU with CUDA 11.8+ / 12.x for accelerated PyTorch inference
  - Intel / AMD GPU supporting Direct3D 11 (Windows) or VA-API (Linux) for hardware video decode

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/bytebymint/vector.git
cd vector
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

*Note for NVIDIA CUDA users:* Install the PyTorch build tailored to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) prior to running `pip install -r requirements.txt`.

---

## Quick Start

### Run the Application

From the root repository directory:

```bash
python run.py
```

Or from the `outputs` directory:

```bash
cd outputs
python -u app.py
```

Open your browser at:
```text
http://127.0.0.1:5000
```

To run on a custom port:
```bash
PORT=8080 python run.py
```

---

## User Interface & Controls

### Split-Screen Studio Mode

Switching to **People View** (`02 People counter`) activates the studio layout:
- **Left Column (62% width):** Large video player with live face bounding boxes, tracking labels, corner lock brackets, facial landmarks, and the full video playback HUD.
- **Right Column (38% width):** The **Anonymous Person Profiles** gallery displaying confirmed subject cards with high-resolution original and enhanced face crops, quality scores, duration, and live search/filters.

### Interactive Playback Controls

- **`Pause` / `Resume`:** Pauses the live video buffer; resumes seamlessly with clock synchronization.
- **`-30s` / `+30s`:** Instantly seeks backward or forward by 30 seconds.
- **Scrubbing Slider:** Drag the slider across the timeline to scrub to any timestamp.
- **Keyboard Shortcuts:**
  - `Space` or `k`: Toggle Play / Pause
  - `ArrowLeft` or `j`: Step backward 30 seconds
  - `ArrowRight` or `l`: Step forward 30 seconds
  - `Escape`: Exit fullscreen / dialog views

### Profile Inspector Modal

Clicking any profile card opens the modal inspector:
- **Side-by-Side Mode:** Direct comparison of the 4K original face crop with the ESPCN-x4 super-resolved view.
- **Toggle Mode:** Single-frame toggle button to inspect micro-texture sharpening.
- **Split Slider Mode:** Interactive draggable divider slider comparing original and enhanced pixels.
- **Biometric Telemetry:** Detailed breakdown of resolution, pose yaw/pitch, and verification confidence.

---

## API Reference

### Video Control Endpoints

- **`GET /api/state`**
  - Returns overall stream telemetry, playback position, duration, and current renditions.
- **`POST /api/video/control`**
  - Payload: `{"action": "pause" | "resume" | "seek" | "skip", "seconds": <float>}`
  - Controls video playback and seeks smoothly to target seconds.
- **`GET /api/video/info`**
  - Returns native video transport capabilities and codec information.
- **`GET /api/video/live`**
  - HTTP stream delivering fragmented MP4 chunks directly to HTML5 MediaSource.

### People & Facial Analysis Endpoints

- **`GET /api/people/state`**
  - Returns face pipeline state, active tracks count, total profiles, and processing FPS.
- **`POST /api/analysis/focus`**
  - Payload: `{"mode": "people" | "traffic"}`
  - Toggles processing priorities between face capture and road vehicle tracking.
- **`GET /api/people/profiles`**
  - Query Params: `?limit=50&offset=0&status=all&min_quality=0&sort_by=newest`
  - Returns list of stored Anonymous Person Profiles.
- **`GET /api/people/captures/<capture_id>/image?type=original|enhanced`**
  - Retrieves JPEG binary image of original 4K face crop or super-resolved rendition.

---

## Running Automated Tests

### Python Backend Test Suite

Run unit and integration tests with `pytest`:

```bash
python -m pytest outputs/tests/test_playback_controls.py outputs/tests/test_preview_stream.py outputs/tests/test_face_detector.py -v
```

### Frontend Dashboard Tests (Node.js)

Verify DOM contracts, canvas rendering, and modal mechanics:

```bash
node outputs/tests/test_people_dashboard.cjs
node outputs/tests/test_dashboard_render.cjs
```

---

## Folder Structure

```text
vector/
├── LICENSE                     # MIT Open Source License
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── run.py                      # Root launcher script
├── .gitignore                  # Git exclusion rules
└── outputs/
    ├── app.py                  # Main Flask & Socket.IO server application
    ├── services/               # Core backend services
    │   ├── face_detector.py    # YuNet face detector runtime
    │   ├── face_verifier.py    # Caffe Res10 SSD face verification
    │   ├── face_enhancer.py    # ESPCN-x4 super-resolution & CLAHE
    │   ├── person_tracker.py   # Kalman filter & Hungarian multi-object tracker
    │   ├── face_coverage.py    # Optical flow camera shift & mosaic engine
    │   ├── capture_selector.py # Face quality & angle selection heuristics
    │   ├── profile_repository.py # SQLite profile database & memory store
    │   ├── preview_stream.py   # Low-latency 1080p fragmented MP4 relay
    │   ├── stream_manager.py   # Source acquisition, clock drift & seek manager
    │   └── youtube_source.py   # yt-dlp resolver & HLS format ranker
    ├── models/                 # Pretrained ONNX & Caffe model weights
    ├── static/                 # Frontend assets
    │   ├── css/                # Studio dark dashboard styling
    │   └── js/                 # Dashboard, overlay, timeline, & gallery scripts
    ├── templates/              # Jinja2 HTML templates
    ├── tests/                  # Pytest and Node.js test suites
    └── tools/                  # Offline benchmarks and recording tools
```

---

## License & Authors

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Developed and maintained by **ByteByMint**  
Contact: [info@bytebymint.space](mailto:info@bytebymint.space)
