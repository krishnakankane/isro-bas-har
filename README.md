# ISRO PS 26174 — AI Human Activity Recognition for On-board BAS Experiments

![Mission Status](https://img.shields.io/badge/Mission-Bharatiya_Antariksh_Station_(BAS)-0B3D91?style=for-the-badge&logo=spacex&logoColor=white)
![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Inference Target](https://img.shields.io/badge/Edge_GPU-GTX_1650_(4GB_VRAM)-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Operation Mode](https://img.shields.io/badge/Operation-100%25_Offline_Autonomous-green?style=for-the-badge)

---

## 1. Project Overview
An autonomous, low-latency, 100% offline Edge-AI assistant designed for the **Bharatiya Antariksh Station (BAS)** to guide astronauts through complex scientific experiments, monitor multi-step protocol compliance, detect skipped/out-of-sequence steps, and issue real-time offline voice guidance.

> **Protocol Notice:** The 10-step biological experiment protocol referenced in documentation and configuration files is a **Synthetic Demonstration Protocol** developed for prototype benchmarking and validation. The validation engine is completely agnostic and JSON-configurable to any official ISRO experiment protocol.

---

## 2. Quick-Start: Live Perception Demo

> **Requires:** `.venv` active, model assets present in `models/mediapipe/` (auto-downloaded on first benchmark run).

### Synthetic test feed (no camera required)
```powershell
.\.venv\Scripts\python.exe -m scripts.run_perception_demo --source synthetic
```

### Webcam (camera index 0)
```powershell
.\.venv\Scripts\python.exe -m scripts.run_perception_demo --source 0
```

### Local video file
```powershell
.\.venv\Scripts\python.exe -m scripts.run_perception_demo --source path\to\video.mp4
```

### Full options
```
--source   Camera index ('0'), video file path, or 'synthetic'  [default: synthetic]
--width    Target frame width in pixels                          [default: 1280]
--height   Target frame height in pixels                        [default: 720]
--fps      Target acquisition FPS                               [default: 30]
```

### Demo controls (while window is open)
| Key | Action |
| :--- | :--- |
| `Q` / `Esc` | Quit |
| `S` | Save current annotated frame → `demo_snapshot.jpg` |
| `P` | Pause / Resume |

### What you will see
- **YOLOv8n** bounding boxes with class name + confidence on all detected COCO objects
- **MediaPipe Pose** 33-point 3D skeletal wireframe (green bones, red/white joints)
- **MediaPipe Hands** 21-point dual-hand graph per hand (with GRASP label when pinch detected)
- **HUD overlay** (top-left) showing live FPS, per-component latency (YOLO / Pose / Hands / Total), resolution, and device

### Measured performance (GTX 1650, 1280×720)
| Metric | Value |
| :--- | :--- |
| YOLO (CUDA) | ~27 ms/frame |
| Pose Lite (CPU/XNNPACK) | ~15 ms/frame |
| Hand (CPU/XNNPACK) | ~17 ms/frame |
| Total pipeline | ~52–60 ms/frame |
| Effective throughput | **~17–20 FPS** |

---

## 3. Run Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 4. Key Architecture & Modules

* **Video Ingestion:** Multi-source reader (Webcam, offline MP4, RTSP stream) with circular buffer and zero-g frame orientation normalizer.
* **Spatial Perception:**
  * Object Detection: Custom-trained lightweight YOLOv8n for space apparatus.
  * Human Pose Estimation: 33 3D body keypoints via MediaPipe Tasks PoseLandmarker (`mediapipe 1.0.1`).
  * Hand Landmark Tracking: 21 3D landmarks per hand via MediaPipe Tasks HandLandmarker.
  * Hand-Object Interaction (HOI): Real-time geometric proximity & grasp engine.
* **Temporal Activity Recognition (HAR):** 60-frame sliding window feature-vector sequence classifier (1D-CNN / Bi-LSTM).
* **Protocol Validation Engine:** Deterministic Finite State Machine (FSM) driven by JSON configuration (`config/experiment_protocols.json`) detecting:
  * Skipped steps
  * Out-of-order execution
  * Dwell-time / duration violations
  * Dynamic next-step suggestion
* **Offline Voice Guidance:** Non-blocking speech synthesis via native Windows SAPI5 (`pyttsx3`) / Piper TTS.
* **Structured Telemetry & Audit Logs:** Millisecond-accurate JSONL stream, SQLite database, and CSV export.
* **Local IP Streaming & Recording:** Local network MJPEG stream with HUD diagnostic overlay and H.264 circular buffer video recording.
* **Mission Control GUI:** PyQt6 hardware-accelerated dark dashboard with live video feed, protocol checklist tree, and telemetry HUD.
* **Optional 3D HMR:** Asynchronous zero-g 3D human body mesh recovery (CLIFF / SMPL).

---

## 5. Documentation Index
* [Model Decision Matrix & Candidate Evaluation](docs/MODEL_DECISION.md)
* [Performance Baseline (Measured)](docs/PERFORMANCE_BASELINE.md)
* [Dataset Strategy & Annotation Plan](docs/DATASET_PLAN.md)
* [System Architecture Specification](docs/ARCHITECTURE.md)
* [Functional & Non-Functional Requirements](docs/REQUIREMENTS.md)
* [Development Roadmap](docs/DEVELOPMENT_PLAN.md)
* [Project Master Context](PROJECT_CONTEXT.md)

