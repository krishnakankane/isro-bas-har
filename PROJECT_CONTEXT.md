# Project Context: ISRO PS 26174 — AI Human Activity Recognition for On-board BAS Experiments

## 1. Project Objective & Context
* **Challenge:** Smart India Hackathon 2026 / ISRO Problem Statement 26174.
* **Context:** The **Bharatiya Antariksh Station (BAS)** is India's upcoming space station program. During scientific and biological payload operations, communication latency and orbital blackout periods prevent real-time guidance from ground telemetry/control.
* **Goal:** Develop a completely standalone, offline, low-latency AI-based Human Activity Recognition (HAR) and experiment workflow validation assistant for on-board BAS experiments.
* **Important Protocol Clarification:** The 10-step biological experiment protocol utilized in documentation and configuration files is a **Synthetic Demonstration Protocol** developed for prototype benchmarking and validation. ISRO PS 26174 does not mandate a specific biological experiment; the state machine and perception architecture are completely agnostic and JSON-configurable to any official ISRO experiment protocol.
* **Core Capabilities:**
  1. Fixed-camera video ingestion (webcam, RTSP, offline video files).
  2. Multi-modal spatial perception: Object detection, Astronaut pose estimation, Hand tracking, and Hand-Object Interaction (HOI).
  3. Temporal activity recognition across fine-grained experiment sub-tasks.
  4. Finite State Machine (FSM) / DAG-based experiment sequence validator (detecting skipped, repeated, or out-of-order steps and recommending the correct next step).
  5. Low-latency offline voice alerts (Text-To-Speech).
  6. Timestamped structured experiment telemetry & audit logging (JSONL/CSV).
  7. Local video recording with analytical visual overlays (bounding boxes, skeletal wireframes, step progress HUD).
  8. Local IP streaming (RTSP/WebRTC/HTTP MJPEG) for station-wide crew monitoring.
  9. Modern, responsive, mission-control monitoring GUI.
  10. (Optional) Orientation-agnostic 3D Human Mesh Recovery (HMR) for microgravity postures.

---

## 2. Hardware & Environment Baseline
* **Host Operating System:** Microsoft Windows 11 Home Single Language (64-bit, Build 22631)
* **Processor (CPU):** 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10 GHz (4 Cores, 8 Logical Processors)
* **Dedicated GPU:** NVIDIA GeForce GTX 1650 (4 GB GDDR6 VRAM, Driver 529.04, Max CUDA 12.0)
* **Integrated GPU:** Intel(R) Iris(R) Xe Graphics (1 GB shared)
* **Storage Available:** > 394 GB total across drives C, D, E.
* **Determined Python Runtime:** **Python 3.10.x** (64-bit). (Optimal shared compatibility across PyTorch CUDA, MediaPipe, Ultralytics, and 3D SMPL mesh tools).
* **Current Execution Gate:** Review of `MODEL_DECISION.md` and `DATASET_PLAN.md`. No models trained, no heavy frameworks installed, and no GUI/streaming code generated prior to explicit user approval.

---

## 3. Current Documentation & Scaffold State

* [x] Environment and hardware capability assessment.
* [x] Synthetic demonstration protocol clarification in config and documentation.
* [x] AI Model Decision Matrix ([docs/MODEL_DECISION.md](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/docs/MODEL_DECISION.md)).
* [x] Dataset Strategy & Annotation Plan ([docs/DATASET_PLAN.md](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/docs/DATASET_PLAN.md)).
* [x] System Architecture ([docs/ARCHITECTURE.md](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/docs/ARCHITECTURE.md)).
* [x] Requirements Specification ([docs/REQUIREMENTS.md](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/docs/REQUIREMENTS.md)).
* [x] Modular abstract interface definitions ([src/core/interfaces.py](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/src/core/interfaces.py)).
* [x] Deterministic Protocol State Machine with anomaly detection ([src/core/state_machine.py](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/src/core/state_machine.py)).
* [x] Core unit tests ([tests/test_interfaces.py](file:///C:/Users/krish/.gemini/antigravity-ide/scratch/isro-bas-har/tests/test_interfaces.py)).

---

## 4. Model Decision Summary

| Pipeline Component | Final Candidate & Recommendation | Compute Footprint on GTX 1650 | Offline Capability | Custom Training Needed? |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection** | `YOLOv8n` / `YOLO11n` (Ultralytics) | ~6.5 ms, ~380 MB VRAM | 100% Offline (.pt / ONNX) | **Yes** (Apparatus dataset) |
| **Pose Estimation** | `MediaPipe Pose` (BlazePose GHUM 3D) | ~3.5 ms, ~150 MB (CPU/GPU) | 100% Offline | **No** (Pretrained) |
| **Hand Tracking** | `MediaPipe Hands` (21 3D landmarks) | ~4.5 ms, ~100 MB | 100% Offline | **No** (Pretrained) |
| **HOI Spatial Engine**| Geometric Distance Matrix & Heuristic | <0.5 ms, 0 MB VRAM | 100% Offline | **No** (Calibrated logic) |
| **Temporal HAR** | Feature Vector (128-d) + 1D-CNN / Bi-LSTM | <1.5 ms, <30 MB | 100% Offline | **Yes** (Trained on features) |
| **Sequence Validation**| Deterministic Protocol State Machine | <0.2 ms, 0 MB VRAM | 100% Offline | **No** (JSON-configured) |
| **Voice Alerts** | `pyttsx3` (Windows SAPI5) / Piper TTS | <30 ms, 0 MB VRAM | 100% Offline | **No** (Native OS) |
| **(Optional) 3D HMR** | `CLIFF` (SMPL Mesh Recovery) | ~25 ms, ~1.1 GB VRAM | 100% Offline | **No** (Pretrained) |

---

## 5. Next Implementation Task (Pending User Approval)
Upon approval of `MODEL_DECISION.md` and `DATASET_PLAN.md`, we will provision Python 3.10 and establish the lightweight test fixture pipeline.
