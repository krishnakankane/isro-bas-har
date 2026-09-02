# Development & Phased Implementation Plan: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

---

## 1. Roadmap Overview

```
 +-------------------------------------------------------------------------------+
 | Phase 1: Environment Inspection, Baseline Architecture & Docs [CURRENT]       |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 2: Ingestion, Streaming & Mission Control Dashboard GUI Shell           |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 3: Spatial Perception Pipeline (Objects, 2D Pose, Hands, HOI)          |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 4: Temporal HAR & Sequence Validation FSM Engine                        |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 5: Voice Alerts, Telemetry Audit Logging & Video Archival               |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 6: Edge Optimization (FP16/ONNX) & Optional 3D HMR Plug-in              |
 +---------------------------------------+---------------------------------------+
                                         |
                                         v
 +-------------------------------------------------------------------------------+
 | Phase 7: End-to-End System Testing & Mock Space Mission Demonstration         |
 +-------------------------------------------------------------------------------+
```

---

## 2. Phase-by-Phase Breakdown

### Phase 1: Environment & Baseline Setup *(In Progress / Completing Now)*
* **Objective:** Establish formal technical documentation, modular repository structure, configuration definitions, and abstract interfaces.
* **Deliverables:**
  * System architecture documentation (`docs/ARCHITECTURE.md`).
  * Functional & non-functional requirements (`docs/REQUIREMENTS.md`).
  * AI model evaluation and selection report (`docs/MODEL_RESEARCH.md`).
  * `PROJECT_CONTEXT.md` and `README.md`.
  * Python dependency definitions (`requirements.txt`, `pyproject.toml`).
  * Base interfaces and abstract contract classes in `src/core/interfaces.py`.
  * Deterministic protocol state machine specification in `config/experiment_protocols.json`.

---

### Phase 2: Ingestion, Streaming & Mission Control GUI Shell
* **Objective:** Build the real-time video foundation and interactive operator dashboard.
* **Tasks:**
  * Implement `CameraFeed` (supporting Webcam index, offline video files, and RTSP streams) with a threaded reader and FPS throttling.
  * Build PyQt6 dark-themed mission control interface with:
    * Live video display with telemetry bounding boxes and skeletal toggle.
    * Real-time 10-step protocol checklist widget.
    * Anomaly alert banner with visual warning badges.
    * Start / Pause / Reset controls and file playback picker.
  * Integrate Flask/FastAPI HTTP MJPEG local streaming server.
  * Implement local H.264 video recorder with circular buffer management.

---

### Phase 3: Spatial Perception Pipeline (Objects, Pose, Hands, HOI)
* **Objective:** Implement real-time frame-level perception models.
* **Tasks:**
  * Implement `ObjectDetector` using YOLOv8n (detecting pipettes, vials, tip boxes, centrifuges, well plates, switches).
  * Implement `PoseEstimator` using MediaPipe Pose (33 3D body keypoints).
  * Implement `HandTracker` using MediaPipe Hands (21 3D landmarks per hand).
  * Implement `HOIDetector` (computing spatial proximity, bounding box overlap, fingertip contact, and grasp duration).
  * Unit test frame-level perception pipeline on sample video frames.

---

### Phase 4: Temporal HAR & Sequence FSM Engine
* **Objective:** Recognize multi-frame actions and strictly validate experiment protocol sequences.
* **Tasks:**
  * Build feature vector extractor aggregating joint angles, hand velocities, and HOI contact flags across a 60-frame sliding window.
  * Implement `TemporalHARClassifier` mapping feature sequences to action classes (e.g., `ASPIRATING_REAGENT`, `ATTACHING_TIP`).
  * Implement `ProtocolStateMachine` parsing `config/experiment_protocols.json`:
    * Verify step order.
    * Detect skipped steps (e.g. aspirating without attaching tip).
    * Detect out-of-order execution.
    * Compute dynamic next-step suggestion.
  * Unit test FSM against simulated action streams (both valid and erroneous sequences).

---

### Phase 5: Offline Voice Alerts, Telemetry & Logging
* **Objective:** Implement hands-free audio guidance and tamper-evident audit logging.
* **Tasks:**
  * Implement asynchronous `VoiceAlertDispatcher` using `pyttsx3` (Windows SAPI5) with alert deduplication and cooldown timers.
  * Implement `AuditLogger` writing millisecond-precision JSON Lines (`.jsonl`) logs and exporting CSV mission summaries.
  * Implement automated snapshot trigger saving marked JPEG frames upon procedural violation.

---

### Phase 6: Edge Optimization & Optional 3D HMR
* **Objective:** Maximize pipeline throughput on edge hardware and demonstrate 3D zero-g posture recovery.
* **Tasks:**
  * Export YOLOv8 to ONNX / FP16 TensorRT for low latency.
  * Implement optional modular 3D Human Mesh Recovery (CLIFF / SMPL) submodule running as an isolated worker.
  * Benchmark end-to-end latency and VRAM consumption on NVIDIA GeForce GTX 1650.

---

### Phase 7: End-to-End Integration & Space Mission Demo
* **Objective:** Validate the complete integrated assistant under realistic simulated laboratory conditions.
* **Tasks:**
  * Execute end-to-end benchmark scenarios:
    1. Perfect nominal run (all 10 steps executed in exact order).
    2. Skipped step run (Step 5 skipped; system issues voice alert and flags anomaly).
    3. Out-of-order run (Step 7 executed before Step 6; system triggers warning and updates telemetry).
    4. Timeout run (Astronaut stalls during step; system prompts next action).
  * Generate final audit logs, video recordings, and verification walkthrough.
