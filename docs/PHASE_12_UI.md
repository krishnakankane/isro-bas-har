# Phase 12 — Operator UI & Real-Time Mission Dashboard

**ISRO Behavioral Activity & Sequence Monitoring (ISRO BAS-HAR)**  
*Operator Dashboard Architecture & Verification Specification*

---

## 1. System Architecture & Live Data Flow

The Phase 12 Operator UI is a local aerospace-grade mission dashboard connected directly to the unified Python perception, temporal HAR, and deterministic protocol verification engine without duplicating inference:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          UNIFIED BACKEND PIPELINE                              │
│                                                                                │
│   Video Ingestion (Webcam Index 0 / SES_001_NOMINAL.mp4)                       │
│        │                                                                       │
│        ▼                                                                       │
│   Perception Stack (YOLOv8 + MediaPipe Pose + MediaPipe Hands + HOI Engine)    │
│        │                                                                       │
│        ▼                                                                       │
│   154-Dimensional Spatial-Temporal Feature Vector (Per-Frame)                  │
│        │                                                                       │
│        ▼                                                                       │
│   Rolling 60-Frame Temporal Buffer                                             │
│        │                                                                       │
│        ▼                                                                       │
│   Conv1D-BiGRU-Attention Neural Network (models/har/conv1d_bigru_best.pt)      │
│        │                                                                       │
│        ▼                                                                       │
│   Prediction Smoothing & Confidence Filter (>= 65% Threshold)                  │
│        │                                                                       │
│        ▼                                                                       │
│   Deterministic Finite State Machine (HOME_DEMO_PROTOCOL_01)                   │
│        │                                                                       │
│        ▼                                                                       │
│   Rule-Based Anomaly Detection & JSONL Telemetry Logger                        │
└────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI OPERATOR SERVER                                │
│                                                                                │
│   • MJPEG Video Streamer: GET /video_feed                                      │
│   • 20Hz Telemetry WebSocket: WS /ws/telemetry                                 │
│   • Control & State REST APIs: /api/status, /api/start, /api/stop, /api/reset  │
└────────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                 AEROSPACE MISSION OPERATOR DASHBOARD (SPA)                     │
│                                                                                │
│   • Live Annotated Perception & HUD Video Feed (1280x720)                      │
│   • Top Temporal HAR Activity Recognition & Animated Confidence Meter          │
│   • 10-Step Interactive Protocol Stepper (IDLE -> CLOSE_BOX)                   │
│   • Real-Time Anomaly Score Gauge & Categorical Anomaly Monitor                │
│   • Pipeline Latency Breakdown Meters (YOLO / Pose / Hands / HOI / HAR)        │
│   • Chronological Telemetry Transition & Audit Table                           │
│   • Operator Controls (Start Webcam, Demo Video, Stop, Reset Protocol)         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Scientific Statement & AI vs Deterministic Attribution

The UI strictly demarcates AI/ML perceptual components from auditable deterministic protocol safety rules:

* **AI/ML Perceptual Subsystems:**
  - Object Detection: YOLOv8 (`yolov8n.pt`)
  - Human Pose Estimation: MediaPipe Pose (33 3D keypoints)
  - Hand Tracking & Grasp Detection: MediaPipe Hands (21 3D landmarks)
  - Activity Recognition: Conv1D-BiGRU-Attention Neural Network (`models/har/conv1d_bigru_best.pt`)
* **Deterministic Sequence & Safety Rules:**
  - Sequence Graph & Preconditions: Finite State Machine (`config/experiment_protocols.json`)
  - Confidence Gating: Authoritative threshold $\tau = 0.65$
  - Step Timeout & Anomaly Scoring: Deterministic rules $S_{\text{anomaly}} \in [0.0, 1.0]$

---

## 3. REST & WebSocket API Specification

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the single-page mission operator dashboard. |
| `/video_feed` | `GET` | Low-latency MJPEG live annotated stream (`multipart/x-mixed-replace`). |
| `/ws/telemetry` | `WS` | 20Hz WebSocket pushing real-time metrics, FSM state, and telemetry. |
| `/api/status` | `GET` | Returns subsystem readiness, active session metadata, and device info. |
| `/api/protocol`| `GET` | Returns active protocol definition, 10 step statuses, and completion. |
| `/api/anomalies`| `GET` | Returns current anomaly score, FSM status, and anomaly history. |
| `/api/events` | `GET` | Returns chronological telemetry transition records. |
| `/api/performance` | `GET`| Returns real-time FPS and latency breakdown across all perception models. |
| `/api/start` | `POST` | Starts live ingestion (`{"source": "0"|"video_path", "mode": "webcam"|"video"}`). |
| `/api/stop` | `POST` | Stops video ingestion and frees capture hardware. |
| `/api/reset` | `POST` | Resets protocol state machine and telemetry buffers without restarting server. |

---

## 4. How to Launch & Operate the Dashboard

### Launch Command:
From the workspace root directory (`isro-bas-har`):
```bash
python -m src.ui.server --host 127.0.0.1 --port 8000
```

### Auto-Start with Real Demo Video:
```bash
python -m src.ui.server --host 127.0.0.1 --port 8000 --auto_start --video data/raw/videos/SES_001_NOMINAL.mp4
```

### Access in Browser:
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Operator Controls:
* **`START WEBCAM`**: Connects to the local physical camera (index 0) and processes frames live.
* **`DEMO VIDEO (SES_001)`**: Streams `data/raw/videos/SES_001_NOMINAL.mp4` through the real-time recognition pipeline.
* **`STOP`**: Safely terminates active camera/video capture.
* **`RESET PROTOCOL`**: Resets the protocol state machine back to Step 1 (`IDLE`).
* **`CLEAR VIEW`**: Clears the client-side event view table.

---

## 5. Verification & Test Suite Results

* UI Unit & Integration Tests: `tests/test_ui_server.py` (**7 / 7 PASSED**)
* Full Project Regression Test Suite: `pytest -v` (**101 / 101 PASSED (100%)** in 57.04s)

---

## 6. Known Limitations

1. **Browser Camera Permission:** In browser clients, camera capture is handled server-side through OpenCV device access rather than client WebRTC, allowing full GPU hardware acceleration on the server machine.
2. **Single Camera Stream:** Supports one active primary video source (webcam index or video file) per running server instance.
