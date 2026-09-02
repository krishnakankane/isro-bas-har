# System Architecture: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

---

## 1. Architectural Philosophy & Design Principles
1. **Edge-First Offline Design:** Every inference, validation, logging, and audio alert component executes entirely on local space station edge compute without internet/cloud connectivity.
2. **Decoupled Asynchronous Pipelining:** Perception, temporal action modeling, state machine evaluation, voice synthesis, video streaming, and GUI rendering run across dedicated threads/workers connected via thread-safe FIFO queues to prevent UI frame stutter or audio blocking.
3. **Strict Determinism in Protocol Validation:** Procedural compliance is governed by an explicit Finite State Machine (FSM) / Directed Acyclic Graph (DAG) rule engine, ensuring zero hallucinations and deterministic next-step recommendations.
4. **Strict Memory Budgeting ($\le 4$ GB VRAM):** Designed specifically for edge GPUs (such as NVIDIA GeForce GTX 1650), keeping spatial perception models quantized (YOLOv8n FP16), leveraging MediaPipe CPU/GPU landmark engines, and utilizing low-dimensional geometric feature representations for temporal activity modeling.
5. **Modular Dependency Inversion:** Every core component implements an abstract base interface (`CameraFeedInterface`, `ObjectDetectorInterface`, `PoseEstimatorInterface`, `HandTrackerInterface`, `ActivityClassifierInterface`, `SequenceValidatorInterface`, `VoiceAlertInterface`, `TelemetryLoggerInterface`), facilitating drop-in replacements.

---

## 2. Complete End-to-End Pipeline Diagram

```
 +---------------------------------------------------------------------------------------------------+
 |                                   INPUT & INGESTION LAYER                                         |
 |  +--------------------+       +----------------------+       +---------------------------------+  |
 |  | USB Payload Camera |  OR   | Offline MP4 Video    |  OR   | Local RTSP Video Stream         |  |
 |  +---------+----------+       +----------+-----------+       +----------------+----------------+  |
 +------------|-----------------------------|------------------------------------|-------------------+
              +-----------------------------+------------------------------------+
                                            |
                                            v
              +------------------------------------------------------------------+
              |               Module 1: Video Ingestion & Buffer                 |
              |   • OpenCV VideoCapture & Threaded Reader Loop                   |
              |   • FPS Throttler (25-30 FPS) & Image Normalizer                 |
              |   • Rolling Circular Frame Buffer & Multi-Consumer Dispatcher    |
              +-----------------------------+------------------------------------+
                                            |
 +------------------------------------------v--------------------------------------------------------+
 |                              SPATIAL PERCEPTION LAYER (Per Frame)                                 |
 |                                                                                                   |
 |   +---------------------------------+ +--------------------------------+ +--------------------+   |
 |   | Module 2: Object Detector       | | Module 3: Pose Estimator       | | Module 4: Hands    |   |
 |   | • YOLOv8n / YOLOv11n (ONNX/FP16)| | • MediaPipe Pose (33 3D kpts)  | | • MediaPipe Hands  |   |
 |   | • Detects: Pipette, Vial, Tip   | | • Torso orientation axis       | | • 21 Landmarks/hand|   |
 |   |   box, Centrifuge, Well Plate   | | • Arm, shoulder, elbow vectors | | • Pinch/grasp flex |   |
 |   +----------------+----------------+ +---------------+----------------+ +---------+----------+   |
 |                    |                                  |                          |                |
 |                    +----------------------------------+--------------------------+                |
 |                                                       |                                           |
 |                                                       v                                           |
 |                               +-----------------------------------------------+                   |
 |                               | Module 5: Hand-Object Interaction (HOI) Engine|                   |
 |                               | • Bounding Box Proximity & Contact Distance   |                   |
 |                               | • Fingertip-to-Centroid Spatial Association   |                   |
 |                               | • Grasp Confidence & Manipulation State      |                   |
 |                               +-----------------------+-----------------------+                   |
 +-------------------------------------------------------|-------------------------------------------+
                                                         |
 +-------------------------------------------------------v-------------------------------------------+
 |                            TEMPORAL RECOGNITION & REASONING LAYER                                 |
 |                                                                                                   |
 |   +-------------------------------------------------------------------------------------------+   |
 |   | Module 6: Temporal Feature Aggregator & Action Classifier                                 |   |
 |   | • 60-frame sliding window of normalized feature vectors (Joint angles + HOI states)       |   |
 |   | • Temporal 1D-CNN / Bi-LSTM / LightGBM Classifier                                         |   |
 |   | • Output: Current Action ID (`ASPIRATING_REAGENT`, `ATTACHING_TIP`, etc.) + Confidence     |   |
 |   +-------------------------------------------+-----------------------------------------------+   |
 |                                               |                                                   |
 |                                               v                                                   |
 |   +-------------------------------------------------------------------------------------------+   |
 |   | Module 7: Experiment Sequence Validation Engine (FSM / DAG)                               |   |
 |   | • Experiment Protocol Graph: Step Definitions, Mandatory Preconditions, Max Timeouts      |   |
 |   | • State Evaluator: Detects `STEP_OK`, `OUT_OF_SEQUENCE`, `STEP_SKIPPED`, `STEP_TIMEOUT`   |   |
 |   | • Next Step Resolver: Computes target `suggested_next_step`                                |   |
 |   +-------------------------------------------+-----------------------------------------------+   |
 +-----------------------------------------------|---------------------------------------------------+
                                                 |
 +-----------------------------------------------v---------------------------------------------------+
 |                              FEEDBACK, TELEMETRY & OUTPUT LAYER                                   |
 |                                                                                                   |
 |   +---------------------------+ +----------------------------+ +------------------------------+   |
 |   | Module 8: Voice Alerts    | | Module 9: Audit Logger     | | Module 10: Video Streamer    |   |
 |   | • pyttsx3 / Piper TTS     | | • JSONL Telemetry Stream   | | • H.264 Local Circular Rec   |   |
 |   | • Async Audio Queue       | | • SQLite Database          | | • Flask/FastAPI MJPEG Stream |   |
 |   | • Verbal Anomaly Cues     | | • CSV Mission Report Export| | • Overlaid HUD Diagnostics   |   |
 |   +---------------------------+ +----------------------------+ +------------------------------+   |
 |                                               |                                                   |
 |                                               v                                                   |
 |   +-------------------------------------------------------------------------------------------+   |
 |   | Module 11: Mission Control Desktop Dashboard (PyQt6)                                      |   |
 |   | • Live Telemetry Video Viewport (Skeleton, BBox, Interaction Lines)                       |   |
 |   | • Protocol Checklist Tree (Real-time Step Status, Checkmarks, Warning Badges)             |   |
 |   | • Telemetry HUD (FPS, Inference Latency, Active Action, Next Step, Anomaly Banner)         |   |
 |   | • Operator Controls (Start, Pause, Reset, Protocol Selector, Export Audit Log)            |   |
 |   +-------------------------------------------------------------------------------------------+   |
 +---------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Subsystem Specifications

### 3.1 Ingestion Subsystem (`src/ingestion/`)
* **Threaded Video Ingestion:** Runs in a dedicated background worker (`CameraFeed`), pulling frames into a bounded queue (`maxsize=5`).
* **Frame Drops:** If the downstream perception pipeline experiences transient load, the oldest frame is discarded to preserve zero-latency real-time alignment.
* **Orientation Normalizer:** Can apply 90°/180°/270° hardware flips or digital rotations to account for zero-g camera mounting configurations.

### 3.2 Perception Subsystem (`src/perception/`)
* **Object Detector (`ObjectDetectorInterface`):** Custom YOLOv8n detector trained/fine-tuned on biological apparatus. Output: List of `DetectedObject` with bounding boxes, class labels, and confidences.
* **Pose Estimator (`PoseEstimatorInterface`):** MediaPipe Pose / YOLOv8n-Pose. Output: `PoseKeypoints` structure containing 33 body landmarks (x, y, z, visibility).
* **Hand Tracker (`HandTrackerInterface`):** MediaPipe Hands. Output: `HandLandmarks` for Left and Right hands with 21 3D landmarks each.
* **Hand-Object Interaction (`HOIDetectorInterface`):** Calculates Euclidean proximity matrix between wrist/index/thumb coordinates and all object bounding boxes. Flags `ContactEvent` when intersection thresholds and dwell-times are satisfied.

### 3.3 Activity Recognition & Protocol Engine (`src/activity_engine/`, `src/core/`)
* **Feature Vectorizer:** Concatenates normalized keypoint angles + hand velocity vectors + binary contact matrix into a 128-dimensional vector per frame.
* **Temporal HAR Classifier:** Processes a 60-frame buffer through a lightweight 1D-CNN or Bi-LSTM.
* **Protocol State Machine (`ProtocolStateMachine`):**
  * Loads protocol definition from JSON (`config/experiment_protocols.json`).
  * Maintains state: `current_step_index`, `completed_steps`, `active_errors`, `step_start_time`.
  * Triggers transition events:
    * `ON_STEP_STARTED`
    * `ON_STEP_COMPLETED`
    * `ON_STEP_SKIPPED`
    * `ON_OUT_OF_SEQUENCE`
    * `ON_TIMEOUT_WARNING`

### 3.4 Feedback & Telemetry Subsystem (`src/alerting/`, `src/telemetry/`, `src/streaming/`)
* **Voice Dispatcher:** Uses a background thread with an audio queue. De-duplicates repeated voice triggers (cooldown window of 3 seconds per alert).
* **Audit Logger:** Writes append-only `.jsonl` records and saves anomaly frame snapshots to disk.
* **Streaming Server:** Exposes an HTTP MJPEG endpoint at `http://<station_ip>:8080/stream` for multi-screen monitoring.

---

## 4. Concurrency & Threading Architecture

| Thread Name | Responsibility | Synchronization Primitive |
| :--- | :--- | :--- |
| **`MainUIThread`** | PyQt6 event loop, widget rendering, user interactions | Qt Signals / Slots |
| **`VideoIngestionThread`** | OpenCV `read()` loop, frame timestamping | Thread-safe `Queue(maxsize=3)` |
| **`PerceptionWorkerThread`** | Object Detection, Pose, Hands, HOI, Temporal HAR, FSM | Ingest Queue -> Perception Result Signals |
| **`VoiceAlertWorkerThread`** | pyttsx3 / Piper synthesis and audio device output | `Queue(maxsize=10)` + Event Cooldown Lock |
| **`StreamingServerThread`** | Flask / FastAPI HTTP MJPEG endpoint socket server | Frame shared memory lock (`threading.Lock`) |
| **`TelemetryWriterThread`** | JSONL disk writes and image snapshot persistence | Non-blocking telemetry queue |
