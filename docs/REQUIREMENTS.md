# Functional & Non-Functional Requirements: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

---

## 1. Problem Statement Overview
* **Problem Code:** SIH26174
* **Organization:** Indian Space Research Organisation (ISRO)
* **Mission Context:** Bharatiya Antariksh Station (BAS) & microgravity biological/physical science payload experiments.
* **Core Need:** Autonomous, real-time, on-board AI assistant capable of tracking astronaut activity, validating multi-step scientific protocols, detecting procedural errors/skipped steps, and issuing immediate audio/visual feedback without relying on ground mission control.

---

## 2. Functional Requirements (FR)

### FR-1: Multi-Source Video Ingestion
* **FR-1.1:** The system shall ingest video frames from standard local USB/fixed payload webcams.
* **FR-1.2:** The system shall support streaming ingestion via RTSP/HTTP endpoints.
* **FR-1.3:** The system shall support offline playback from pre-recorded MP4/AVI/MKV video files for testing, validation, and post-mission analysis.
* **FR-1.4:** The system shall handle variable input resolutions (720p, 1080p) and frame rates (15–60 FPS) with internal frame rate normalization.

### FR-2: Object & Instrument Detection
* **FR-2.1:** The system shall detect relevant experiment objects (e.g., micropipette, pipette tip box, reagent vial, incubation well plate, centrifuge chamber, glovebox latch/switch, waste bin) with bounding boxes and confidence scores.
* **FR-2.2:** The system shall maintain real-time tracking across brief visual occlusions.

### FR-3: Astronaut Pose & Hand Landmark Tracking
* **FR-3.1:** The system shall estimate 2D/3D full-body skeletal keypoints (torso, arms, shoulders, wrists) of the operating astronaut.
* **FR-3.2:** The system shall track 21 discrete hand landmarks per hand, recognizing wrist orientation, finger extension/flexion, and grasp gestures.
* **FR-3.3:** The system shall function reliably across arbitrary astronaut orientations (upright, tilted, inverted/upside-down) characteristic of microgravity.

### FR-4: Hand-Object Interaction (HOI) Detection
* **FR-4.1:** The system shall determine spatial proximity and geometric intersection between astronaut hands and experiment objects.
* **FR-4.2:** The system shall classify interaction states (e.g., `APPROACHING`, `GRASPING`, `MANIPULATING`, `RELEASED`, `NONE`).

### FR-5: Temporal Human Activity Recognition (HAR)
* **FR-5.1:** The system shall recognize fine-grained atomic actions over sliding temporal windows (e.g., 1.5–3.0 seconds).
* **FR-5.2:** Recognized actions shall include:
  1. Sanitizing workspace / glovebox
  2. Opening incubator / chamber door
  3. Picking sample container
  4. Picking micropipette
  5. Attaching pipette tip
  6. Aspirating reagent from vial
  7. Dispensing reagent into well plate
  8. Ejecting pipette tip into waste
  9. Sealing and placing sample in centrifuge
  10. Closing chamber door and engaging lock

### FR-6: Experiment Sequence Validation & Anomaly Detection
* **FR-6.1:** The system shall validate detected activities against a predefined, configurable protocol state machine / Directed Acyclic Graph (DAG).
* **FR-6.2:** The system shall detect **out-of-sequence steps** (e.g., dispensing before aspirating).
* **FR-6.3:** The system shall detect **skipped steps** (e.g., aspirating reagent without attaching a sterile pipette tip).
* **FR-6.4:** The system shall detect **stalled/timeout steps** (e.g., astronaut paused mid-protocol beyond threshold).
* **FR-6.5:** The system shall dynamically compute and display the **suggested next step** to guide the crew.

### FR-7: Offline Voice & Audio Alerts
* **FR-7.1:** The system shall synthesize real-time voice feedback using 100% offline Text-To-Speech (TTS).
* **FR-7.2:** The system shall issue non-blocking verbal cues for:
  * Successful step completions (optional chime/affirmation).
  * Immediate error warnings (e.g., *"Warning: Step 5 skipped. Attach pipette tip before aspirating"*).
  * Next step guidance prompts.

### FR-8: Structured Audit Telemetry & Logging
* **FR-8.1:** The system shall generate timestamped structured logs (JSON Lines `.jsonl` and CSV).
* **FR-8.2:** Each log record shall contain: `timestamp`, `frame_id`, `active_step_id`, `step_name`, `confidence`, `fsm_state`, `violation_flags`, `astronaut_id`, and `snapshot_filepath`.
* **FR-8.3:** The system shall automatically capture keyframe snapshots upon procedural error detection.

### FR-9: Local Video Recording & Circular Buffer
* **FR-9.1:** The system shall record the raw/annotated camera feed locally in H.264/MP4 format.
* **FR-9.2:** The system shall support rolling circular disk buffers to prevent space station disk exhaustion.

### FR-10: Local IP Video & Telemetry Streaming
* **FR-10.1:** The system shall broadcast a low-latency video stream (MJPEG / RTSP) with overlay telemetry to local station network endpoints (e.g., crew tablets, habitat monitors).

### FR-11: Operator Monitoring GUI
* **FR-11.1:** The system shall provide a native desktop dashboard displaying:
  * Live video feed with toggleable overlays (bounding boxes, skeletal wireframes, HOI lines).
  * Real-time experiment progress checklist with step statuses (`PENDING`, `IN_PROGRESS`, `COMPLETED`, `VIOLATED`).
  * Current action, next expected action, and anomaly banner.
  * Audit log console and session controls (Start, Pause, Reset, Export).

### FR-12: (Optional) Orientation-Agnostic 3D Human Mesh Recovery (HMR)
* **FR-12.1:** The system shall optionally reconstruct 3D human body surface meshes (SMPL) to verify zero-g astronaut posture and reachability.

---

## 3. Non-Functional Requirements (NFR)

### NFR-1: Edge Hardware Resource Constraints
* **NFR-1.1:** The baseline inference pipeline shall operate within **4.0 GB VRAM** (compatible with NVIDIA GeForce GTX 1650).
* **NFR-1.2:** The system shall maintain an inference throughput of **$\ge 20$ FPS** in real-time mode.
* **NFR-1.3:** CPU utilization shall remain below 75% on a 4-core / 8-thread modern CPU.

### NFR-2: Offline Standalone Independence
* **NFR-2.1:** Zero runtime external internet or cloud API dependencies. All models, TTS engines, and runtimes must reside locally.

### NFR-3: Reliability & Safety
* **NFR-3.1:** The system shall not crash or throw unhandled exceptions upon camera disconnection, corrupted frames, or extreme camera occlusions.
* **NFR-3.2:** The protocol validation engine shall be strictly deterministic (zero stochastic hallucination).

### NFR-4: Modularity & Extensibility
* **NFR-4.1:** All perception and recognition modules shall implement abstract interfaces, allowing replacement of underlying models (e.g., swapping YOLOv8 with YOLOv11 or MediaPipe with RTMPose) without altering application logic.
* **NFR-4.2:** Experiment protocols shall be defined via external JSON configuration files without requiring source code modifications.
