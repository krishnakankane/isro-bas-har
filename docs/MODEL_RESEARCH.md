# AI Model Research & Comparative Evaluation: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

---

## 1. Research Scope & Constraints
* **Target Hardware:** Edge Space Station Payload Computer (NVIDIA GeForce GTX 1650, 4 GB VRAM, 11th Gen Intel i5 CPU).
* **Operating Environment:** Offline microgravity habitat (Bharatiya Antariksh Station BAS).
* **Throughput Target:** $\ge 20$ FPS real-time processing.
* **Accuracy Target:** High precision on fine-grained laboratory manipulation tasks.

---

## 2. Object Detection for Laboratory Apparatus

### 2.1 Candidate Comparison

| Model Architecture | Parameters | VRAM (FP16) | Inference Latency (GTX 1650) | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n / YOLOv11n** | **3.2M** | **~380 MB** | **~6.5 ms (150 FPS)** | • Extremely fast<br>• Easy fine-tuning on custom classes<br>• Direct ONNX/TensorRT export | • Coarse bounding boxes on heavily overlapping transparent glassware |
| **YOLOv8s** | 11.2M | ~650 MB | ~11.5 ms (85 FPS) | • Higher mAP on small objects | • Slightly higher compute |
| **RT-DETR-R18** | 20M | ~1.2 GB | ~18 ms (55 FPS) | • Transformer-based attention<br>• Excellent on small clustered objects | • Higher VRAM usage |
| **Faster R-CNN (ResNet50-FPN)** | 41M | ~1.8 GB | ~42 ms (24 FPS) | • High precision two-stage detection | • High latency, large memory footprint |

### 2.2 Decision & Justification
* **Selected: `YOLOv8n` (Fine-tuned on BAS apparatus dataset)**
* *Rationale:* Operates well within the 4 GB VRAM budget (<400 MB), yields real-time inference speed (>100 FPS), and supports seamless deployment via ONNX Runtime.

---

## 3. Human Pose Estimation & Hand Tracking

### 3.1 Candidate Comparison

| Component | Model Candidate | Keypoints | Latency | VRAM / Compute | Suitability for Zero-G Posture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Body Pose** | **MediaPipe Pose (BlazePose GHUM 3D)** | 33 3D Keypoints | ~3.5 ms | CPU/GPU (150MB) | **High** (Outputs metric 3D landmarks normalized to torso) |
| Body Pose | YOLOv8n-Pose | 17 Keypoints | ~7.5 ms | GPU (~450MB) | Moderate (2D pixel coordinates, requires manual 3D depth fitting) |
| Body Pose | MMPose RTMPose-m | 17 / 133 Keypoints | ~12.0 ms | GPU (~700MB) | Moderate (Higher compute) |
| **Hands** | **MediaPipe Hands** | 21 3D Landmarks / hand | ~4.5 ms | CPU/GPU (100MB) | **High** (Tracks finger flexion, pinch gestures, palm normal) |
| Hands | YOLOv8-Hand | Hand Bounding Box only | ~6.0 ms | GPU (~400MB) | Low (No discrete joint landmarks for finger manipulation) |
| Hands | HaMeR (3D Hand Mesh) | 3D MANO Mesh | ~55.0 ms | GPU (~2.4GB) | Low for real-time (Exceeds VRAM and latency budget) |

### 3.2 Decision & Justification
* **Selected: `MediaPipe Pose + MediaPipe Hands`**
* *Rationale:* Lightweight, high framerate, multi-platform, capable of running concurrently on CPU/GPU without occupying significant GPU memory.

---

## 4. Hand-Object Interaction (HOI) Modeling

### 4.1 Candidate Comparison

| Approach | Architecture | Latency | Memory | Explainability |
| :--- | :--- | :--- | :--- | :--- |
| **Geometric Proximity & Grasp Heuristic** | Spatial distance matrix between 21 hand landmarks (fingertips, palm) and object 2D bounding boxes + contact persistence | **<1 ms** | **<5 MB** | **100% Deterministic & Explainable** |
| QPIC (Query-based Pairwise Interaction) | DETR Transformer for HOI | ~35 ms | ~2.1 GB | Moderate (Black-box neural attention) |
| Action-CLIP HOI | Dual-stream vision-language embedding | ~50 ms | ~2.8 GB | Low (Prone to zero-shot hallucination) |

### 4.2 Decision & Justification
* **Selected: `Geometric Spatio-Temporal Intersector`**
* *Rationale:* Guaranteed deterministic behavior, sub-millisecond execution, zero VRAM footprint, and complete inspectability for space mission audits.

---

## 5. Temporal Activity Recognition (HAR)

### 5.1 Candidate Comparison

| HAR Approach | Input Representation | Architecture | Latency | Memory | Zero-G Invariance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Feature-Vector Temporal Sequence** | Joint angles + velocities + HOI contact states (128-d per frame) | **Temporal 1D-CNN / Bi-LSTM / LightGBM** | **<2 ms** | **<50 MB** | **High** (Invariant to camera angle and lighting) |
| Spatio-Temporal Graph Conv (ST-GCN) | 2D/3D Skeletal Graph | GCN over spatial & temporal graph | ~12 ms | ~500 MB | High |
| 3D Video CNN (SlowFast-50) | Raw RGB Video Clips (32 frames) | Dual-pathway 3D ResNet | ~45 ms | ~2.2 GB | Moderate (Sensitive to visual background) |
| Video Transformer (VideoMAE) | RGB Video Patches | Spatio-Temporal Attention | ~90 ms | ~3.5 GB (OOM risk) | Moderate |

### 5.2 Decision & Justification
* **Selected: `Pose+HOI Feature Vector Sequence with Temporal 1D-CNN / Bi-LSTM`**
* *Rationale:* Drastically reduces computational overhead, avoids RGB background overfitting, guarantees orientation invariance, and executes in <2 ms on CPU/GPU.

---

## 6. (Optional) 3D Human Mesh Recovery (HMR)

### 6.1 Candidate Comparison
* **CLIFF (SMPL):** Carries camera coordinate orientation directly into SMPL regression; lightweight and robust to cropped frames.
* **4D-Humans (HMR 2.0):** State-of-the-art transformer mesh recovery, but requires ~1.8 GB VRAM.
* **Architecture Strategy:** The 3D HMR module is designed as an **optional asynchronous plug-in**, executed only when full-body 3D biomechanical posture verification is requested by ground telemetry.
