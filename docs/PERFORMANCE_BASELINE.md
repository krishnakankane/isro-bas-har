# Performance Baseline: ISRO BAS HAR Perception Pipeline

> **Document Status:** Phase 2 — Initial MediaPipe Tasks API Baseline  
> **Measurement Date:** 2026-08-31  
> **Benchmark Type:** Synthetic-feed (blank 1280×720 frames, no person visible — represents worst-case overhead without detection hits)

---

## 1. Hardware & Software Platform

| Parameter | Value |
| :--- | :--- |
| **Operating System** | Windows 11 Home 64-bit (Build 10.0.22631) |
| **CPU** | Intel Core i5-11300H @ 3.10 GHz (4 Cores, 8 Logical Processors) |
| **GPU** | NVIDIA GeForce GTX 1650 — 4 GB GDDR6 VRAM |
| **GPU Driver** | 529.04 |
| **CUDA Runtime** | 12.1 (via `torch 2.5.1+cu121`) |
| **CUDA Compute Capability** | SM 7.5 (Turing) |
| **Python Runtime** | 3.10.11 (64-bit, `.venv`) |
| **PyTorch** | 2.5.1+cu121 |
| **MediaPipe** | 1.0.1 |
| **Ultralytics** | 8.4.136 |
| **OpenCV** | 5.0.0.93 |

---

## 2. MediaPipe Tasks API Details

### API Version & Migration Status

| Item | Value |
| :--- | :--- |
| **MediaPipe Version** | `1.0.1` |
| **API Namespace** | `mediapipe.tasks.python.vision` |
| **Running Mode** | `RunningMode.IMAGE` (synchronous, frame-by-frame) |
| **Backend** | TensorFlow Lite + XNNPACK CPU Delegate |
| **Legacy API Removed** | `mediapipe.solutions.pose` / `mediapipe.solutions.hands` (removed in MP 1.0+) |
| **Migration Status** | ✅ Complete — both modules use modern Tasks API |

### Pose Estimator

| Item | Value |
| :--- | :--- |
| **Class** | `mediapipe.tasks.python.vision.PoseLandmarker` |
| **Options** | `PoseLandmarkerOptions` |
| **Running Mode** | `RunningMode.IMAGE` |
| **Active Model** | `pose_landmarker_lite.task` |
| **Comparison Model** | `pose_landmarker_full.task` |
| **Landmarks Output** | 33 × 3D (x, y, z, visibility) |
| **Initialization** | `PoseLandmarker.create_from_options(options)` |

### Hand Tracker

| Item | Value |
| :--- | :--- |
| **Class** | `mediapipe.tasks.python.vision.HandLandmarker` |
| **Options** | `HandLandmarkerOptions` |
| **Running Mode** | `RunningMode.IMAGE` |
| **Active Model** | `hand_landmarker.task` |
| **Max Hands** | 2 (Dual-hand simultaneous tracking) |
| **Landmarks Output** | 21 × 3D (x, y, z) per hand + handedness label + pinch/grasp heuristic |
| **Initialization** | `HandLandmarker.create_from_options(options)` |

---

## 3. Model Assets

All model assets are stored in `models/mediapipe/` within the project root. No internet access required after download.

| Filename | Type | Size (MB) | Source URL |
| :--- | :--- | :--- | :--- |
| `pose_landmarker_lite.task` | MediaPipe Task Bundle (TFLite) | 5.51 | `storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/` |
| `pose_landmarker_full.task` | MediaPipe Task Bundle (TFLite) | 8.96 | `storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/` |
| `hand_landmarker.task` | MediaPipe Task Bundle (TFLite) | 7.46 | `storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/` |
| `yolov8n.pt` | Ultralytics PyTorch COCO Pretrained | ~6.0 | Auto-downloaded by `ultralytics` on first use |

> **Offline Capability:** Once assets are present in `models/mediapipe/`, no network access is required for any inference. The system is fully self-contained and offline-capable.

---

## 4. Measured Latency — REAL BENCHMARK RESULTS

> **Benchmark Conditions:**
> - Resolution: 1280×720 (HD)
> - Iterations: 50 timed runs (after 10 warm-up passes)
> - Input: Synthetic blank frames (no detectable objects/persons — measures pure model overhead)
> - All measurements are wall-clock time using `time.perf_counter()`

### Per-Component Latency (50 iterations, 1280×720)

| Component | Device | Mean (ms) | Std Dev (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n Object Detection** | CUDA (GTX 1650) | **27.04** | ±6.99 | 24.72 | 48.24 | 23.11 | 50.40 |
| **Pose Landmarker Lite** | CPU / XNNPACK | **15.46** | ±1.33 | 15.33 | 17.57 | 12.88 | 18.67 |
| **Pose Landmarker Full** | CPU / XNNPACK | **15.39** | ±1.54 | 15.07 | 18.23 | 13.07 | 18.67 |
| **Hand Landmarker (Dual)** | CPU / XNNPACK | **16.85** | ±1.74 | 16.06 | 20.26 | 14.43 | 20.65 |
| **Total Pipeline (Lite+Hands)** | Hybrid | **59.35** | ±7.86 | — | 79.78 | — | — |

### Throughput

| Metric | Value |
| :--- | :--- |
| **Synchronous Pipeline FPS** | **16.8 FPS** (mean latency basis) |
| **P95 Throughput** | ~12.5 FPS |
| **GPU VRAM Allocated** | 12.09 MB (YOLO runtime) |
| **GPU VRAM Reserved** | 60.00 MB |
| **CPU Utilization** | ~10.1% system-wide during benchmark |

---

## 5. Notes & Limitations

| Item | Detail |
| :--- | :--- |
| **Blank-frame bias** | Benchmark uses synthetic blank frames with no human present. MediaPipe will not run full landmark processing when no person/hand is detected; actual per-frame latency with a real person visible may be **slightly higher** (estimate: +3–8 ms for Pose, +2–5 ms for Hands). |
| **YOLOv8n CUDA variance** | High std dev (±6.99 ms) and P95 (48.24 ms) for YOLO indicate CUDA kernel launch overhead on the GTX 1650. CUDA warm-up was performed (10 iterations). |
| **MediaPipe XNNPACK Delegate** | Both Pose and Hand run on CPU via the TFLite XNNPACK delegate. Lite and Full models show similar latency (~15 ms) on synthetic frames because both converge quickly on empty input. Lite is preferred for real-time operation. |
| **MP 1.0 TFLite warnings** | `inference_feedback_manager.cc: Disabling support for feedback tensors` is a benign TFLite log, not an error. Suppressed in production via stderr redirection. |
| **`landmark_projection_calculator` warning** | `Using NORM_RECT without IMAGE_DIMENSIONS is only supported for square ROI` — this is a known MediaPipe TFLite diagnostic for non-square input. Does not affect landmark accuracy at standard resolutions. |
| **No GPU acceleration for MediaPipe** | MediaPipe Tasks 1.0 on Windows does not currently expose a GPU delegate for the PoseLandmarker / HandLandmarker Tasks API (GPU delegate is supported on Android/iOS only). CPU/XNNPACK is the intended Windows runtime target. |
| **FPS ceiling** | Pipeline runs synchronously at 16.8 FPS mean. Asynchronous frame-skip or parallel CPU/GPU execution of Pose+Hand in a thread pool would increase throughput. This is scheduled for Phase 3 optimization. |

---

## 6. Previous Theoretical Estimates vs. Measured Reality

| Component | Theoretical Estimate | Measured Reality | Ratio |
| :--- | :--- | :--- | :--- |
| Object Detection (YOLO) | ~6.5 ms | 27.04 ms | 4.2× slower |
| Pose Estimation | ~3.5 ms | 15.46 ms | 4.4× slower |
| Hand Tracking | ~4.5 ms | 16.85 ms | 3.7× slower |
| **Total Pipeline** | **~14.5 ms (69 FPS)** | **59.35 ms (16.8 FPS)** | **4.1× slower** |

> The theoretical estimates were derived from datacenter GPU benchmarks. The GTX 1650 CUDA Turing architecture and Windows XNNPACK runtime show predictably higher latency. **The system remains real-time capable** for offline BAS experiments where 15–20 FPS is sufficient for activity monitoring.
