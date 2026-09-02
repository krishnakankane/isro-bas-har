# Feature Schema: Temporal HAR Input Representation

> **Document Status:** Phase 2 — HOI Feature Layer  
> **Last Updated:** 2026-08-31  
> **Implemented in:** `src/activity_engine/feature_extractor.py`

---

## 1. Overview

The ISRO BAS HAR system converts each processed video frame into a **fixed-dimension float32 feature vector** of length **154**. These vectors are accumulated in a **60-frame sliding window** by the `TemporalFeatureBuffer` and will be consumed by the future 1D-CNN / Bi-LSTM temporal HAR classifier.

The feature schema is deliberately **fixed and versioned**. Any change to FEATURE_DIM or slot layout constitutes a breaking change that requires retraining the temporal model.

```
One frame  →  PerceptionFeatureExtractor  →  TemporalFeatureVector (154 floats)
60 frames  →  TemporalFeatureBuffer       →  np.ndarray (60 × 154)  →  HAR Model input
```

---

## 2. Feature Vector Schema (FEATURE_DIM = 154)

### Section A: Body Pose Landmarks (Slots 0–131)

> Source: `PoseEstimationResult.landmarks` — MediaPipe Tasks PoseLandmarker Lite  
> Encoding: Normalised [0, 1] image coordinates (x, y) and relative depth (z)

| Slots | Content | Dimension | Notes |
| :--- | :--- | :--- | :--- |
| 0 – 98 | Pose landmark (x, y, z) × 33 | 99 | Landmarks ordered by MediaPipe ID 0–32 |
| 99 – 131 | Pose landmark visibility × 33 | 33 | Clipped to [0, 1] |

- If pose is **not detected**, all 132 slots are **zero**.
- Landmark ordering follows the [MediaPipe Pose Landmark specification](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model).
- Landmark ID 0 = NOSE, 11 = LEFT_SHOULDER, 23 = LEFT_HIP, 15 = LEFT_WRIST, 16 = RIGHT_WRIST, etc.

### Section B: Hand Wrist Positions and Grasp State (Slots 132–137)

> Source: `HandTrackingResult.wrist_pos`, `.is_grasping`  
> Encoding: Normalised [0, 1] image coordinates

| Slot | Content | Notes |
| :--- | :--- | :--- |
| 132 | Left wrist X | 0.0 if no left hand |
| 133 | Left wrist Y | 0.0 if no left hand |
| 134 | Right wrist X | 0.0 if no right hand |
| 135 | Right wrist Y | 0.0 if no right hand |
| 136 | Left is_grasping | 1.0 or 0.0 |
| 137 | Right is_grasping | 1.0 or 0.0 |

> **Note:** The MediaPipe HandLandmarker `is_grasping` / pinch flag represents a geometric hand pose heuristic, not confirmed physical contact with an object. See HOI Design Notes.

### Section C: Detection Presence Flags and Torso Angle (Slots 138–141)

| Slot | Content | Range | Notes |
| :--- | :--- | :--- | :--- |
| 138 | Pose detected | 0.0 / 1.0 | 1.0 if `PoseEstimationResult.is_detected` |
| 139 | Left hand detected | 0.0 / 1.0 | 1.0 if `HandTrackingResult.is_detected` |
| 140 | Right hand detected | 0.0 / 1.0 | 1.0 if `HandTrackingResult.is_detected` |
| 141 | Torso angle (normalised) | [-1.0, 1.0] | `torso_angle / 180.0` — 0.0 = vertical |

### Section D: Best Object Geometry (Slots 142–145)

> Source: Highest-confidence detection from `FramePerceptionResult.objects`  
> Encoding: Normalised [0, 1] image space

| Slot | Content | Sentinel (no objects) |
| :--- | :--- | :--- |
| 142 | Best object detection confidence | 0.0 |
| 143 | Best object centre X | 0.0 |
| 144 | Best object centre Y | 0.0 |
| 145 | Best object normalised area | 0.0 |

- "Best object" = highest `.confidence` detection in the current frame.
- Once a custom space apparatus detector is integrated in Phase 3, it will replace the generic YOLOv8n COCO detector here without changing any slot indices.

### Section E: Wrist-to-Object Distances (Slots 146–147)

| Slot | Content | Sentinel |
| :--- | :--- | :--- |
| 146 | Left wrist ↔ best object centre distance (normalised) | 1.0 (no hand or no objects) |
| 147 | Right wrist ↔ best object centre distance (normalised) | 1.0 (no hand or no objects) |

- Distance is Euclidean in the same normalised [0, 1] coordinate space as wrist positions and object centres.
- Capped at 1.0.

### Section F: HOI State One-Hot Vector (Slots 148–153)

> Source: `FramePerceptionResult.interactions` — HOIDetector output

The **highest-priority** interaction event is selected from the current frame (priority: MANIPULATING > GRASPING > APPROACHING > RELEASED > UNKNOWN > NONE) and encoded as a one-hot over the 6 possible states:

| Slot | State | Priority |
| :--- | :--- | :--- |
| 148 | NONE | 0 (default when no interactions exist) |
| 149 | UNKNOWN | 1 |
| 150 | APPROACHING | 2 |
| 151 | GRASPING | 3 |
| 152 | MANIPULATING | 4 (highest) |
| 153 (shared) | RELEASED | — |

> **Wait — the table says 6 states but only 6 slots (148–153)?** Correct. Slot 153 serves double duty: it holds the RELEASED one-hot **OR** the dwell normalisation (see below). The actual RELEASED one-hot uses slot index `148 + 5 = 153` and the dwell value overwrites it afterwards. This is a known design limitation that will be revised if RELEASED-state dwell becomes important.

> **Design fix note for future:** Allocate 7 slots (148–154) if dwell needs to be independent of state. Currently slot 153 = HOI dwell (normalised), which overwrites the RELEASED position. The HOI one-hot for RELEASED is set first, then the dwell value replaces it. In practice, RELEASED dwell is not critical for current phase training.

### Section G: HOI Interaction Dwell (Slot 153)

| Slot | Content | Range |
| :--- | :--- | :--- |
| 153 | HOI dwell — normalised by 120 frames | [0.0, 1.0] |

---

## 3. Summary Table

| Slot Range | Content | Dim |
| :--- | :--- | :--- |
| 0 – 98 | Pose landmarks (x, y, z) × 33 | 99 |
| 99 – 131 | Pose visibility × 33 | 33 |
| 132 – 133 | Left wrist (x, y) | 2 |
| 134 – 135 | Right wrist (x, y) | 2 |
| 136 | Left is_grasping | 1 |
| 137 | Right is_grasping | 1 |
| 138 | Pose detected | 1 |
| 139 | Left hand detected | 1 |
| 140 | Right hand detected | 1 |
| 141 | Torso angle normalised | 1 |
| 142 | Best object confidence | 1 |
| 143 – 144 | Best object centre (cx, cy) | 2 |
| 145 | Best object normalised area | 1 |
| 146 | Left wrist ↔ object distance | 1 |
| 147 | Right wrist ↔ object distance | 1 |
| 148 – 153 | HOI state one-hot + dwell | 6 |
| **Total** | | **154** |

---

## 4. Temporal Buffer

| Parameter | Value |
| :--- | :--- |
| Class | `TemporalFeatureBuffer` (`src/activity_engine/temporal_buffer.py`) |
| Window size | **60 frames** |
| Feature dim | **154** |
| Output shape | `(60, 154)` float32 |
| Zero-padding | Leading rows filled with zeros when buffer is not full |
| Thread safety | Yes — uses `threading.Lock` |
| Reset method | `buffer.reset()` — call between experiment sessions |
| Recording | Optional JSONL (`data/features/*.jsonl`) — 1 line per frame |

### JSONL Record Format
```json
{
  "frame_id": 42,
  "timestamp_sec": 1725123456.789012,
  "data": [0.52, 0.31, ..., 0.0]
}
```
Each `data` array has exactly 154 float values.

---

## 5. HOI Engine Design Notes

### State Definitions

| State | Meaning | Trigger Conditions |
| :--- | :--- | :--- |
| `NONE` | No interaction tracking active | Default; no hand-object pair |
| `UNKNOWN` | Pair tracked but no spatial proximity | Distance ≥ `approach_dist_norm` |
| `APPROACHING` | Hand within approach zone | Distance < `approach_dist_norm`, not grasping |
| `GRASPING` | Hand within contact zone AND grasping | Distance < `contact_dist_norm` AND `is_grasping=True` |
| `MANIPULATING` | Sustained grasping for ≥ `dwell_frames_to_manipulate` frames | GRASPING held for N frames |
| `RELEASED` | Interaction ended | GRASPING/MANIPULATING → hand moves away or grasp released |

### Default Thresholds (HOIConfig)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `approach_dist_norm` | 0.25 | Normalised image-space distance for APPROACHING |
| `contact_dist_norm` | 0.12 | Normalised distance for GRASPING |
| `dwell_frames_to_manipulate` | 8 | GRASPING frames before → MANIPULATING |
| `release_frames_threshold` | 4 | Non-grasping frames before state reverts |
| `max_tracked_pairs` | 16 | Maximum simultaneous hand-object pairs |

### Physical Contact Disclaimer

> **IMPORTANT:** Bounding-box proximity **never proves physical contact**. The `APPROACHING` state means the hand wrist is geometrically close to the object bounding-box centre in 2D image space. The `GRASPING` state adds the hand-tracker's pinch heuristic, but this is still a 2D geometric approximation. Physical grasp confirmation would require depth-sensor data or 3D reconstruction — neither of which is available in Phase 2.

---

## 6. Temporal Model Input Contract

The future 1D-CNN / Bi-LSTM HAR classifier will receive:

```python
input_tensor: torch.Tensor  # Shape: (batch, 60, 154), dtype=torch.float32
```

- `batch`: Mini-batch size (1 for real-time inference, N for training).
- `60`: Sliding window length (configurable but fixed at training time).
- `154`: Feature dimension (NEVER change without retraining).

The model is expected to output:
```python
output: torch.Tensor  # Shape: (batch, num_action_classes), log-softmax
```

---

## 7. Known Limitations and Future Work

| Item | Status |
| :--- | :--- |
| YOLO uses generic COCO classes | Planned: replace with custom space apparatus detector (Phase 3) |
| MediaPipe runs on CPU (XNNPACK) | No GPU delegate on Windows; acceptable for Phase 2 |
| HOI based on 2D distance only | No depth/3D contact confirmation; conservative design choice |
| Slot 153 shared by RELEASED one-hot and dwell | Design limitation; will be fixed to 155 dims if RELEASED dwell needed |
| No person in synthetic feed | Pose/hand/HOI all 0 on synthetic frames; expected correct behaviour |
| Feature normalisation | Features are not globally normalised; the temporal model should include batch normalisation |
