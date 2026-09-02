# Dataset & Annotation Tooling Reference: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

> **Software Architecture Manual:** Comprehensive CLI reference and API documentation for all local, offline dataset recording, frame extraction, annotation, validation, and feature preparation tools.

---

## 1. Tooling Architecture Overview

```
[Camera / Video Source]
        │
        ├──► scripts.record_session
        │      ├── Output: data/raw/videos/<session_id>.mp4
        │      └── Output: data/raw/metadata/<session_id>.json
        │
        ├──► scripts.extract_frames
        │      └── Output: data/object_detection/images/staging/<session_id>/
        │
        ├──► scripts.prepare_object_dataset
        │      └── Output: data/object_detection/images/{train,val,test}/
        │                  data/object_detection/labels/{train,val,test}/
        │
        ├──► scripts.annotate_temporal
        │      └── Output: data/temporal_har/manifests/<session_id>_actions.json
        │
        ├──► scripts.annotate_sequence_eval
        │      └── Output: data/sequence_validation/manifests/<trial_id>.json
        │
        ├──► scripts.extract_har_features
        │      └── Output: data/temporal_har/features/<session_id>_windows.npz (N, 60, 154)
        │
        ├──► scripts.validate_dataset (Integrity & Anti-Leakage Audit)
        │
        └──► scripts.dataset_statistics (Inventory & Distribution Report)
```

---

## 2. CLI Tool Reference

### 2.1 `scripts.record_session`
Records raw video and writes structured JSON session metadata.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | `"0"` | Camera device index or `"synthetic"` |
| `--actor` | `str` | `"ACTOR_01"` | Operator identifier |
| `--session` | `str` | Auto | Session identifier string |
| `--camera` | `str` | `"CAM_FRONT_45DEG"` | Camera angle identifier |
| `--lighting` | `str` | `"DIFFUSE_LED"` | Lighting condition category |
| `--handedness` | `str` | `"RIGHT"` | `"RIGHT"`, `"LEFT"`, or `"BIMANUAL"` |
| `--protocol` | `str` | `"HOME_DEMO_PROTOCOL_01"`| Protocol ID |
| `--width` | `int` | `1280` | Frame pixel width |
| `--height` | `int` | `720` | Frame pixel height |
| `--fps` | `int` | `30` | Target acquisition framerate |
| `--no_gui` | `flag` | `False` | Run in headless mode |
| `--auto_record_frames`| `int` | `0` | Auto-terminate after N recorded frames |

---

### 2.2 `scripts.extract_frames`
Intelligently extracts non-redundant frames from MP4 video for object labeling.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | `Path` | Required | Path to input MP4 video file |
| `--output_dir` | `Path` | Auto | Staging destination directory |
| `--interval` | `int` | `15` | Frame stride sampling step |
| `--diff_thresh` | `float` | `8.0` | Minimum mean pixel difference vs. last saved frame |
| `--max_frames` | `int` | `0` | Maximum frames to extract (0 = unlimited) |
| `--no_contact_sheet` | `flag` | `False` | Disable generation of thumbnail contact sheet |

---

### 2.3 `scripts.prepare_object_dataset`
Stages frames into subject-separated train/val/test splits and initializes `.txt` label templates.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--staging_dir` | `Path` | `""` | Source folder of extracted frames |
| `--split` | `str` | `"train"` | Target dataset partition (`train`, `val`, `test`) |
| `--distribute` | `flag` | `False` | Run automatic anti-leakage subject-wise distribution |
| `--train_actors` | `list` | `ACTOR_01..04` | Actors allocated to Train set |
| `--val_actors` | `list` | `ACTOR_05` | Actors allocated to Validation set |
| `--test_actors` | `list` | `ACTOR_06` | Actors allocated to Test set |

---

### 2.4 `scripts.annotate_temporal`
Interactive OpenCV video scrubber for frame-accurate action interval labeling.

| Key / Control | Function |
| :--- | :--- |
| `SPACE` | Play / Pause playback |
| `A` / `D` (or Left / Right) | Step backward / forward 1 frame |
| `S` / `F` (or Down / Up) | Jump backward / forward 30 frames (1 second) |
| `[` or `I` | Mark action interval start frame ($t_{start}$) |
| `]` or `O` | Mark action interval end frame ($t_{end}$) |
| `0` to `9` | Select Action Class (`0: IDLE`, `1: SANITIZE`, `2: HOLD_BOTTLE`, `3: PLACE_BOTTLE`, `4: HOLD_BOX`, `5: OPEN_BOX`, `6: PICK_OBJECT`, `7: TRANSFER_OBJECT`, `8: RETURN_OBJECT`, `9: CLOSE_BOX`) |
| `ENTER` or `C` | Commit marked interval to manifest |
| `BACKSPACE` or `X` | Remove last committed interval |
| `W` | Save manifest JSON to disk |
| `Q` or `ESC` | Save and quit |

---

### 2.5 `scripts.annotate_sequence_eval`
Creates ground-truth test manifests for the sequence state machine validator.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--trial` | `str` | Required | Unique evaluation trial ID (e.g. `EVAL_ERR_01`) |
| `--actor` | `str` | `"ACTOR_01"` | Operator identifier |
| `--type` | `str` | `"CORRECT_SEQUENCE"` | Anomaly category (`SKIPPED_STEP`, `OUT_OF_ORDER`, etc.) |
| `--faulty_step` | `int` | `0` | Step ID where anomaly occurs |
| `--skip` | `int list` | `[]` | Injected skipped step ID(s) |
| `--repeat` | `int list` | `[]` | Injected repeated step ID(s) |
| `--stall_step` | `int` | `0` | Injected stalled step ID |

---

### 2.6 `scripts.validate_dataset`
Audits the complete local dataset repository for structural defects and data leakage.

| Audit Check | Failure Condition | Impact |
| :--- | :--- | :--- |
| **Orphaned Images** | Image exists without matching `.txt` label file | Incomplete YOLO dataset |
| **Illegal Class IDs** | Label contains class ID $\notin [0, 6]$ | YOLO training crash |
| **Invalid Coordinates**| $cx, cy, w, h \notin [0.0, 1.0]$ or $w \le 0$ | Bounding box corruption |
| **Temporal Inversion** | Segment $t_{start} \ge t_{end}$ | HAR training sequence corruption |
| **Subject Leakage** | Same `actor_id` in Train and Val/Test splits | Over-optimistic evaluation metrics |

---

### 2.7 `scripts.extract_har_features`
Processes annotated videos through the Perception + HOI pipeline to generate $(N \times 60 \times 154)$ tensors.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--manifest` | `Path` | `""` | Path to single action manifest JSON |
| `--all` | `flag` | `False` | Process all manifests in `data/temporal_har/manifests/` |
| `--stride` | `int` | `10` | Sliding window temporal step in frames |

---

### 2.8 `scripts.dataset_statistics`
Computes and prints a formatted terminal summary of the entire dataset inventory.
