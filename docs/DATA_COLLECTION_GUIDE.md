# Step-by-Step Data Collection & First Recording Guide: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments (Home Demo Protocol)

> **Field Manual for Home-Based Recording & Annotation:** This guide details the complete end-to-end workflow to set up, record, annotate, and validate experiment sessions using our offline toolchain with the practical 10-step home demo protocol.

---

## 1. Physical Environment & Setup

### 1.1 Home Tabletop Workspace Layout
```
                           [Primary Camera (Front / 45° View, 720p/1080p)]
                                           │
                                           ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                              TABLE SURFACE                              │
   │                                                                         │
   │   [Hand Sanitizer]       [Bottle / Container]        [Small Box]        │
   │                                                    (with small object)  │
   │                                                                         │
   │                      [Target Destination / Zone]                        │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘
                                ▲
                                │
                        [Operator / Actor]
```

### 1.2 Setup & Apparatus Items
1. **Hand Sanitizer bottle / pump**
2. **Bottle / Container**
3. **Small Box** (with lid that opens/closes easily)
4. **Small Harmless Object** (placed inside the box, e.g., marker, vial, token, USB drive)
5. **Target Zone / Destination Tray** (on the table)

### 1.3 Camera Configuration
- **Resolution:** 720p ($1280 \times 720$) or 1080p ($1920 \times 1080$).
- **Framerate:** 30 FPS constant.
- **Position:** 45° overhead / front-quarter view, 1.0–1.3 m distance.
- **Framing:** Operator's upper torso, arms, hands, and all objects clearly visible in frame.
- **Lighting:** Even ambient room lighting; avoid heavy shadows or glare.

---

## 2. How to Record Your First Real Demo Video (`SES_001_NOMINAL`)

### Step 1: Launch the Session Recorder
Run the offline session recording utility from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.record_session `
    --source 0 `
    --actor ACTOR_01 `
    --session SES_001_NOMINAL `
    --lighting DIFFUSE_LED `
    --handedness RIGHT `
    --camera CAM_FRONT_45DEG `
    --protocol HOME_DEMO_PROTOCOL_01 `
    --notes "First nominal home demo recording of 10-step protocol."
```

### Step 2: Interactive Recording Workflow
1. The OpenCV live preview window will open with telemetry and a grey `PAUSED` indicator.
2. Place the items in position: Hand sanitizer, Bottle, Box (with object inside), and destination zone.
3. Sit or stand in front of the table, assume a resting stance with hands stationary on the table or lap.
4. Press **`SPACE`** or **`R`**. The red `REC` badge will begin flashing.
5. Perform the 10 physical actions in sequence:

| Step | Action Name | Physical Movement |
| :--- | :--- | :--- |
| **1** | **`IDLE`** | Remain still in the starting position for ~3 seconds. |
| **2** | **`SANITIZE`** | Pick up hand sanitizer and sanitize hands / bottle. |
| **3** | **`HOLD_BOTTLE`** | Pick up the bottle and hold it clearly in view for ~3 seconds. |
| **4** | **`PLACE_BOTTLE`**| Place the bottle back down on the table. |
| **5** | **`HOLD_BOX`** | Pick up the small box and hold it clearly. |
| **6** | **`OPEN_BOX`** | Open the box lid and briefly inspect the contents inside. |
| **7** | **`PICK_OBJECT`** | Pick up the small object from inside the box. |
| **8** | **`TRANSFER_OBJECT`** | Move the object and place it into the designated target location. |
| **9** | **`RETURN_OBJECT`** | Pick up the object and return it back into the box. |
| **10**| **`CLOSE_BOX`** | Close the box lid and place the box back in its original position. |

6. Once completed, pause hands for 2 seconds, then press **`SPACE`** to pause recording.
7. Press **`Q`** or **`ESC`** to finalize and save the MP4 video and JSON metadata.

### Step 3: Verified Artifacts
- Video file: `data/raw/videos/SES_001_NOMINAL.mp4`
- Metadata file: `data/raw/metadata/SES_001_NOMINAL.json`

---

## 3. How to Annotate Your Video for Temporal HAR

### Step 1: Launch the Action Annotator
```powershell
.\.venv\Scripts\python.exe -m scripts.annotate_temporal `
    --video data/raw/videos/SES_001_NOMINAL.mp4 `
    --actor ACTOR_01
```

### Step 2: Mark Action Intervals
1. Use **`SPACE`** to play/pause. Use **`A`** / **`D`** to step backward/forward single frames.
2. Scrub to the frame where Step 1 starts and press **`[`** (set Start).
3. Scrub to where Step 1 ends and press **`]`** (set End).
4. Press the corresponding number key to select the action:
   - **`0`**: `ACTION_IDLE`
   - **`1`**: `ACTION_SANITIZE`
   - **`2`**: `ACTION_HOLD_BOTTLE`
   - **`3`**: `ACTION_PLACE_BOTTLE`
   - **`4`**: `ACTION_HOLD_BOX`
   - **`5`**: `ACTION_OPEN_BOX`
   - **`6`**: `ACTION_PICK_OBJECT`
   - **`7`**: `ACTION_TRANSFER_OBJECT`
   - **`8`**: `ACTION_RETURN_OBJECT`
   - **`9`**: `ACTION_CLOSE_BOX`
5. Press **`ENTER`** or **`C`** to commit the interval.
6. Repeat for all 10 steps. Press **`W`** to save, and **`Q`** to exit.
7. Output manifest is saved to `data/temporal_har/manifests/SES_001_NOMINAL_actions.json`.

---

## 4. How to Extract & Annotate Object Frames for YOLO

### Step 1: Extract Non-Duplicate Frames
```powershell
.\.venv\Scripts\python.exe -m scripts.extract_frames `
    --video data/raw/videos/SES_001_NOMINAL.mp4 `
    --interval 15 `
    --diff_thresh 8.0
```
Extracted frames and contact sheets appear in:
`data/object_detection/images/staging/SES_001_NOMINAL/`

### Step 2: Stage Frames into Split
```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_object_dataset `
    --staging_dir data/object_detection/images/staging/SES_001_NOMINAL `
    --split train
```
Images and `.txt` label templates are placed in `data/object_detection/images/train/` and `data/object_detection/labels/train/`.

---

## 5. How to Validate Dataset Integrity

Run the master offline validator:
```powershell
.\.venv\Scripts\python.exe -m scripts.validate_dataset
```
Audits:
- Image/label pairing and bounding box normalized coordinates
- Temporal HAR interval continuity and valid action IDs (0..9)
- Sequence evaluation trial schemas
- Anti-leakage checks across train/val/test splits

---

## 6. How to Extract 60-Frame Feature Tensors

```powershell
.\.venv\Scripts\python.exe -m scripts.extract_har_features `
    --manifest data/temporal_har/manifests/SES_001_NOMINAL_actions.json
```
Creates `data/temporal_har/features/SES_001_NOMINAL_windows.npz` containing $(N \times 60 \times 154)$ float32 sliding-window tensors.

---

## 7. How to View Dataset Statistics

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_statistics
```
Displays an offline inventory report of video hours, action segment distributions, and sliding window counts.
