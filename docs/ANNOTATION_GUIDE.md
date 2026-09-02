# Laboratory Apparatus & Action Annotation Guide: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

> **Standard Operating Procedure (SOP):** This guide defines the exact annotation rules, boundary selection criteria, label taxonomy, and quality control protocols for human annotators working on the ISRO BAS HAR dataset.

---

## 1. Overview & Tooling Environment

To ensure 100% data sovereignty and enable completely offline workflow execution, all annotations must be generated using local, open-source tools:

| Modality | Recommended Offline Tool | Output Format | Output Directory |
| :--- | :--- | :--- | :--- |
| **Object Detection** | `LabelImg` or `CVAT (Local Docker)` | YOLO format (`.txt`) | `data/labels/train/`, `data/labels/val/` |
| **Temporal Action Intervals** | `Anvil`, `ELAN`, or Custom CLI Tagger | Structured Manifest (`.json`) | `data/annotations/action_manifest.json` |
| **Sequence Validation Ground Truth** | Text Editor / JSON Validator | JSON Trial Manifest (`.json`) | `data/annotations/fsm_eval_manifest.json` |

---

## 2. Object Detection Bounding-Box Annotation Rules (Part 5A)

### 2.1 General Bounding-Box Standards
1. **Tight Fit:** Bounding boxes must enclose all visible pixels of the apparatus without excessive background padding (padding $\le 3$ pixels).
2. **Occlusion Handling:**
   - If an object is **partially occluded** by fingers or wires, draw the bounding box to enclose the entire estimated convex boundary of the object.
   - If an object is **severely occluded ($> 70\%$ hidden)** or completely enclosed inside an astronaut's closed fist, **DO NOT ANNOTATE** that frame.
3. **No Hands:** Do **NOT** annotate human hands. MediaPipe Tasks HandLandmarker automatically detects hands and fingers.
4. **Resolution Normalization:** All coordinate outputs must be normalized $[0.0, 1.0]$:
   ```text
   <class_id> <x_center> <y_center> <width> <height>
   ```

---

### 2.2 Class-Specific Bounding Box Guidelines

```
+---------------------------------------------------------------------------------------------------------+
|                                    7 CLASS BOUNDING BOX ANNOTATION RULES                                |
+----+----------------------+-----------------------------------------------------------------------------+
| ID | Class Name           | Strict Annotation Boundary Rule                                             |
+----+----------------------+-----------------------------------------------------------------------------+
| 0  | `micropipette`       | Enclose from top of push plunger down to the tip mounting nozzle.           |
|    |                      | Include attached disposable tip if present.                                 |
+----+----------------------+-----------------------------------------------------------------------------+
| 1  | `pipette_tip_box`    | Enclose the rectangular outer perimeter of the plastic box.                 |
|    |                      | If the lid is open, include the open lid in the bounding box.               |
+----+----------------------+-----------------------------------------------------------------------------+
| 2  | `sample_vial`        | Enclose the cylindrical body and cap. If cap is detached, annotate body only|
+----+----------------------+-----------------------------------------------------------------------------+
| 3  | `well_plate`         | Enclose the entire outer rectangular flange of the microplate.              |
+----+----------------------+-----------------------------------------------------------------------------+
| 4  | `centrifuge`         | Enclose the outer chassis of the machine including the dome lid.            |
+----+----------------------+-----------------------------------------------------------------------------+
| 5  | `waste_bin`          | Enclose the entire discard beaker / container up to the top lip.            |
+----+----------------------+-----------------------------------------------------------------------------+
| 6  | `chamber_door_latch` | Enclose the mechanical handle/latch mechanism and adjacent door contact rim.|
+----+----------------------+-----------------------------------------------------------------------------+
```

---

## 3. Temporal Action Interval Annotation Rules (Part 5B)

### 3.1 Frame-Accurate Action Boundary Rules

When labeling video clips into temporal intervals, use the following rigorous physical criteria:

```
                  ACTION_IDLE               ACTION_ASPIRATE               ACTION_IDLE
Timeline:  [───────────────────────────][═════════════════════════][───────────────────────────]
                                        ▲                         ▲
                                    Start Frame                End Frame
                               (First hand contact       (Plunger released, tip
                               with vial/pipette)        withdrawn from vial)
```

| Action Label | Exact Start Frame ($t_{start}$) | Exact End Frame ($t_{end}$) |
| :--- | :--- | :--- |
| **`ACTION_IDLE`** | Operator hands resting or stationary; no active reaching toward objects. | Operator begins purposeful reaching motion toward an object/container. |
| **`ACTION_SANITIZE`** | First contact with sanitizer or start of hand/container sanitizing rubbing. | Hands complete sanitizing motion and disengage. |
| **`ACTION_HOLD_BOTTLE`** | Fingers close around bottle and lift/hold it clearly in view. | Bottle hold concludes and placement motion begins. |
| **`ACTION_PLACE_BOTTLE`**| Hand begins placing bottle downward toward table surface. | Bottle rests on table and fingers completely release grip. |
| **`ACTION_HOLD_BOX`** | Fingers grasp and lift small box into clear view. | Box hold concludes and opening motion begins. |
| **`ACTION_OPEN_BOX`** | Fingers begin lifting/opening the box lid for inspection. | Box lid reaches open position and inspection concludes. |
| **`ACTION_PICK_OBJECT`** | Fingers touch and grasp the small object inside the box. | Object is lifted cleanly out of the box. |
| **`ACTION_TRANSFER_OBJECT`** | Hand moves object toward the target container/zone. | Object is placed down in target location and grip releases. |
| **`ACTION_RETURN_OBJECT`** | Fingers grasp object from target location and move toward box. | Object is placed back inside box and grip releases. |
| **`ACTION_CLOSE_BOX`** | Hand touches box lid to close it and return box to resting position. | Box is shut and rests back in original position on table. |

---

### 3.2 Transition Periods & Ambiguity Handling
- **Transition Gaps:** If there is a brief transition between two steps (e.g. 0.5 seconds of moving hands from tip box to sample vial), label the transition segment as **`ACTION_IDLE`** if $\ge 0.5\text{ s}$ (15 frames). If $< 0.5\text{ s}$, boundary frames are snapped to the onset of the incoming action.
- **Bimanual Conflict:** When one hand is holding an object (e.g. holding vial) while the other performs an action (e.g. aspirating), the primary manipulative action (**`ACTION_ASPIRATE`**) takes precedence.

---

## 4. Quality Control & Inter-Annotator Verification Protocol

To maintain high data fidelity across multiple student annotators, follow this QA protocol:

### 4.1 Verification Metrics
1. **Object Detection Bounding Box IoU:** Cross-annotator Intersection-over-Union on a shared 50-image calibration subset must achieve:
   $$\text{Mean IoU} \ge \mathbf{0.85}$$
2. **Temporal Action Boundary Alignment:** Start and end timestamps for the same action clip must agree within:
   $$|\Delta t_{start}| \le \mathbf{0.20\text{ s}}\text{ (6 frames)}, \quad |\Delta t_{end}| \le \mathbf{0.20\text{ s}}$$
3. **Cohen's Kappa ($\kappa$):** Categorical label agreement on temporal segments must exceed:
   $$\kappa \ge \mathbf{0.90}$$

### 4.2 Automated Sanity Checks (`scripts/validate_annotations.py`)
Run an automated Python validation script before ingesting any batch:
- Asserts all class IDs are integers between `0` and `6` (for objects) or `0` and `9` (for actions).
- Asserts all normalized coordinates $x, y, w, h \in [0.0, 1.0]$.
- Asserts $w > 0$ and $h > 0$.
- Asserts temporal segment start frames $< $ end frames.
- Asserts no overlapping segments with conflicting action labels.

---

## 5. Directory Structure for Labeled Assets

```
data/
├── custom_objects/
│   ├── images/
│   │   ├── train/          # ~560 frames
│   │   ├── val/            # ~120 frames
│   │   └── test/           # ~120 frames
│   └── labels/
│       ├── train/          # Matching .txt YOLO labels
│       ├── val/
│       └── test/
│
└── action_sequences/
    ├── raw_videos/         # 60 MP4 recordings (Actors 1–6)
    ├── annotations/
    │   ├── session_01_labels.json
    │   ├── session_02_labels.json
    │   └── action_manifest.json (compiled database)
    └── features/           # Extracted 154-d JSONL streams
        ├── session_01.jsonl
        └── session_02.jsonl
```
