# Dataset Strategy & Collection Plan: ISRO PS 26174
## AI Human Activity Recognition for On-board BAS Experiments

> **Protocol Notice:** All experiment scenarios and action classes in this plan represent a **Synthetic Demonstration Protocol** developed for software architecture validation, state machine verification, and prototype benchmarking. The system is architected to ingest any official ISRO experiment protocol in JSON format.

---

## 1. Executive Summary & Design Scope

To train and evaluate a robust, 100% offline edge-AI perception and temporal HAR pipeline targeting an **NVIDIA GTX 1650 (4 GB VRAM)** and an Intel Core i5 processor, we require two distinct datasets:

1. **Custom Space Apparatus Object Detection Dataset:** Static frames for fine-tuning YOLOv8n to detect physical experiment tools and containers.
2. **Temporal Human Activity Recognition (HAR) Video Dataset:** Continuous multi-view video recordings of operators performing 10-step experiment sequences (nominal, erroneous, and background activities).

This document specifies the target classes, collection protocols, split strategies, and negative testing requirements.

---

## 2. Part 1 — Custom Object Detection Dataset

### 2.1 Critical Review of Candidate Object Classes

We evaluated all apparatus items mentioned in `config/experiment_protocols.json` and initial design notes against four technical criteria:
- **Object Localization Utility:** Can a discrete bounding box be drawn around it?
- **Visual Distinguishability:** Can YOLO reliably differentiate it from background and other tools?
- **Downstream HOI Value:** Does detecting this object directly inform the 154-dimensional feature vector and action classifier?
- **Redundancy:** Is this already tracked by MediaPipe (e.g., hands)?

#### Discarded / Consolidated Items
- ❌ **`astronaut_hand` (Discarded):** Redundant. MediaPipe Tasks HandLandmarker already extracts 21 3D landmarks, wrist coordinates, and pinch heuristics. Detecting hands in YOLO introduces redundant bounding boxes and occludes apparatus labels.
- ❌ **`glovebox_surface` (Discarded):** Background surface spanning the entire frame; not a localized object.
- ❌ **`latch` / `lock_switch` / `chamber_latch` (Consolidated):** Small mechanical switches suffer from extreme occlusion by fingers. Consolidated into `chamber_door_latch` (inclusive of surrounding door bezel).
- ❌ **`pipette_stand` / `storage_rack` (Consolidated):** Both are stationary support racks; consolidated into `apparatus_rack`.

### 2.2 Final Recommended 7 Object Classes

The custom object detector will be trained exclusively on **7 visually distinct apparatus classes**:

```
+---------------------------------------------------------------------------------------------------------+
|                                    FINAL 7 APPARATUS OBJECT CLASSES                                     |
+----+----------------------+-----------------------------------------------------------------------------+
| ID | Class Name           | Primary Downstream Action Association                                       |
+----+----------------------+-----------------------------------------------------------------------------+
| 0  | `micropipette`       | ACTION_PICK_PIPETTE, ACTION_MOUNT_TIP, ACTION_ASPIRATE, ACTION_DISPENSE     |
| 1  | `pipette_tip_box`    | ACTION_MOUNT_TIP                                                            |
| 2  | `sample_vial`        | ACTION_RETRIEVE_VIAL, ACTION_ASPIRATE, ACTION_LOAD_CENTRIFUGE               |
| 3  | `well_plate`         | ACTION_DISPENSE                                                             |
| 4  | `centrifuge`         | ACTION_LOAD_CENTRIFUGE, ACTION_LOCK_DEVICE                                  |
| 5  | `waste_bin`          | ACTION_EJECT_TIP                                                            |
| 6  | `chamber_door_latch` | ACTION_OPEN_DOOR, ACTION_LOCK_DEVICE                                        |
+----+----------------------+-----------------------------------------------------------------------------+
```

---

### 2.3 Detailed Class Specifications

#### Class 0: `micropipette`
- **Purpose:** Identifies precision liquid handling tool.
- **Visual Distinguishability:** High (distinctive slender cylindrical barrel, top push-button plunger, volume display window).
- **Appearance Variations:** Standard volume sizes (P20, P200, P1000), colors (blue, red, yellow plunger caps), angled/vertical orientations.
- **Occlusion Concerns:** High occlusion during handling (astronaut fingers wrap around the body). Plunger top and tip nozzle usually remain visible.
- **Hand Interaction Rule:** Annotate the full visible boundary of the pipette even when partially occluded by grasping fingers.
- **Recommended Samples:** Minimum **500 annotated instances** (across diverse angles and in-hand grasps).

#### Class 1: `pipette_tip_box`
- **Purpose:** Identifies sterile tip reservoir for tip mounting verification.
- **Visual Distinguishability:** High (rectangular plastic enclosure with top lid, internal 8×12 grid matrix of cone tips).
- **Appearance Variations:** Lid open vs. lid closed, tip matrix full vs. partially depleted, colored racks (blue/green/yellow).
- **Occlusion Concerns:** Low to moderate (pipette enters from above during tip mounting).
- **Hand Interaction Rule:** Annotate full box perimeter including open hinged lid.
- **Recommended Samples:** Minimum **350 annotated instances**.

#### Class 2: `sample_vial`
- **Purpose:** Tracks biological specimen container (1.5 mL / 2.0 mL microcentrifuge tube / cryovial).
- **Visual Distinguishability:** Moderate (small cylindrical plastic tube with conical bottom and snap/screw cap).
- **Appearance Variations:** Transparent vs. amber/opaque plastic, snap-cap open vs. closed, fluid filled vs. empty, barcode/label markings.
- **Occlusion Concerns:** Very High. When grasped by thumb and index finger, only the cap or bottom may be visible.
- **Hand Interaction Rule:** Annotate the entire vial boundary if at least 30% is visible. If completely concealed inside the palm, do not annotate.
- **Recommended Samples:** Minimum **600 annotated instances** (critical for HOI distance calculations).

#### Class 3: `well_plate`
- **Purpose:** Identifies target assay plate (24-well / 96-well standard microplate).
- **Visual Distinguishability:** High (flat rectangular clear plastic grid with circular/square wells).
- **Appearance Variations:** 24 vs. 96 wells, clear vs. white/black plastic, flat on workbench vs. angled.
- **Occlusion Concerns:** Low (pipette hovers above wells during dispensing).
- **Hand Interaction Rule:** Annotate entire outer rectangular rim of plate.
- **Recommended Samples:** Minimum **300 annotated instances**.

#### Class 4: `centrifuge`
- **Purpose:** Identifies benchtop microcentrifuge machine.
- **Visual Distinguishability:** Very High (large circular/cubical desktop unit with transparent dome lid and rotor cavity).
- **Appearance Variations:** Lid open vs. lid closed, spinning rotor vs. stopped rotor, digital control panel on/off.
- **Occlusion Concerns:** Low (large static apparatus).
- **Hand Interaction Rule:** Annotate the entire machine chassis.
- **Recommended Samples:** Minimum **300 annotated instances**.

#### Class 5: `waste_bin`
- **Purpose:** Identifies contaminated tip / biohazard discard receptacle.
- **Visual Distinguishability:** High (upright beaker, plastic discard flask, or biohazard box with open mouth).
- **Appearance Variations:** Cylindrical beaker vs. rectangular box, red/yellow biohazard bag liner, empty vs. partially filled with tips.
- **Occlusion Concerns:** Low to moderate.
- **Hand Interaction Rule:** Annotate entire container.
- **Recommended Samples:** Minimum **250 annotated instances**.

#### Class 6: `chamber_door_latch`
- **Purpose:** Identifies incubator/glovebox access door and locking handle.
- **Visual Distinguishability:** Moderate (door perimeter with mechanical handle/latch lever).
- **Appearance Variations:** Door closed and latched, door unlatched, door swung fully open (45°–90°).
- **Occlusion Concerns:** High during opening/closing hand contact.
- **Hand Interaction Rule:** Annotate the latch and door handle zone as a single unified bounding box.
- **Recommended Samples:** Minimum **300 annotated instances**.

---

### 2.4 Object Dataset Volume & Split Recommendation

| Split | Percentage | Number of Distinct Images | Annotations Target | Primary Criteria |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 70% | ~560 images | ~2,100 boxes | Diverse lighting, all 7 classes, varied backgrounds |
| **Validation** | 15% | ~120 images | ~450 boxes | Different camera angle from training set |
| **Test** | 15% | ~120 images | ~450 boxes | Independent subjects, novel apparatus placements |
| **TOTAL** | **100%** | **~800 images** | **~3,000 boxes** | Fully balanced across all 7 classes |

---

## 3. Part 2 — Temporal HAR Dataset

### 3.1 10 Target Action Classes (Home Demo Protocol)

The temporal HAR model classifies sliding windows into **10 fine-grained activity classes**:

```
+---------------------------------------------------------------------------------------------------------+
|                                  10 TEMPORAL HAR ACTIVITY CLASSES                                       |
+----+-----------------------------+----------------------------------------------------------------------+
| ID | Action Label                | Core Physical Interaction                                            |
+----+-----------------------------+----------------------------------------------------------------------+
| 0  | `ACTION_IDLE`               | Baseline / resting in starting position for a few seconds            |
| 1  | `ACTION_SANITIZE`           | Picking up sanitizer and sanitizing hands / bottle / container       |
| 2  | `ACTION_HOLD_BOTTLE`        | Picking up a bottle and holding it clearly for a few seconds         |
| 3  | `ACTION_PLACE_BOTTLE`       | Placing the bottle back on the table                                 |
| 4  | `ACTION_HOLD_BOX`           | Picking up a small box and holding it clearly                        |
| 5  | `ACTION_OPEN_BOX`           | Opening the box and briefly inspecting it                            |
| 6  | `ACTION_PICK_OBJECT`        | Picking up a small harmless object from inside the box               |
| 7  | `ACTION_TRANSFER_OBJECT`    | Moving the object from the box to another clearly defined container  |
| 8  | `ACTION_RETURN_OBJECT`      | Picking up the object and returning it to the box                    |
| 9  | `ACTION_CLOSE_BOX`          | Closing the box and placing it back in its original position         |
+----+-----------------------------+----------------------------------------------------------------------+
```

---

### 3.2 Fine-Grained Action Class Specifications

#### 0. `ACTION_IDLE`
- **Definition:** Astronaut is not actively performing any protocol step (waiting, reading instructions, resting hands, adjusting posture).
- **Start / End:** Hand velocity drops below active threshold, or no interaction for > 1.0 s.
- **Visual Cues:** Hands stationary on workbench, arms crossed, or hands hovering without purposeful motion.
- **Key Features:** Low hand velocity, HOI state `NONE` or `UNKNOWN`, large hand-object distances ($d > 0.35$).
- **Minimum Target Clips:** 80 clips (1.0–5.0 s duration each).
- **Negative Examples:** Pauses between pipetting strokes; holding a vial stationary while inspecting it.

#### 1. `ACTION_SANITIZE`
- **Definition:** Wiping glovebox workspace surfaces with a wipe or rubbing gloved hands together.
- **Start Condition:** Hands contact surface/wipe and initiate rhythmic oscillatory wiping motions.
- **End Condition:** Hands cease wiping motion and discard wipe or lift hands from surface.
- **Visual Cues:** Rhythmic 2D translational wrist motion across horizontal plane; bimanual hand rubbing.
- **Key Features:** High wrist kinetic variance, periodic wrist $(x, y)$ trajectory, torso lean forward ($10^\circ–25^\circ$).
- **Minimum Target Clips:** 50 recordings (3.0–8.0 s duration).
- **Difficult Cases:** Distinguishing wiping from simply searching or moving items across the desk. (Cue: wiping has periodic velocity profiles).

#### 2. `ACTION_OPEN_DOOR`
- **Definition:** Disengaging chamber latch and swinging open the access door.
- **Start Condition:** Hand reaches toward `chamber_door_latch` and contacts handle.
- **End Condition:** Door completes outward arc and hand releases latch.
- **Visual Cues:** Hand reaches forward-high, grasps latch, pulls backward/sideways in an arc.
- **Key Features:** Wrist $y$ elevated, HOI pair `(Hand, chamber_door_latch)` enters `MANIPULATING`, door bbox expands.
- **Minimum Target Clips:** 45 recordings (1.5–4.0 s duration).

#### 3. `ACTION_RETRIEVE_VIAL`
- **Definition:** Selecting and lifting a sample vial from the storage rack.
- **Start Condition:** Hand reaches toward `sample_vial` in rack.
- **End Condition:** Vial is lifted free of rack and held in active manipulation zone.
- **Visual Cues:** Precision pincer grasp (thumb + index pinch), upward vertical hand translation.
- **Key Features:** HOI `(Hand, sample_vial)` enters `GRASPING` then `MANIPULATING`, wrist $y$ translates upward ($-\Delta y$), vial bbox moves.
- **Minimum Target Clips:** 55 recordings (1.5–4.0 s duration).
- **Ambiguity:** Reaching into rack without picking vs. actual retrieval. (Cue: vial bounding box translates with wrist).

#### 4. `ACTION_PICK_PIPETTE`
- **Definition:** Reaching to the tool stand and lifting the micropipette.
- **Start Condition:** Hand reaches toward `micropipette` on stand.
- **End Condition:** Pipette clears the stand mount and assumes vertical handling posture.
- **Visual Cues:** Power grasp (palm wraps around upper barrel), upward lift.
- **Key Features:** HOI `(Hand, micropipette)` enters `MANIPULATING`, pipette bbox center translates.
- **Minimum Target Clips:** 50 recordings (1.5–3.5 s duration).

#### 5. `ACTION_MOUNT_TIP`
- **Definition:** Aligning pipette nozzle with an open tip box and pressing downward to seat a sterile tip.
- **Start Condition:** Pipette nozzle enters the vertical column above `pipette_tip_box`.
- **End Condition:** Pipette is pressed downward with brief pause, then lifted upward with tip attached.
- **Visual Cues:** Vertical downward thrust of pipette into box, followed by upward lift.
- **Key Features:** Distance `(micropipette, pipette_tip_box) < 0.10`, sharp downward-then-upward wrist acceleration spike.
- **Minimum Target Clips:** 55 recordings (2.0–5.0 s duration).
- **Critical Disambiguation:** *Mount Tip vs. Eject Tip:* Mount Tip occurs over `pipette_tip_box` (downward press); Eject Tip occurs over `waste_bin` (side plunger push).

#### 6. `ACTION_ASPIRATE`
- **Definition:** Lowering pipette tip into sample vial and releasing top plunger to draw liquid.
- **Start Condition:** Pipette tip enters neck of `sample_vial`.
- **End Condition:** Liquid drawn, tip withdrawn from vial.
- **Visual Cues:** Bimanual coordination (one hand holds vial, other holds pipette vertically above vial); thumb depresses and slowly releases plunger.
- **Key Features:** Both wrists in close proximity to `sample_vial` ($d < 0.12$), HOI `(Pipette, sample_vial)` proximity minimal, sustained dwell $\ge 2.0$ s.
- **Minimum Target Clips:** 60 recordings (2.5–6.0 s duration).
- **Critical Disambiguation:** *Aspirate vs. Dispense:* Aspirate co-occurs with `sample_vial`; Dispense co-occurs with `well_plate`.

#### 7. `ACTION_DISPENSE`
- **Definition:** Positioning pipette over target well plate and fully depressing plunger.
- **Start Condition:** Pipette tip positioned over a specific well of `well_plate`.
- **End Condition:** Plunger fully depressed, tip lifted away from well plate.
- **Visual Cues:** Pipette vertical above flat rectangular `well_plate`, thumb depresses plunger, slight dwell.
- **Key Features:** Distance `(Hand/Pipette, well_plate) < 0.12`, `well_plate` confidence high, thumb landmark flexes downward.
- **Minimum Target Clips:** 60 recordings (2.5–6.0 s duration).

#### 8. `ACTION_EJECT_TIP`
- **Definition:** Positioning pipette over waste receptacle and pressing tip ejector button.
- **Start Condition:** Pipette positioned over opening of `waste_bin`.
- **End Condition:** Tip detached and falls into bin; pipette lifted away.
- **Visual Cues:** Pipette held over `waste_bin`, thumb/index pushes side ejector lever, tip separates.
- **Key Features:** Distance `(Hand/Pipette, waste_bin) < 0.15`, rapid downward detachment of tip artifact.
- **Minimum Target Clips:** 50 recordings (1.5–3.5 s duration).

#### 9. `ACTION_LOAD_CENTRIFUGE`
- **Definition:** Inserting capped sample vial into centrifuge rotor slot.
- **Start Condition:** Hand carrying `sample_vial` reaches into open `centrifuge` chamber.
- **End Condition:** Vial released inside rotor bucket; hand withdraws.
- **Visual Cues:** Hand carrying vial reaches into circular centrifuge opening, releases grip, withdraws empty hand.
- **Key Features:** Distance `(Hand/vial, centrifuge) < 0.10`, HOI `(Hand, sample_vial)` transitions from `MANIPULATING` to `RELEASED`.
- **Minimum Target Clips:** 50 recordings (2.0–5.0 s duration).

#### 10. `ACTION_LOCK_DEVICE`
- **Definition:** Closing centrifuge lid and engaging physical lock switch / safety latch.
- **Start Condition:** Hand reaches to open lid and pushes downward.
- **End Condition:** Lid clicks shut and lock lever is turned/latched.
- **Visual Cues:** Downward rotation of lid, hand presses firmly on latch mechanism.
- **Key Features:** Centrifuge bbox transitions to closed state, hand contacts `chamber_door_latch` / lid handle.
- **Minimum Target Clips:** 45 recordings (1.5–4.0 s duration).

---

### 3.3 Disambiguation Matrix for Visually Similar Actions

| Confusable Action Pair | Distinguishing Spatial Feature | Distinguishing Object Feature | Distinguishing HOI / Kinematic Feature |
| :--- | :--- | :--- | :--- |
| **`MOUNT_TIP` vs `EJECT_TIP`** | Hand location relative to workbench | Target object is `pipette_tip_box` (Mount) vs `waste_bin` (Eject) | Mount has downward press velocity; Eject has side button push |
| **`ASPIRATE` vs `DISPENSE`** | Co-occurring container type | Target object is `sample_vial` (Aspirate) vs `well_plate` (Dispense) | Aspirate is typically bimanual (holding vial); Dispense is single-handed hovering over plate |
| **`RETRIEVE_VIAL` vs `LOAD_CENTRIFUGE`** | Source/target apparatus | Target is `apparatus_rack` (Retrieve) vs `centrifuge` rotor (Load) | Retrieve: hand moves *out* of rack; Load: hand moves *into* centrifuge cavity |
| **`PICK_PIPETTE` vs `MOUNT_TIP`** | Vertical elevation | Tool rack / stand vs Tip box | Pick moves pipette away from stand; Mount moves pipette into tip box |

---

## 4. Part 3 — Sequence Validation & Anomaly Dataset

To test the **Protocol State Machine** (`ProtocolStateMachine`) independently of the HAR classifier, we design a dedicated **Sequence Validation Benchmark Suite** consisting of full-length end-to-end experiment trials.

### 4.1 Test Scenarios & Anomaly Categories

```
+---------------------------------------------------------------------------------------------------------+
|                                  SEQUENCE VALIDATION TEST SCENARIOS                                     |
+----+----------------------------+-----------------------------------+-----------------------------------+
| ID | Scenario Category          | Action Execution Pattern          | Expected State Machine Behavior   |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S1 | Nominal Complete Run       | Steps 1 → 2 → 3 → 4 → 5 → 6 →     | All 10 steps validated.           |
|    |                            | 7 → 8 → 9 → 10 (perfect order)    | Status: VALID_COMPLETED           |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S2 | Skipped Critical Step      | Steps 1 → 2 → 3 → 4 → [SKIP 5] →  | Flags: SKIPPED_STEP (Step 5)      |
|    | (No tip mounted)           | 6 → 7 ...                         | Alert: "Mount sterile tip first"  |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S3 | Out-of-Order Execution     | Steps 1 → 2 → 3 → 4 → 5 → 7       | Flags: OUT_OF_SEQUENCE            |
|    | (Dispense before Aspirate) | (Dispense before Aspirating #6)   | Alert: "Aspirate reagent first"   |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S4 | Redundant Step Repeat      | Steps 1 → 2 → 3 → 4 → 5 → 5 →     | FSM ignores duplicate #5;         |
|    | (Re-mounting tip)          | 6 → 7 ...                         | Maintains step 5 as completed     |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S5 | Stalled / Timeout Trial    | Step 6 initiated, but operator    | Flags: TIMEOUT_EXCEEDED           |
|    | (Inactivity > max_duration)| pauses stationary for > 45 sec    | Alert: "Procedure stalled"        |
+----+----------------------------+-----------------------------------+-----------------------------------+
| S6 | Spurious / Unknown Action  | Steps 1 → 2 → [Scratch Head] →    | FSM stays in Step 2 state;        |
|    | (Background gesture)       | 3 → 4 ...                         | Ignores spurious non-step action  |
+----+----------------------------+-----------------------------------+-----------------------------------+
```

### 4.2 Representation Decoupling: HAR vs. FSM Validation Data

- **HAR Training Data:** Segment-level sliding windows of $(60 \times 154)$ feature vectors labeled with an integer `action_id` (0 to 10). The HAR model has **no concept of protocol rules or sequence history**; it is a memoryless temporal window classifier.
- **Sequence Validation Data:** Full-length timestamped action streams (ground-truth sequence of $(action\_id, timestamp)$ pairs) fed into `ProtocolStateMachine.evaluate_action()`. This verifies FSM transition correctness, precondition enforcement, and voice alert triggering independently of computer vision noise.

---

## 5. Part 4 — Data Collection Protocol for Student Team

To maximize real-world robustness without capturing redundant near-identical frames, follow this structured recording matrix:

### 5.1 Hardware & Camera Setup

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Primary Camera** | 1080p (1920×1080) @ 30 FPS, fixed mount | Front-quarter overhead view (45° depression angle) |
| **Secondary Camera** | 1080p (1920×1080) @ 30 FPS, tripod | Eye-level side profile (90° lateral view) |
| **Working Distance** | 0.8 m to 1.4 m from glovebox workspace | Captures head, torso, arms, and all bench apparatus |
| **Lens Settings** | Fixed focus, auto-exposure locked during session | Prevents focus hunting during fast hand movements |

---

### 5.2 Controlled Variations Matrix

```
+---------------------------------------------------------------------------------------------------------+
|                                    DATASET VARIATION AXES (ROBUSTNESS)                                  |
+--------------------------+------------------------------------------------------------------------------+
| 1. Actor Diversity       | Minimum 6 distinct operators (differing heights, arm lengths, body builds)  |
| 2. Hand Dominance        | 4 Right-handed operators, 2 Left-handed operators                            |
| 3. Glove Types & Colors  | Bare hands, Nitrile Blue, Nitrile Purple, Latex White, Heavy Space Gloves   |
| 4. Clothing & Sleeves    | Short sleeves, long white lab coat, dark fleece, bulky space suit sleeves    |
| 5. Lighting Conditions   | • Nominal diffuse overhead LED (500 lux)                                     |
|                          | • High-glare direct directional spotlight (simulating spacecraft LED)       |
|                          | • Dim ambient lighting (150 lux)                                             |
| 6. Apparatus Placement   | Standard layout, mirror layout (vials left, pipette right), clustered layout |
| 7. Execution Pace        | Nominal speed (45–60 s/trial), Slow deliberate (90 s), Fast paced (30 s)     |
| 8. Zero-G Posture Tilt   | Upright standing (0°), Torso pitch-forward (30°), Lateral bank (15°–25°)     |
+--------------------------+------------------------------------------------------------------------------+
```

---

### 5.3 Recording Session Schedule

Each recording session follows this script:
1. **Calibration Shot (10 s):** Empty workbench with all 7 apparatus items placed in standard positions.
2. **Actor Entry & ID (5 s):** Actor steps into view, announces Subject ID and trial number.
3. **Nominal Trials (3 runs):** Complete 10-step protocol performed smoothly.
4. **Deliberate Error Trials (2 runs):** Specific injected anomaly from Table S2–S6.
5. **Background / Idle Noise (1 run):** Actor organizing tools, reaching for notes, resting hands.

**Total Video Volume Target:**
- **60 full-sequence video sessions** (~45–90 s each).
- Total raw video duration: **~65 minutes** (~117,000 raw frames).

---

## 6. Part 5 — Annotation Strategy & Formats

All annotations are stored in lightweight, human-readable text and JSON formats parseable by standard Python scripts without external cloud services.

### 6.1 YOLO Object Detection Format (`data/images/` and `data/labels/`)

Standard normalized bounding-box coordinates for each static frame `frame_XXXXX.txt`:
```text
<class_id> <x_center> <y_center> <width> <height>
```
*Example (`frame_00240.txt`):*
```text
0 0.4521 0.6124 0.0820 0.2310
1 0.6812 0.7210 0.1450 0.1820
2 0.4120 0.5890 0.0350 0.0780
```

### 6.2 Temporal Action Interval Format (`data/annotations/action_manifest.json`)

```json
{
  "dataset_version": "1.0.0",
  "video_id": "session_04_actor_B_nominal_01",
  "fps": 30.0,
  "total_frames": 1650,
  "metadata": {
    "actor_id": "ACTOR_02",
    "hand_dominance": "RIGHT",
    "glove_type": "NITRILE_BLUE",
    "lighting": "DIFFUSE_LED",
    "camera_view": "FRONT_QUARTER_45DEG"
  },
  "segments": [
    {
      "segment_id": 1,
      "action_id": 1,
      "action_name": "ACTION_SANITIZE",
      "start_frame": 60,
      "end_frame": 210,
      "start_time_sec": 2.00,
      "end_time_sec": 7.00
    },
    {
      "segment_id": 2,
      "action_id": 2,
      "action_name": "ACTION_OPEN_DOOR",
      "start_frame": 225,
      "end_frame": 315,
      "start_time_sec": 7.50,
      "end_time_sec": 10.50
    }
  ]
}
```

### 6.3 Sequence Validation Ground-Truth Format (`data/annotations/fsm_eval_manifest.json`)

```json
{
  "trial_id": "EVAL_ERR01_SKIP_TIP",
  "expected_outcome": "SKIPPED_STEP",
  "faulty_step_id": 5,
  "action_sequence": [
    {"action": "ACTION_SANITIZE", "timestamp": 2.0},
    {"action": "ACTION_OPEN_DOOR", "timestamp": 7.5},
    {"action": "ACTION_RETRIEVE_VIAL", "timestamp": 12.0},
    {"action": "ACTION_PICK_PIPETTE", "timestamp": 16.5},
    {"action": "ACTION_ASPIRATE", "timestamp": 20.0}
  ]
}
```

---

## 7. Part 6 — Leakage-Safe Dataset Partitioning

To guarantee zero data contamination between training and evaluation, data splitting is enforced strictly at the **Subject and Session Level**:

```
+---------------------------------------------------------------------------------------------------------+
|                                  SUBJECT-SEPARATED DATASET PARTITIONS                                   |
+--------------------+------------+------------------------+----------------------------------------------+
| Split              | Proportion | Subject Allocation     | Session Characteristics                      |
+--------------------+------------+------------------------+----------------------------------------------+
| **Train Set**      | **70%**    | Actors 1, 2, 3, 4      | Nominal + Error runs, all lighting variants  |
| **Validation Set** | **15%**    | Actor 5                | Nominal runs, secondary camera angle         |
| **Test Set**       | **15%**    | Actor 6 (unseen actor) | Full benchmark: Nominal + Injected anomalies |
+--------------------+------------+------------------------+----------------------------------------------+
```

### Strict Anti-Leakage Constraints
1. ❌ **No Frame Splitting:** Never split frames from the same video file across train and test sets.
2. ❌ **No Same-Subject Overlap:** Actor 6 videos appear **only** in the final Test Set.
3. ❌ **Temporal Window Boundaries:** Sliding windows $(60 \times 154)$ are extracted within segment boundaries or explicitly marked with transition labels.

---

## 8. Summary of Dataset Targets

| Metric | Target |
| :--- | :--- |
| **Object Detection Images** | 800 annotated frames |
| **Object Detection Bounding Boxes** | ~3,000 labeled instances across 7 classes |
| **Temporal Video Recordings** | 60 full-protocol sessions (~65 min total footage) |
| **Extracted Temporal Sliding Windows** | ~35,000 window tensors of shape `(60, 154)` |
| **Subjects / Operators** | 6 distinct individuals |
| **Storage Footprint** | < 8 GB raw MP4 video; < 45 MB structured JSON/JSONL features |
