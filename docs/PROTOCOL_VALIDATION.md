# Intelligent Temporal Protocol Validation & Anomaly Detection

**ISRO Behavioral Activity & Sequence Monitoring (ISRO BAS-HAR)**  
*Phase 10 Architectural Specification & Verification*

---

## 1. System Architecture & Scientific Attribution

To guarantee verifiable safety and mission-critical auditability, the system strictly separates perceptual classification from sequence intelligence:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PERCEPTION STACK                                        │
│  [Camera Feed / Video]                                                                 │
│      ├── Object Detection (YOLOv8)                                                      │
│      ├── Pose Estimation (MediaPipe Pose 33 3D Keypoints)                               │
│      ├── Hand Tracking (MediaPipe Hands 21 3D Landmarks)                                │
│      └── Hand-Object Interaction (HOI Proximity & Grasp Geometry)                       │
│                                      │                                                  │
│                                      ▼                                                  │
│                      [154-D Spatial-Temporal Feature Vector]                            │
│                                      │                                                  │
│                                      ▼                                                  │
│                [Rolling 60-Frame Temporal Feature Buffer]                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 HAR CLASSIFICATION                                      │
│  [Conv1D-BiGRU-Attention Neural Network] (models/har/conv1d_bigru_best.pt)              │
│      ├── 1D-CNN Feature Projection: 154-D -> 128-D                                      │
│      ├── 2-Layer Bidirectional GRU (Hidden Dim: 64)                                     │
│      ├── Temporal Self-Attention Pooling                                                │
│      └── Softmax Probability Distribution over 10 Action Classes                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                INTELLIGENT TEMPORAL PROTOCOL VALIDATION & ANOMALY DETECTOR              │
│                       (Deterministic FSM & Rule-Based Engine)                            │
│                                                                                         │
│  CRITICAL SCIENTIFIC DISTINCTION:                                                       │
│  This layer is a deterministic Finite State Machine with auditable rules.               │
│  It is NOT an AI/ML learned black-box anomaly detector.                                 │
│                                                                                         │
│  Components:                                                                            │
│    1. Protocol Graph & Precondition Enforcer (HOME_DEMO_PROTOCOL_01)                    │
│    2. Confidence-Gated Multi-Frame Dwell Step Advancement (>= 60% confidence)            │
│    3. Categorical Anomaly Discriminator                                                 │
│    4. Deterministic Anomaly Scoring Function S_anomaly in [0.0, 1.0]                    │
│    5. Structured JSONL Runtime Audit Log Streamer                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Protocol Graph (`HOME_DEMO_PROTOCOL_01`)

The protocol definition is specified in `config/experiment_protocols.json`:

| Step ID | Step Name | Expected Action Class | Preconditions | Object Dependency |
|---|---|---|---|---|
| **01** | `IDLE` | `ACTION_IDLE` | `[]` | *None* |
| **02** | `SANITIZE` | `ACTION_SANITIZE` | `[1]` | `bottle` / sanitizer |
| **03** | `HOLD_BOTTLE` | `ACTION_HOLD_BOTTLE` | `[1, 2]` | `bottle` |
| **04** | `PLACE_BOTTLE` | `ACTION_PLACE_BOTTLE` | `[1, 2, 3]` | `bottle` |
| **05** | `HOLD_BOX` | `ACTION_HOLD_BOX` | `[1, 2, 3, 4]` | `box` |
| **06** | `OPEN_BOX` | `ACTION_OPEN_BOX` | `[1, 2, 3, 4, 5]` | `box` |
| **07** | `PICK_OBJECT` | `ACTION_PICK_OBJECT` | `[1, 2, 3, 4, 5, 6]` | `box`, `object` |
| **08** | `TRANSFER_OBJECT` | `ACTION_TRANSFER_OBJECT`| `[1, 2, 3, 4, 5, 6, 7]` | `object` |
| **09** | `RETURN_OBJECT` | `ACTION_RETURN_OBJECT` | `[1, 2, 3, 4, 5, 6, 7, 8]` | `box`, `object` |
| **10** | `CLOSE_BOX` | `ACTION_CLOSE_BOX` | `[1, 2, 3, 4, 5, 6, 7, 8, 9]` | `box` |

---

## 3. Categorical Anomaly Taxonomy

When an action prediction is received by the state machine, it is evaluated deterministically against the protocol state:

| Anomaly Type Enum | Trigger Condition | Severity | Default Score Penalty |
|---|---|---|---|
| `NONE` | Action matches current step and fulfills preconditions. | `NOMINAL` | $+0.00$ |
| `LOW_CONFIDENCE` | Prediction confidence $< 0.60$ or uncertain filter state. Protocol does not advance. | `UNCERTAIN` | $+0.25 \times (1.0 - c)$ |
| `REPEATED_ACTION` | An already-completed step action is performed again after transitioning away. | `MEDIUM` | $+0.30$ |
| `OUT_OF_ORDER_ACTION` | A valid step action is executed before prior required prerequisite steps are completed. | `HIGH` | $+0.70$ |
| `UNEXPECTED_ACTION` | An action not defined within the protocol graph is observed. | `HIGH` | $+0.85$ |
| `INVALID_TRANSITION` | An invalid transition between activity states is detected. | `HIGH` | $+0.85$ |
| `MISSING_EXPECTED_ACTION` | A required step was skipped when a downstream step starts. | `HIGH` | $+0.70$ |

---

## 4. Deterministic Anomaly Scoring Formula

The anomaly score $S_{\text{anomaly}} \in [0.0, 1.0]$ is computed deterministically:

$$S_{\text{anomaly}} = \begin{cases}
0.0 & \text{if nominal execution or idle dwell} \\
\min(1.0, 0.25 \cdot (1.0 - c)) & \text{if } \text{confidence } c < 0.60 \\
0.30 & \text{if } \text{REPEATED\_ACTION} \\
0.70 & \text{if } \text{OUT\_OF\_ORDER\_ACTION or SKIPPED\_STEP} \\
0.85 & \text{if } \text{UNEXPECTED\_ACTION or INVALID\_TRANSITION}
\end{cases}$$

---

## 5. Structured JSONL Runtime Audit Logging

Every transition and anomaly event is appended as a single JSON line to `data/runtime/protocol_events/<session_id>_protocol_events.jsonl`.

### Sample Event Record (`TRANSITION`):
```json
{
  "session_id": "SES_001_NOMINAL",
  "event_type": "TRANSITION",
  "timestamp_sec": 12.7,
  "frame": 381,
  "previous_action": "ACTION_IDLE",
  "current_action": "ACTION_SANITIZE",
  "confidence": 1.0,
  "protocol_step_id": 2,
  "expected_action": "ACTION_SANITIZE",
  "valid": true,
  "anomaly_type": null
}
```

### Sample Event Record (`ANOMALY`):
```json
{
  "session_id": "SES_001_NOMINAL",
  "event_type": "ANOMALY",
  "timestamp_sec": 45.2,
  "frame": 1356,
  "protocol_step_id": 3,
  "expected_action": "ACTION_HOLD_BOTTLE",
  "observed_action": "ACTION_OPEN_BOX",
  "confidence": 0.95,
  "anomaly_type": "OUT_OF_ORDER_ACTION",
  "anomaly_score": 0.7,
  "severity": "HIGH",
  "explanation": "Out-of-order step 6 ('OPEN_BOX') detected. Expected step 3 ('HOLD_BOTTLE'). Skipped: [3, 4, 5]"
}
```

---

## 6. Offline Validation Verification on Real Recording (`SES_001_NOMINAL`)

Offline replay command:
```bash
python -m scripts.validate_protocol --manifest data/temporal_har/manifests/SES_001_NOMINAL_actions.json
```

Output:
```
====================================================================
ISRO BAS HAR -- PROTOCOL VALIDATION REPORT
====================================================================
  Session ID            : SES_001_NOMINAL
  Protocol ID           : HOME_DEMO_PROTOCOL_01
  Total Expected Steps  : 10
  Completed Steps       : 10/10 (100.0%)
  Protocol Completion   : [COMPLETED]
  Completion Time       : 84.93s (Frame 2548)
  Observed Transitions  : 10 (10 Valid, 0 Invalid)
  Anomalies Detected    : 0
  JSONL Event Audit Log : data\runtime\protocol_events\SES_001_NOMINAL_protocol_events.jsonl
====================================================================
```
