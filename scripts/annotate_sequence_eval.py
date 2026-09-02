"""ISRO BAS HAR — Protocol Sequence Error & Benchmark Ground-Truth Creator.

Generates ground-truth evaluation manifests for testing the Protocol State Machine
against correct sequences, skipped steps, out-of-order execution, repetitions, and stalls.

Stores the expected nominal sequence and the observed trial sequence separately,
providing decoupled ground truth for deterministic FSM benchmark suites.

Usage
-----
# Generate a nominal 10-step sequence trial manifest:
    python -m scripts.annotate_sequence_eval --trial EVAL_NOMINAL_01 --actor ACTOR_01 --type CORRECT_SEQUENCE

# Generate an injected skipped-step error manifest (e.g. skip step 5 MOUNT_TIP):
    python -m scripts.annotate_sequence_eval --trial EVAL_ERR_SKIP_TIP --actor ACTOR_02 \
        --type SKIPPED_STEP --faulty_step 5 --skip 5

# Generate an out-of-order error manifest:
    python -m scripts.annotate_sequence_eval --trial EVAL_ERR_OUT_OF_ORDER --actor ACTOR_03 \
        --type OUT_OF_ORDER --faulty_step 7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import AnomalyType

STANDARD_PROTOCOL_STEPS = [
    {"step_id": 1, "name": "IDLE", "action": "ACTION_IDLE", "expected_duration": 3.0},
    {"step_id": 2, "name": "SANITIZE", "action": "ACTION_SANITIZE", "expected_duration": 4.0},
    {"step_id": 3, "name": "HOLD_BOTTLE", "action": "ACTION_HOLD_BOTTLE", "expected_duration": 3.0},
    {"step_id": 4, "name": "PLACE_BOTTLE", "action": "ACTION_PLACE_BOTTLE", "expected_duration": 2.5},
    {"step_id": 5, "name": "HOLD_BOX", "action": "ACTION_HOLD_BOX", "expected_duration": 3.0},
    {"step_id": 6, "name": "OPEN_BOX", "action": "ACTION_OPEN_BOX", "expected_duration": 3.5},
    {"step_id": 7, "name": "PICK_OBJECT", "action": "ACTION_PICK_OBJECT", "expected_duration": 3.0},
    {"step_id": 8, "name": "TRANSFER_OBJECT", "action": "ACTION_TRANSFER_OBJECT", "expected_duration": 3.5},
    {"step_id": 9, "name": "RETURN_OBJECT", "action": "ACTION_RETURN_OBJECT", "expected_duration": 3.0},
    {"step_id": 10, "name": "CLOSE_BOX", "action": "ACTION_CLOSE_BOX", "expected_duration": 2.5},
]


def create_sequence_manifest(
    trial_id: str,
    actor_id: str = "ACTOR_01",
    protocol_id: str = "HOME_DEMO_PROTOCOL_01",
    anomaly_type: str = "CORRECT_SEQUENCE",
    faulty_step_id: int = 0,
    skip_step_ids: Optional[List[int]] = None,
    repeat_step_ids: Optional[List[int]] = None,
    stall_step_id: int = 0,
    stall_duration_sec: float = 45.0,
    notes: str = "",
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Build and save an FSM sequence validation trial manifest."""
    out_dir = output_dir or (_PROJECT_ROOT / "data" / "sequence_validation" / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{trial_id}.json"

    skip_set = set(skip_step_ids or [])
    repeat_set = set(repeat_step_ids or [])

    observed_actions: List[Dict[str, object]] = []
    current_time = 2.0

    for step in STANDARD_PROTOCOL_STEPS:
        sid = step["step_id"]

        if sid in skip_set:
            continue  # Deliberately omit this step

        # Normal execution entry
        dur = step["expected_duration"]
        observed_actions.append({
            "step_id": sid,
            "action_name": step["action"],
            "timestamp_sec": round(current_time, 2),
            "confidence": 0.95,
        })
        current_time += dur + 1.0

        # Injected repetition
        if sid in repeat_set:
            observed_actions.append({
                "step_id": sid,
                "action_name": step["action"],
                "timestamp_sec": round(current_time, 2),
                "confidence": 0.92,
                "is_repeated": True,
            })
            current_time += dur + 0.5

        # Injected stall
        if sid == stall_step_id:
            current_time += stall_duration_sec

    # Handle Out of Order: swap steps 6 and 7 if specified
    if anomaly_type == "OUT_OF_ORDER" and len(observed_actions) >= 7:
        for idx, act in enumerate(observed_actions):
            if act["step_id"] == 6:
                idx_6 = idx
            elif act["step_id"] == 7:
                idx_7 = idx
        if "idx_6" in locals() and "idx_7" in locals():
            observed_actions[idx_6], observed_actions[idx_7] = observed_actions[idx_7], observed_actions[idx_6]

    manifest = {
        "trial_id": trial_id,
        "actor_id": actor_id,
        "protocol_id": protocol_id,
        "anomaly_category": anomaly_type,
        "faulty_step_id": faulty_step_id,
        "expected_sequence": STANDARD_PROTOCOL_STEPS,
        "observed_sequence": observed_actions,
        "expected_fsm_outcome": {
            "CORRECT_SEQUENCE": "VALID_COMPLETED",
            "SKIPPED_STEP": "SKIPPED_STEP",
            "OUT_OF_ORDER": "OUT_OF_SEQUENCE",
            "REPEATED_STEP": "STEP_REPETITION",
            "TIMEOUT_STALL": "TIMEOUT_EXCEEDED",
            "UNEXPECTED_ACTION": "UNKNOWN_ANOMALY",
        }.get(anomaly_type, "UNKNOWN"),
        "notes": notes,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SequenceAnnotator] Manifest written ({anomaly_type}) → {manifest_path}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Protocol Sequence Error Manifest Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--trial", required=True, help="Unique Trial ID")
    parser.add_argument("--actor", default="ACTOR_01", help="Operator ID")
    parser.add_argument("--protocol", default="HOME_DEMO_PROTOCOL_01", help="Protocol ID")
    parser.add_argument(
        "--type",
        default="CORRECT_SEQUENCE",
        choices=["CORRECT_SEQUENCE", "SKIPPED_STEP", "OUT_OF_ORDER", "REPEATED_STEP", "TIMEOUT_STALL", "UNEXPECTED_ACTION"],
        help="Injected anomaly category",
    )
    parser.add_argument("--faulty_step", type=int, default=0, help="Target step ID with anomaly")
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="Step ID(s) to skip")
    parser.add_argument("--repeat", type=int, nargs="*", default=[], help="Step ID(s) to repeat")
    parser.add_argument("--stall_step", type=int, default=0, help="Step ID to inject duration timeout")
    parser.add_argument("--notes", default="", help="Trial notes")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_sequence_manifest(
        trial_id=args.trial,
        actor_id=args.actor,
        protocol_id=args.protocol,
        anomaly_type=args.type,
        faulty_step_id=args.faulty_step,
        skip_step_ids=args.skip,
        repeat_step_ids=args.repeat,
        stall_step_id=args.stall_step,
        notes=args.notes,
    )
