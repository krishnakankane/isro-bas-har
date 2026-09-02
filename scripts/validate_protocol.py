"""ISRO BAS HAR — Offline Protocol Replay & Sequence Anomaly Validation.

Replays recorded session manifests or pre-extracted temporal HAR sliding windows through
the deterministic ProtocolStateMachine to verify sequence adherence, step dwell,
preconditions, and anomaly detection.

Scientific Notice:
------------------
This layer provides deterministic state validation and rule-based safety verification.
It does NOT use an AI/ML learned anomaly model.

Usage
-----
    python -m scripts.validate_protocol --manifest data/temporal_har/manifests/SES_001_NOMINAL_actions.json
    python -m scripts.validate_protocol --features data/temporal_har/features/SES_001_NOMINAL_windows.npz --checkpoint models/har/conv1d_bigru_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.activity_engine.har_model import Conv1DBiGRUHAR, HAR_ACTION_CLASSES, NUM_HAR_CLASSES
from src.core.state_machine import ProtocolStateMachine
from src.core.models import AnomalyType
from config.settings import DEFAULT_CONFIDENCE_THRESHOLD


def validate_from_features(
    features_path: Path,
    checkpoint_path: Path,
    protocol_path: Path,
    log_dir: Path,
    device: str = "cpu",
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, object]:
    """Replay pre-extracted feature tensors through trained HAR classifier + State Machine."""
    data = np.load(features_path)
    windows = data["windows"].astype(np.float32)  # (N, 60, 154)
    ranges = data.get("ranges", None)             # (N, 2)
    session_id = str(data.get("session_id", features_path.stem.replace("_windows", "")))

    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    config = ckpt.get("config", {})

    model = Conv1DBiGRUHAR(
        input_dim=config.get("input_dim", 154),
        seq_len=config.get("seq_len", 60),
        conv_channels=config.get("conv_channels", 128),
        gru_hidden_dim=config.get("gru_hidden_dim", 64),
        num_classes=config.get("num_classes", NUM_HAR_CLASSES),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state_machine = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=log_dir,
        session_id=session_id,
    )

    t_step_sim = 0.0
    for idx, win in enumerate(windows):
        win_tensor = torch.from_numpy(win).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(win_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        cid = int(np.argmax(probs))
        conf = float(probs[cid])
        action_name = HAR_ACTION_CLASSES.get(cid, f"ACTION_{cid}")

        frame_num = int(ranges[idx][1]) if ranges is not None else idx * 10
        t_step_sim = float(frame_num) / 30.0

        state_machine.evaluate_action(
            action_name=action_name,
            confidence=conf,
            timestamp=t_step_sim,
            frame=frame_num,
        )

    return _generate_validation_report(state_machine, session_id)


def validate_from_manifest(
    manifest_path: Path,
    protocol_path: Path,
    log_dir: Path,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, object]:
    """Replay annotated ground truth action intervals from manifest through State Machine."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    session_id = manifest.get("session_id", manifest_path.stem.replace("_actions", ""))
    segments = manifest.get("segments", manifest.get("actions", []))

    state_machine = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=5,
        min_confidence=min_confidence,
        log_dir=log_dir,
        session_id=session_id,
    )

    for seg in segments:
        act_name = seg["action_name"]
        start_f = seg.get("start_frame", 0)
        end_f = seg.get("end_frame", start_f + 30)

        # Simulate frame-by-frame progression over interval
        for f in range(start_f, end_f + 1, 5):
            t_sec = float(f) / 30.0
            state_machine.evaluate_action(
                action_name=act_name,
                confidence=1.0,
                timestamp=t_sec,
                frame=f,
            )

    return _generate_validation_report(state_machine, session_id)


def _generate_validation_report(
    fsm: ProtocolStateMachine,
    session_id: str,
) -> Dict[str, object]:
    """Compile formatted validation report from state machine state."""
    total_steps = len(fsm.protocol.steps) if fsm.protocol else 0
    completed_steps_count = len(fsm.completed_steps)
    completion_pct = (completed_steps_count / total_steps * 100.0) if total_steps > 0 else 0.0

    valid_transitions = [t for t in fsm.transition_history if t.valid]
    invalid_transitions = [t for t in fsm.transition_history if not t.valid]

    anomalies_summary = [
        {
            "anomaly_type": a.anomaly_type.name,
            "step": fsm.steps_by_id[a.current_step_id].name if a.current_step_id in fsm.steps_by_id else "UNKNOWN",
            "expected": a.expected_action,
            "observed": a.observed_action,
            "confidence": round(a.confidence, 4),
            "frame": a.frame,
            "time_sec": a.timestamp_sec,
            "explanation": a.explanation,
        }
        for a in fsm.anomaly_history
    ]

    report = {
        "session_id": session_id,
        "protocol_id": fsm.protocol.protocol_id if fsm.protocol else "NONE",
        "total_protocol_steps": total_steps,
        "completed_step_ids": sorted(list(fsm.completed_steps)),
        "completed_steps_count": completed_steps_count,
        "completion_percentage": round(completion_pct, 1),
        "is_completed": fsm.is_completed,
        "completion_timestamp_sec": fsm.completion_timestamp,
        "completion_frame": fsm.completion_frame,
        "total_transitions_observed": len(fsm.transition_history),
        "valid_transitions_count": len(valid_transitions),
        "invalid_transitions_count": len(invalid_transitions),
        "anomalies_detected_count": len(fsm.anomaly_history),
        "anomalies": anomalies_summary,
        "event_log_file": str(fsm.log_file) if fsm.log_file else "None",
    }

    # Print summary to console
    print("\n" + "=" * 68)
    print("ISRO BAS HAR -- PROTOCOL VALIDATION REPORT")
    print("=" * 68)
    print(f"  Session ID            : {session_id}")
    print(f"  Protocol ID           : {report['protocol_id']}")
    print(f"  Total Expected Steps  : {total_steps}")
    print(f"  Completed Steps       : {completed_steps_count}/{total_steps} ({completion_pct:.1f}%)")
    print(f"  Protocol Completion   : {'[COMPLETED]' if fsm.is_completed else '[INCOMPLETE]'}")
    if fsm.is_completed:
        print(f"  Completion Time       : {fsm.completion_timestamp:.2f}s (Frame {fsm.completion_frame})")
    print(f"  Observed Transitions  : {len(fsm.transition_history)} ({len(valid_transitions)} Valid, {len(invalid_transitions)} Invalid)")
    print(f"  Anomalies Detected    : {len(fsm.anomaly_history)}")
    print(f"  JSONL Event Audit Log : {report['event_log_file']}")
    print("=" * 68)

    print("\nStep Execution Summary:")
    if fsm.protocol:
        for s in fsm.protocol.steps:
            done_flag = "[COMPLETED]" if s.step_id in fsm.completed_steps else "[PENDING]"
            print(f"  Step {s.step_id:02d}: {s.name:<25} (Expected: {s.expected_action:<22}) {done_flag}")

    if anomalies_summary:
        print("\nAnomalies Encountered:")
        for idx, anom in enumerate(anomalies_summary, 1):
            print(f"  {idx}. [{anom['anomaly_type']}] Frame {anom['frame']} (t={anom['time_sec']:.2f}s): {anom['explanation']}")
    else:
        print("\nAnomalies Encountered: None (Nominal Execution Verified).")
    print("=" * 68 + "\n")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISRO BAS HAR — Replay and Validate Experiment Protocol")
    parser.add_argument("--manifest", default="", help="Path to action manifest JSON")
    parser.add_argument("--features", default="data/temporal_har/features/SES_001_NOMINAL_windows.npz",
                        help="Path to pre-extracted feature NPZ")
    parser.add_argument("--checkpoint", default="models/har/conv1d_bigru_best.pt",
                        help="Path to HAR model checkpoint .pt")
    parser.add_argument("--protocol", default="config/experiment_protocols.json",
                        help="Path to experiment protocols JSON")
    parser.add_argument("--log_dir", default="data/runtime/protocol_events",
                        help="Directory to write JSONL audit log")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda')")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    log_dir_p = Path(args.log_dir)
    protocol_p = Path(args.protocol)

    if args.manifest and Path(args.manifest).exists():
        validate_from_manifest(
            manifest_path=Path(args.manifest),
            protocol_path=protocol_p,
            log_dir=log_dir_p,
        )
    elif args.features and Path(args.features).exists() and Path(args.checkpoint).exists():
        validate_from_features(
            features_path=Path(args.features),
            checkpoint_path=Path(args.checkpoint),
            protocol_path=protocol_p,
            log_dir=log_dir_p,
            device=args.device,
        )
    elif Path("data/temporal_har/manifests/SES_001_NOMINAL_actions.json").exists():
        validate_from_manifest(
            manifest_path=Path("data/temporal_har/manifests/SES_001_NOMINAL_actions.json"),
            protocol_path=protocol_p,
            log_dir=log_dir_p,
        )
    else:
        print("[Error] No valid manifest or feature dataset found to validate.")
