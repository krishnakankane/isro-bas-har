"""ISRO BAS HAR — Phase 11 Controlled Adversarial Protocol & Anomaly Validation.

Executes a systematic battery of controlled, software-injected adversarial sequence tests
against the deterministic ProtocolStateMachine without requiring additional human recordings.

SCIENTIFIC NOTICE:
------------------
These tests evaluate deterministic state-machine and rule-based safety invariants under
controlled software-injected conditions. They are NOT empirical real-world human anomaly trials.

Scenarios Tested:
  A. NOMINAL: Full 10-step nominal sequence (10/10 completed, 0 anomalies)
  B. OUT_OF_ORDER: Out-of-sequence jump (OUT_OF_ORDER_ACTION, no advancement)
  C. REPEATED_ACTION: Re-executing completed step after transitioning (REPEATED_ACTION)
  D. UNEXPECTED_ACTION: Injecting unknown/alien action class (UNEXPECTED_ACTION)
  E. LOW_CONFIDENCE_BOUNDARY: Testing 0.59, 0.60, 0.64, 0.65, 0.66 vs 0.65 threshold
  F. INVALID_TRANSITION: Disallowed direct transition sequence (INVALID_TRANSITION)
  G. PREMATURE_COMPLETION: Triggering Step 10 action early (OUT_OF_ORDER, no completion)
  H. DUPLICATE_COMPLETION: Post-completion idempotent stability (Idempotent 1 completion)
  I. MISSING_ACTION_TIMEOUT: Step dwell timeout during idle (TIMEOUT_EXCEEDED)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DEFAULT_CONFIDENCE_THRESHOLD
from src.core.models import AnomalyType, ValidationResult
from src.core.state_machine import ProtocolStateMachine


def run_adversarial_battery(
    protocol_path: Path,
    output_report_path: Path,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Execute complete battery of adversarial protocol validation scenarios."""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_results: List[Dict[str, Any]] = []

    # ── Scenario A: NOMINAL ──────────────────────────────────────────────────
    fsm_a = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_NOMINAL",
    )
    nominal_actions = [
        "ACTION_IDLE", "ACTION_SANITIZE", "ACTION_HOLD_BOTTLE", "ACTION_PLACE_BOTTLE",
        "ACTION_HOLD_BOX", "ACTION_OPEN_BOX", "ACTION_PICK_OBJECT", "ACTION_TRANSFER_OBJECT",
        "ACTION_RETURN_OBJECT", "ACTION_CLOSE_BOX",
    ]
    frame_ctr = 0
    for act in nominal_actions:
        for _ in range(3):
            frame_ctr += 1
            fsm_a.evaluate_action(act, confidence=0.95, timestamp=frame_ctr * 0.1, frame=frame_ctr)

    passed_a = (len(fsm_a.completed_steps) == 10 and fsm_a.is_completed and len(fsm_a.anomaly_history) == 0)
    scenario_results.append({
        "scenario_id": "SCENARIO_A_NOMINAL",
        "name": "Nominal 10-Step Execution",
        "description": "Sequential execution of all 10 steps with high confidence (0.95).",
        "expected_behavior": "10/10 steps completed, is_completed=True, 0 anomalies.",
        "actual_completed_steps": len(fsm_a.completed_steps),
        "is_completed": fsm_a.is_completed,
        "anomalies_detected": [a.anomaly_type.name for a in fsm_a.anomaly_history],
        "anomaly_score_max": max([a.confidence for a in fsm_a.anomaly_history], default=0.0),
        "passed": passed_a,
    })

    # ── Scenario B: OUT_OF_ORDER ─────────────────────────────────────────────
    fsm_b = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_OUT_OF_ORDER",
    )
    # Step 1 IDLE
    for f in range(3):
        fsm_b.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
    # Step 2 SANITIZE
    for f in range(3, 6):
        fsm_b.evaluate_action("ACTION_SANITIZE", confidence=0.95, timestamp=f * 0.1, frame=f)
    # Out of order: Jump directly to Step 7 PICK_OBJECT (skipping 3, 4, 5, 6)
    res_b = fsm_b.evaluate_action("ACTION_PICK_OBJECT", confidence=0.95, timestamp=0.7, frame=7)

    passed_b = (
        not res_b.is_valid
        and res_b.anomaly_type in (AnomalyType.OUT_OF_ORDER_ACTION, AnomalyType.SKIPPED_STEP)
        and res_b.current_step_id == 3  # Expected step remains Step 3 (HOLD_BOTTLE)
        and 7 not in fsm_b.completed_steps
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_B_OUT_OF_ORDER",
        "name": "Out-of-Order Step Injection",
        "description": "Jump directly from Step 2 (SANITIZE) to Step 7 (PICK_OBJECT).",
        "expected_behavior": "OUT_OF_ORDER_ACTION anomaly, protocol remains at Step 3.",
        "actual_anomaly_type": res_b.anomaly_type.name,
        "anomaly_score": res_b.anomaly_score,
        "current_step_id": res_b.current_step_id,
        "passed": passed_b,
    })

    # ── Scenario C: REPEATED_ACTION ──────────────────────────────────────────
    fsm_c = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_REPEATED",
    )
    for f in range(3):
        fsm_c.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
    for f in range(3, 6):
        fsm_c.evaluate_action("ACTION_SANITIZE", confidence=0.95, timestamp=f * 0.1, frame=f)
    for f in range(6, 9):
        fsm_c.evaluate_action("ACTION_HOLD_BOTTLE", confidence=0.95, timestamp=f * 0.1, frame=f)
    # Repeated action: SANITIZE executed again at Step 4
    res_c = fsm_c.evaluate_action("ACTION_SANITIZE", confidence=0.95, timestamp=1.0, frame=10)

    passed_c = (
        not res_c.is_valid
        and res_c.anomaly_type == AnomalyType.REPEATED_ACTION
        and res_c.anomaly_score == 0.30
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_C_REPEATED_ACTION",
        "name": "Repeated Completed Action",
        "description": "Re-execute Step 2 (SANITIZE) after advancing to Step 4 (PLACE_BOTTLE).",
        "expected_behavior": "REPEATED_ACTION anomaly, anomaly_score=0.30.",
        "actual_anomaly_type": res_c.anomaly_type.name,
        "anomaly_score": res_c.anomaly_score,
        "passed": passed_c,
    })

    # ── Scenario D: UNEXPECTED_ACTION ────────────────────────────────────────
    fsm_d = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_UNEXPECTED",
    )
    res_d = fsm_d.evaluate_action("ACTION_SPACECRAFT_PILOTING", confidence=0.95, timestamp=0.5, frame=15)
    passed_d = (
        not res_d.is_valid
        and res_d.anomaly_type == AnomalyType.UNEXPECTED_ACTION
        and res_d.anomaly_score == 0.85
        and len(fsm_d.completed_steps) == 0
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_D_UNEXPECTED_ACTION",
        "name": "Unexpected Alien Action",
        "description": "Inject an action class not present anywhere in the active protocol.",
        "expected_behavior": "UNEXPECTED_ACTION anomaly, anomaly_score=0.85, 0 completed steps.",
        "actual_anomaly_type": res_d.anomaly_type.name,
        "anomaly_score": res_d.anomaly_score,
        "passed": passed_d,
    })

    # ── Scenario E: LOW_CONFIDENCE_BOUNDARY ──────────────────────────────────
    boundary_values = [0.59, 0.60, 0.64, 0.65, 0.66]
    boundary_results = []
    all_boundary_passed = True

    for val in boundary_values:
        fsm_e = ProtocolStateMachine(
            protocol_json_path=protocol_path,
            min_dwell_frames=3,
            min_confidence=min_confidence,
            log_dir=None,
            session_id=f"ADV_CONF_{val}",
        )
        res_e = fsm_e.evaluate_action("ACTION_IDLE", confidence=val, timestamp=0.1, frame=3)
        expected_valid = (val >= min_confidence)
        test_passed = (res_e.is_valid == expected_valid)
        if not test_passed:
            all_boundary_passed = False

        boundary_results.append({
            "confidence_injected": val,
            "threshold": min_confidence,
            "is_valid": res_e.is_valid,
            "status_text": res_e.status_text,
            "anomaly_type": res_e.anomaly_type.name,
            "anomaly_score": res_e.anomaly_score,
            "passed": test_passed,
        })

    scenario_results.append({
        "scenario_id": "SCENARIO_E_LOW_CONFIDENCE_BOUNDARY",
        "name": "Confidence Threshold Boundary Testing",
        "description": f"Evaluate confidence values [0.59, 0.60, 0.64, 0.65, 0.66] against authoritative threshold ({min_confidence}).",
        "expected_behavior": "Values < 0.65 flagged UNCERTAIN; values >= 0.65 accepted.",
        "boundary_tests": boundary_results,
        "passed": all_boundary_passed,
    })

    # ── Scenario F: INVALID_TRANSITION ───────────────────────────────────────
    fsm_f = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_INVALID_TRANS",
    )
    # Direct jump to an arbitrary non-sequential action
    res_f = fsm_f.evaluate_action("ACTION_RETURN_OBJECT", confidence=0.90, timestamp=0.2, frame=6)
    passed_f = (
        not res_f.is_valid
        and res_f.anomaly_type in (AnomalyType.OUT_OF_ORDER_ACTION, AnomalyType.INVALID_TRANSITION)
        and res_f.anomaly_score >= 0.70
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_F_INVALID_TRANSITION",
        "name": "Forced Disallowed Transition",
        "description": "Forced jump to Step 9 (RETURN_OBJECT) from initial unstarted state.",
        "expected_behavior": "Disallowed transition rejected with anomaly score >= 0.70.",
        "actual_anomaly_type": res_f.anomaly_type.name,
        "anomaly_score": res_f.anomaly_score,
        "passed": passed_f,
    })

    # ── Scenario G: PREMATURE_COMPLETION ─────────────────────────────────────
    fsm_g = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_PREMATURE_COMP",
    )
    # Inject Step 10 action (ACTION_CLOSE_BOX) at initial state
    res_g = fsm_g.evaluate_action("ACTION_CLOSE_BOX", confidence=0.95, timestamp=0.5, frame=15)
    passed_g = (
        not res_g.is_valid
        and not fsm_g.is_completed
        and res_g.status_text != "COMPLETED"
        and res_g.anomaly_type in (AnomalyType.OUT_OF_ORDER_ACTION, AnomalyType.SKIPPED_STEP)
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_G_PREMATURE_COMPLETION",
        "name": "Premature Completion Prevention",
        "description": "Attempt to trigger ACTION_CLOSE_BOX without completing steps 1..9.",
        "expected_behavior": "Completion blocked, OUT_OF_ORDER_ACTION anomaly recorded.",
        "is_completed": fsm_g.is_completed,
        "actual_anomaly_type": res_g.anomaly_type.name,
        "passed": passed_g,
    })

    # ── Scenario H: DUPLICATE_COMPLETION ─────────────────────────────────────
    fsm_h = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_DUPLICATE_COMP",
    )
    f_h = 0
    for act in nominal_actions:
        for _ in range(3):
            f_h += 1
            fsm_h.evaluate_action(act, confidence=0.95, timestamp=f_h * 0.1, frame=f_h)

    initial_comp_time = fsm_h.completion_timestamp
    initial_transitions_count = len(fsm_h.transition_history)

    # Continue feeding 15 frames of CLOSE_BOX after completion
    for _ in range(15):
        f_h += 1
        res_h = fsm_h.evaluate_action("ACTION_CLOSE_BOX", confidence=0.95, timestamp=f_h * 0.1, frame=f_h)

    passed_h = (
        fsm_h.is_completed
        and fsm_h.completion_timestamp == initial_comp_time
        and res_h.status_text == "COMPLETED"
        and res_h.anomaly_score == 0.0
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_H_DUPLICATE_COMPLETION",
        "name": "Duplicate Completion Idempotency",
        "description": "Continue feeding post-completion frames of CLOSE_BOX.",
        "expected_behavior": "Idempotent COMPLETED state; exact 1 completion recorded.",
        "is_completed": fsm_h.is_completed,
        "completion_timestamp": fsm_h.completion_timestamp,
        "status_text": res_h.status_text,
        "passed": passed_h,
    })

    # ── Scenario I: MISSING_ACTION_TIMEOUT ───────────────────────────────────
    fsm_i = ProtocolStateMachine(
        protocol_json_path=protocol_path,
        min_dwell_frames=3,
        min_confidence=min_confidence,
        log_dir=None,
        session_id="ADV_MISSING_TIMEOUT",
    )
    # Complete Step 1 (IDLE)
    for f in range(3):
        fsm_i.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=f * 0.1, frame=f)
    # Now in Step 2 (SANITIZE, max_duration_sec = 30.0)
    # Simulate stalling in idle for 35.0 seconds (exceeding timeout)
    res_i = fsm_i.evaluate_action("ACTION_IDLE", confidence=0.95, timestamp=36.0, frame=1080)
    passed_i = (
        not res_i.is_valid
        and res_i.anomaly_type in (AnomalyType.TIMEOUT_EXCEEDED, AnomalyType.MISSING_EXPECTED_ACTION)
        and res_i.anomaly_score == 0.75
    )
    scenario_results.append({
        "scenario_id": "SCENARIO_I_MISSING_ACTION_TIMEOUT",
        "name": "Missing-Action Temporal Timeout",
        "description": "Remain in neutral idle dwell past max step duration without expected action.",
        "expected_behavior": "TIMEOUT_EXCEEDED anomaly, anomaly_score=0.75.",
        "actual_anomaly_type": res_i.anomaly_type.name,
        "anomaly_score": res_i.anomaly_score,
        "passed": passed_i,
    })

    # ── Compile Summary Report ───────────────────────────────────────────────
    all_passed = all(s["passed"] for s in scenario_results)
    full_report = {
        "report_version": "1.0.0",
        "evaluation_type": "CONTROLLED_SOFTWARE_ADVERSARIAL_VALIDATION",
        "disclaimer": "These tests evaluate deterministic FSM safety invariants under software-injected conditions. They are NOT human trial empirical results.",
        "authoritative_confidence_threshold": min_confidence,
        "total_scenarios_tested": len(scenario_results),
        "total_scenarios_passed": sum(1 for s in scenario_results if s["passed"]),
        "all_scenarios_passed": all_passed,
        "scenarios": scenario_results,
    }

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Console display
    print("\n" + "=" * 72)
    print("ISRO BAS HAR — PHASE 11 ADVERSARIAL VALIDATION BATTERY")
    print("=" * 72)
    print(f"  Authoritative Confidence Threshold : {min_confidence}")
    print(f"  Total Scenarios Evaluated          : {len(scenario_results)}")
    print(f"  Scenarios Passed                   : {full_report['total_scenarios_passed']}/{len(scenario_results)}")
    print(f"  Overall Status                     : {'[ALL PASSED]' if all_passed else '[FAILURES DETECTED]'}")
    print(f"  Machine-Readable JSON Report       : {output_report_path}")
    print("=" * 72)

    for idx, s in enumerate(scenario_results, 1):
        status_flag = "[PASS]" if s["passed"] else "[FAIL]"
        print(f"  {idx:02d}. {s['scenario_id']:<36} {status_flag}")
    print("=" * 72 + "\n")

    return full_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISRO BAS HAR — Phase 11 Adversarial Protocol Validator")
    parser.add_argument("--protocol", default="config/experiment_protocols.json",
                        help="Path to experiment protocols JSON")
    parser.add_argument("--output", default="data/runtime/protocol_validation/phase11_adversarial_report.json",
                        help="Output JSON report path")
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Authoritative confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_adversarial_battery(
        protocol_path=Path(args.protocol),
        output_report_path=Path(args.output),
        min_confidence=args.min_confidence,
    )
