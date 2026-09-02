"""Deterministic Experiment Sequence Validation State Machine & Anomaly Detector.

Scientific Architecture Note:
-----------------------------
This layer implements a deterministic finite state machine (FSM) and rule-based
temporal protocol verification. It does NOT use an AI/ML learned anomaly model,
ensuring strict interpretability, auditable preconditions, and deterministic safety.

Features:
  - Step Precondition & Sequence Verification for HOME_DEMO_PROTOCOL_01
  - Explicit Anomaly Categorization (OUT_OF_ORDER, UNEXPECTED, REPEATED, MISSING, LOW_CONFIDENCE)
  - Interpretable Deterministic Anomaly Scoring in [0.0, 1.0]
  - Multi-frame Dwell Step Advancement with Confidence Gating (>= min_confidence)
  - Transition History & Structured JSONL Runtime Audit Logging
  - Idempotent Protocol Completion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .models import (
    AnomalyEvent,
    AnomalyType,
    ExperimentStep,
    ProtocolDefinition,
    TransitionEvent,
    ValidationResult,
)
from .interfaces import ProtocolValidatorInterface


try:
    from config.settings import DEFAULT_CONFIDENCE_THRESHOLD
except ImportError:
    DEFAULT_CONFIDENCE_THRESHOLD = 0.65


class ProtocolStateMachine(ProtocolValidatorInterface):
    """Deterministic Finite State Machine (FSM) for validating experiment protocol sequences."""

    def __init__(
        self,
        protocol_json_path: Optional[Union[str, Path]] = None,
        min_dwell_frames: int = 5,
        min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
        log_dir: Optional[Union[str, Path]] = "data/runtime/protocol_events",
        session_id: str = "SESSION_001",
    ):
        self.min_dwell_frames = min_dwell_frames
        self.min_confidence = min_confidence
        self.session_id = session_id

        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"{self.session_id}_protocol_events.jsonl"
        else:
            self.log_file = None

        self.protocol: Optional[ProtocolDefinition] = None
        self.steps_by_id: Dict[int, ExperimentStep] = {}
        self.steps_by_action: Dict[str, ExperimentStep] = {}

        self.completed_steps: Set[int] = set()
        self.current_step_idx: int = 0
        self.step_dwell_counter: int = 0
        self.step_start_time: float = 0.0

        self.is_completed: bool = False
        self.completion_timestamp: Optional[float] = None
        self.completion_frame: Optional[int] = None

        self.previous_stable_action: str = ""
        self.active_step_action: str = ""
        self.transition_history: List[TransitionEvent] = []
        self.anomaly_history: List[AnomalyEvent] = []

        if protocol_json_path:
            self.load_from_json(str(protocol_json_path))

    def load_from_json(self, filepath: Union[str, Path]) -> None:
        """Load protocol definitions from a JSON file."""
        with open(str(filepath), "r", encoding="utf-8") as f:
            data = json.load(f)

        steps = [
            ExperimentStep(
                step_id=item["step_id"],
                name=item["name"],
                description=item["description"],
                required_objects=item.get("required_objects", []),
                expected_action=item["expected_action"],
                preconditions=item.get("preconditions", []),
                min_duration_sec=item.get("min_duration_sec", 1.0),
                max_duration_sec=item.get("max_duration_sec", 30.0),
                voice_prompt_on_start=item.get("voice_prompt_on_start", ""),
                voice_prompt_on_complete=item.get("voice_prompt_on_complete", ""),
            )
            for item in data.get("steps", [])
        ]

        self.protocol = ProtocolDefinition(
            protocol_id=data.get("protocol_id", "UNKNOWN"),
            protocol_name=data.get("protocol_name", "Unnamed Protocol"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            steps=steps,
        )
        self.load_protocol(self.protocol)

    def load_protocol(self, protocol_def: ProtocolDefinition) -> None:
        """Initialize step lookup mappings and reset state."""
        self.protocol = protocol_def
        self.steps_by_id = {step.step_id: step for step in self.protocol.steps}
        self.steps_by_action = {step.expected_action: step for step in self.protocol.steps}
        self.reset()

    def reset(self) -> None:
        """Reset state machine to the beginning of the experiment."""
        self.completed_steps = set()
        self.step_completion_timestamps: Dict[int, float] = {}
        self.current_step_idx = 0
        self.step_dwell_counter = 0
        self.step_start_time = 0.0
        self.is_completed = False
        self.completion_timestamp = None
        self.completion_frame = None
        self.previous_stable_action = ""
        self.active_step_action = ""
        self.transition_history.clear()
        self.anomaly_history.clear()

    def get_expected_step(self) -> Optional[ExperimentStep]:
        """Return the current anticipated experiment step in the sequence."""
        if not self.protocol or self.current_step_idx >= len(self.protocol.steps):
            return None
        return self.protocol.steps[self.current_step_idx]

    def _log_event(self, event_dict: Dict[str, object]) -> None:
        """Append structured event record to JSONL log."""
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event_dict) + "\n")
            except Exception:
                pass  # Non-blocking telemetry

    def evaluate_action(
        self,
        action_name: str,
        confidence: float,
        timestamp: float,
        frame: int = 0,
    ) -> ValidationResult:
        """Evaluate an incoming action prediction against the deterministic protocol state."""
        # 1. Check if protocol loaded
        if not self.protocol or not self.protocol.steps:
            res = ValidationResult(
                is_valid=False,
                current_step_id=0,
                current_step_name="NO_PROTOCOL_LOADED",
                completed_step_ids=[],
                suggested_next_step_id=0,
                suggested_next_step_name="NONE",
                anomaly_type=AnomalyType.UNKNOWN_ANOMALY,
                anomaly_message="Protocol not loaded into state machine.",
                confidence=confidence,
                is_completed=False,
                anomaly_score=1.0,
                expected_action="NONE",
                observed_action=action_name,
                status_text="ANOMALY",
                step_timestamps=dict(self.step_completion_timestamps),
            )
            return res

        expected_step = self.get_expected_step()

        # 2. Check if already completed (Idempotent)
        if self.is_completed or expected_step is None:
            res = ValidationResult(
                is_valid=True,
                current_step_id=len(self.protocol.steps),
                current_step_name="EXPERIMENT_COMPLETED",
                completed_step_ids=sorted(list(self.completed_steps)),
                suggested_next_step_id=0,
                suggested_next_step_name="DONE",
                confidence=confidence,
                is_completed=True,
                anomaly_score=0.0,
                expected_action="DONE",
                observed_action=action_name,
                status_text="COMPLETED",
                step_timestamps=dict(self.step_completion_timestamps),
            )
            return res

        # 3. Check Confidence Threshold Gating
        is_uncertain = (confidence < self.min_confidence) or action_name.startswith("UNCERTAIN")
        if is_uncertain:
            anomaly_score = min(1.0, max(0.0, round(0.25 * (1.0 - max(0.0, confidence)), 3)))
            res = ValidationResult(
                is_valid=False,
                current_step_id=expected_step.step_id,
                current_step_name=expected_step.name,
                completed_step_ids=sorted(list(self.completed_steps)),
                suggested_next_step_id=expected_step.step_id,
                suggested_next_step_name=expected_step.name,
                anomaly_type=AnomalyType.LOW_CONFIDENCE,
                anomaly_message=f"Low confidence HAR prediction ({confidence*100:.1f}% < {self.min_confidence*100:.0f}% threshold).",
                confidence=confidence,
                is_completed=False,
                anomaly_score=anomaly_score,
                expected_action=expected_step.expected_action,
                observed_action=action_name,
                status_text="UNCERTAIN",
                step_timestamps=dict(self.step_completion_timestamps),
            )
            return res

        # 4. Action matching & sequence analysis
        matched_step = self.steps_by_action.get(action_name)
        anomaly_type = AnomalyType.NONE
        anomaly_msg = ""
        anomaly_score = 0.0
        status_text = "OK"
        is_valid = True

        # Initialize step_start_time on first evaluation
        if self.step_start_time <= 0.0:
            self.step_start_time = timestamp

        # Permitted neutral idle dwell between protocol steps
        if action_name in ("ACTION_IDLE", "IDLE") and expected_step.expected_action not in ("ACTION_IDLE", "IDLE"):
            step_elapsed = timestamp - self.step_start_time
            if expected_step.max_duration_sec > 0.0 and step_elapsed > expected_step.max_duration_sec:
                is_valid = False
                anomaly_type = AnomalyType.TIMEOUT_EXCEEDED
                anomaly_msg = f"Step {expected_step.step_id} ('{expected_step.name}') timed out after {step_elapsed:.1f}s (max allowed: {expected_step.max_duration_sec:.1f}s). Expected action '{expected_step.expected_action}' missing."
                anomaly_score = 0.75
                status_text = "TIMEOUT"
            else:
                is_valid = True
                status_text = "WAITING"
                anomaly_score = 0.0
        # Continuous execution of the active non-idle step that was just completed in this continuous bout
        elif self.active_step_action == action_name and action_name not in ("ACTION_IDLE", "IDLE"):
            # Continuing to dwell/perform the active step
            is_valid = True
            status_text = "COMPLETED" if self.is_completed else "OK"
            anomaly_score = 0.0
        elif matched_step is None:
            # Action does not belong to any protocol step
            is_valid = False
            anomaly_type = AnomalyType.UNEXPECTED_ACTION
            anomaly_msg = f"Unexpected action '{action_name}' unrecognized in protocol."
            anomaly_score = 0.85
            status_text = "ANOMALY"
        elif matched_step.step_id == expected_step.step_id:
            # Nominal expected step
            missing_preconditions = [p for p in matched_step.preconditions if p not in self.completed_steps]
            if missing_preconditions:
                is_valid = False
                anomaly_type = AnomalyType.OUT_OF_ORDER_ACTION
                anomaly_msg = f"Step {matched_step.step_id} executed before prerequisite step {missing_preconditions[0]}."
                anomaly_score = 0.70
                status_text = "ANOMALY"
            else:
                # Accumulate dwell to confirm step
                self.step_dwell_counter += 1
                if self.step_dwell_counter >= self.min_dwell_frames:
                    self.completed_steps.add(matched_step.step_id)
                    self.step_completion_timestamps[matched_step.step_id] = round(timestamp, 2)
                    self.current_step_idx += 1
                    self.step_dwell_counter = 0
                    self.active_step_action = action_name
                    self.step_start_time = timestamp

                    # Check for completion (Step 10 / All steps completed)
                    if self.current_step_idx >= len(self.protocol.steps):
                        self.is_completed = True
                        self.completion_timestamp = timestamp
                        self.completion_frame = frame
                        status_text = "COMPLETED"
        elif matched_step.step_id > expected_step.step_id:
            # Out of order / skipped step
            is_valid = False
            anomaly_type = AnomalyType.OUT_OF_ORDER_ACTION
            skipped_steps = [
                s.step_id for s in self.protocol.steps
                if expected_step.step_id <= s.step_id < matched_step.step_id
                and s.step_id not in self.completed_steps
            ]
            anomaly_msg = f"Out-of-order step {matched_step.step_id} ('{matched_step.name}') detected. Expected step {expected_step.step_id} ('{expected_step.name}'). Skipped: {skipped_steps}"
            anomaly_score = 0.70
            status_text = "ANOMALY"
        elif matched_step.step_id in self.completed_steps:
            # Repeated already-completed action after transitioning away
            is_valid = False
            anomaly_type = AnomalyType.REPEATED_ACTION
            anomaly_msg = f"Repeated step {matched_step.step_id} ('{matched_step.name}') was already completed."
            anomaly_score = 0.30
            status_text = "ANOMALY"
        else:
            is_valid = False
            anomaly_type = AnomalyType.INVALID_TRANSITION
            anomaly_msg = f"Invalid sequence transition for action '{action_name}'."
            anomaly_score = 0.85
            status_text = "ANOMALY"

        # 5. Track Transitions and Anomalies
        next_step = self.get_expected_step()
        suggested_id = next_step.step_id if next_step else 0
        suggested_name = next_step.name if next_step else "DONE"

        if action_name != self.previous_stable_action:
            if action_name != self.active_step_action:
                self.active_step_action = ""
            trans_evt = TransitionEvent(
                timestamp_sec=round(timestamp, 3),
                frame=frame,
                previous_action=self.previous_stable_action or "NONE",
                current_action=action_name,
                expected_action=expected_step.expected_action,
                step_id=expected_step.step_id,
                valid=is_valid,
                anomaly_type=anomaly_type.name if anomaly_type != AnomalyType.NONE else None,
                anomaly_message=anomaly_msg,
            )
            self.transition_history.append(trans_evt)
            self.previous_stable_action = action_name

            self._log_event({
                "session_id": self.session_id,
                "event_type": "TRANSITION",
                "timestamp_sec": round(timestamp, 3),
                "frame": frame,
                "previous_action": trans_evt.previous_action,
                "current_action": action_name,
                "confidence": round(confidence, 4),
                "protocol_step_id": expected_step.step_id,
                "expected_action": expected_step.expected_action,
                "valid": is_valid,
                "anomaly_type": trans_evt.anomaly_type,
            })

        if not is_valid and anomaly_type != AnomalyType.NONE:
            anom_evt = AnomalyEvent(
                anomaly_type=anomaly_type,
                current_step_id=expected_step.step_id,
                current_step_name=expected_step.name,
                expected_action=expected_step.expected_action,
                observed_action=action_name,
                confidence=confidence,
                timestamp_sec=round(timestamp, 3),
                frame=frame,
                severity="HIGH" if anomaly_score >= 0.70 else "MEDIUM",
                explanation=anomaly_msg,
            )
            self.anomaly_history.append(anom_evt)

            self._log_event({
                "session_id": self.session_id,
                "event_type": "ANOMALY",
                "timestamp_sec": round(timestamp, 3),
                "frame": frame,
                "protocol_step_id": expected_step.step_id,
                "expected_action": expected_step.expected_action,
                "observed_action": action_name,
                "confidence": round(confidence, 4),
                "anomaly_type": anomaly_type.name,
                "anomaly_score": anomaly_score,
                "severity": anom_evt.severity,
                "explanation": anomaly_msg,
            })

        return ValidationResult(
            is_valid=is_valid,
            current_step_id=expected_step.step_id,
            current_step_name=expected_step.name,
            completed_step_ids=sorted(list(self.completed_steps)),
            suggested_next_step_id=suggested_id,
            suggested_next_step_name=suggested_name,
            anomaly_type=anomaly_type,
            anomaly_message=anomaly_msg,
            confidence=confidence,
            is_completed=self.is_completed,
            anomaly_score=anomaly_score,
            expected_action=expected_step.expected_action,
            observed_action=action_name,
            status_text=status_text,
            step_timestamps=dict(self.step_completion_timestamps),
        )
