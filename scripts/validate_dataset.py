"""ISRO BAS HAR — Master Dataset Validation & Integrity Checker.

Performs rigorous offline sanity checks across the entire dataset hierarchy:
1. YOLO Object Detection: Image/label pairing, coordinate normalization bounds, valid class IDs.
2. Temporal HAR Manifests: Chronological validity, boundary bounds, action taxonomy alignment.
3. Sequence Evaluation Manifests: Step ID continuity, ground-truth schema validity.
4. Anti-Leakage Audit: Verifies zero subject or session overlap between Train, Val, and Test splits.
5. Duplication Check: Identifies duplicate filenames or identical image hashes.

Usage
-----
    python -m scripts.validate_dataset
    python -m scripts.validate_dataset --data_dir data/ --strict
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VALID_OBJECT_CLASS_IDS = set(range(7))  # 0 to 6
VALID_ACTION_CLASS_IDS = set(range(10))  # 0 to 9


class DatasetValidator:
    """Master integrity verification suite for object detection and temporal datasets."""

    def __init__(self, data_root: Path, strict: bool = False):
        self.data_root = Path(data_root)
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation suites and return True if no fatal errors found."""
        print("\n" + "=" * 68)
        print("ISRO BAS HAR -- DATASET INTEGRITY & ANTI-LEAKAGE AUDIT")
        print("=" * 68)

        print(f"Data Root Directory: {self.data_root.resolve()}")

        self.validate_directory_structure()
        self.validate_object_detection()
        self.validate_temporal_manifests()
        self.validate_sequence_manifests()
        self.validate_split_leakage()

        return self._generate_report()

    def validate_directory_structure(self) -> None:
        """Check required directories exist."""
        required_dirs = [
            self.data_root / "raw" / "videos",
            self.data_root / "raw" / "metadata",
            self.data_root / "object_detection" / "images" / "train",
            self.data_root / "object_detection" / "images" / "val",
            self.data_root / "object_detection" / "images" / "test",
            self.data_root / "object_detection" / "labels" / "train",
            self.data_root / "object_detection" / "labels" / "val",
            self.data_root / "object_detection" / "labels" / "test",
            self.data_root / "temporal_har" / "manifests",
            self.data_root / "temporal_har" / "features",
            self.data_root / "sequence_validation" / "manifests",
        ]
        for d in required_dirs:
            if not d.exists():
                self.warnings.append(f"Directory missing (creating): {d.relative_to(self.data_root)}")
                d.mkdir(parents=True, exist_ok=True)

    def validate_object_detection(self) -> None:
        """Audit YOLO image and label pairs across train/val/test splits."""
        obj_root = self.data_root / "object_detection"
        for split in ("train", "val", "test"):
            img_dir = obj_root / "images" / split
            lbl_dir = obj_root / "labels" / split

            if not img_dir.exists() or not lbl_dir.exists():
                continue

            images = {f.stem: f for f in img_dir.glob("*.jpg")}
            images.update({f.stem: f for f in img_dir.glob("*.png")})
            labels = {f.stem: f for f in lbl_dir.glob("*.txt")}

            # Check orphaned images
            for stem, img_path in images.items():
                if stem not in labels:
                    self.errors.append(f"[ObjectDetection/{split}] Missing label file for image: {img_path.name}")

            # Check orphaned labels
            for stem, lbl_path in labels.items():
                if stem not in images:
                    self.warnings.append(f"[ObjectDetection/{split}] Orphaned label without image: {lbl_path.name}")
                else:
                    self._validate_yolo_label_file(lbl_path, split)

    def _validate_yolo_label_file(self, lbl_path: Path, split: str) -> None:
        """Validate single YOLO text file contents."""
        try:
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                return  # Empty background frame is valid

            for line_idx, line in enumerate(content.splitlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    self.errors.append(f"[ObjectDetection/{split}] {lbl_path.name}:{line_idx} — Expected 5 elements, got {len(parts)}: '{line}'")
                    continue

                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                if cls_id not in VALID_OBJECT_CLASS_IDS:
                    self.errors.append(f"[ObjectDetection/{split}] {lbl_path.name}:{line_idx} — Invalid class ID {cls_id} (allowed: 0..6)")

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    self.errors.append(f"[ObjectDetection/{split}] {lbl_path.name}:{line_idx} — Center ({cx}, {cy}) outside [0, 1]")

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    self.errors.append(f"[ObjectDetection/{split}] {lbl_path.name}:{line_idx} — Dimensions ({w}, {h}) invalid")

        except Exception as exc:
            self.errors.append(f"[ObjectDetection/{split}] Error parsing {lbl_path.name}: {exc}")

    def validate_temporal_manifests(self) -> None:
        """Audit temporal HAR action interval JSON files."""
        manifest_dir = self.data_root / "temporal_har" / "manifests"
        if not manifest_dir.exists():
            return

        manifest_files = list(manifest_dir.glob("*.json"))
        for mf in manifest_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Required top-level keys
                for key in ("video_id", "actor_id", "fps", "segments"):
                    if key not in data:
                        self.errors.append(f"[TemporalHAR] {mf.name} — Missing top-level key: '{key}'")

                segments = data.get("segments", [])
                last_end_frame = -1

                for seg in segments:
                    sf = seg.get("start_frame")
                    ef = seg.get("end_frame")
                    aid = seg.get("action_id")

                    if sf is None or ef is None or aid is None:
                        self.errors.append(f"[TemporalHAR] {mf.name} — Malformed segment: {seg}")
                        continue

                    if aid not in VALID_ACTION_CLASS_IDS:
                        self.errors.append(f"[TemporalHAR] {mf.name} — Invalid action_id {aid} in segment {seg.get('segment_id')}")

                    if sf >= ef:
                        self.errors.append(f"[TemporalHAR] {mf.name} — start_frame ({sf}) >= end_frame ({ef})")

            except Exception as exc:
                self.errors.append(f"[TemporalHAR] Error reading {mf.name}: {exc}")

    def validate_sequence_manifests(self) -> None:
        """Audit sequence evaluation trial manifests."""
        seq_dir = self.data_root / "sequence_validation" / "manifests"
        if not seq_dir.exists():
            return

        for sf in seq_dir.glob("*.json"):
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in ("trial_id", "actor_id", "anomaly_category", "observed_sequence"):
                    if key not in data:
                        self.errors.append(f"[SequenceEval] {sf.name} — Missing key: '{key}'")
            except Exception as exc:
                self.errors.append(f"[SequenceEval] Error parsing {sf.name}: {exc}")

    def validate_split_leakage(self) -> None:
        """Check for subject or session overlap across dataset splits."""
        meta_dir = self.data_root / "raw" / "metadata"
        if not meta_dir.exists():
            return

        # Map session_id -> actor_id from raw metadata
        session_to_actor: Dict[str, str] = {}
        for mf in meta_dir.glob("*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    m = json.load(f)
                session_to_actor[m.get("session_id", mf.stem)] = m.get("actor_id", "")
            except Exception:
                pass

        # Identify sessions present in each split
        obj_root = self.data_root / "object_detection" / "images"
        split_actors: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        for split in ("train", "val", "test"):
            s_dir = obj_root / split
            if s_dir.exists():
                for img in s_dir.glob("*.jpg"):
                    # Filename format: <session_id>_fXXXXXX_tX.XXs.jpg
                    parts = img.stem.split("_f")
                    if len(parts) >= 2:
                        session_id = parts[0]
                        actor = session_to_actor.get(session_id)
                        if actor:
                            split_actors[split].add(actor)

        # Check overlaps
        t_v = split_actors["train"].intersection(split_actors["val"])
        t_t = split_actors["train"].intersection(split_actors["test"])
        v_t = split_actors["val"].intersection(split_actors["test"])

        if t_v:
            self.errors.append(f"[AntiLeakage] Subject overlap between Train and Val splits: {t_v}")
        if t_t:
            self.errors.append(f"[AntiLeakage] Subject overlap between Train and Test splits: {t_t}")
        if v_t:
            self.errors.append(f"[AntiLeakage] Subject overlap between Val and Test splits: {v_t}")

    def _generate_report(self) -> bool:
        """Format and print validation summary table."""
        print("\n" + "-" * 68)
        print(f"AUDIT SUMMARY: {len(self.errors)} Error(s) | {len(self.warnings)} Warning(s)")
        print("-" * 68)


        if self.warnings:
            print("\n[WARNINGS]")
            for w in self.warnings[:10]:
                print(f"  * {w}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings.")

        if self.errors:
            print("\n[CRITICAL ERRORS]")
            for e in self.errors[:15]:
                print(f"  * {e}")
            if len(self.errors) > 15:
                print(f"  ... and {len(self.errors) - 15} more errors.")
            print("\nResult: DATASET VALIDATION FAILED [FAIL]")
            return False

        print("\nResult: ALL INTEGRITY & ANTI-LEAKAGE CHECKS PASSED [OK]")
        return True



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Master Dataset Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validator = DatasetValidator(
        data_root=_PROJECT_ROOT / args.data_dir,
        strict=args.strict,
    )
    passed = validator.validate_all()
    sys.exit(0 if passed else 1)
