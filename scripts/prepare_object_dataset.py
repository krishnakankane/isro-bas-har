"""ISRO BAS HAR — Object Dataset Staging & Annotation Preparation Utility.

Prepares raw extracted images for YOLOv8/YOLO11 annotation, manages train/val/test splits,
and generates empty label files or assisted pre-annotation files for human verification.

DISCLAIMER: Assisted pre-annotations generated with pretrained models are strictly
PROVISIONAL SCAFFOLDS and must be manually verified and corrected by human annotators.
They are NEVER committed directly as final ground truth.

Usage
-----
# Prepare empty annotation templates for staging frames:
    python -m scripts.prepare_object_dataset --staging_dir data/object_detection/images/staging/session01 \
        --split train

# Distribute multiple sessions into subject-separated train/val/test splits:
    python -m scripts.prepare_object_dataset --distribute \
        --train_actors ACTOR_01 ACTOR_02 ACTOR_03 ACTOR_04 \
        --val_actors ACTOR_05 \
        --test_actors ACTOR_06
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CLASS_NAMES_MAP: Dict[int, str] = {
    0: "micropipette",
    1: "pipette_tip_box",
    2: "sample_vial",
    3: "well_plate",
    4: "centrifuge",
    5: "waste_bin",
    6: "chamber_door_latch",
}


def prepare_staging_directory(
    staging_dir: Path,
    target_split: str = "train",
    generate_templates: bool = True,
    base_data_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Copy staging images to target dataset split and create empty .txt label files.

    Args:
        staging_dir: Folder containing extracted JPEG images.
        target_split: 'train', 'val', or 'test'.
        generate_templates: If True, creates empty `<image_name>.txt` label files.
        base_data_dir: Root dataset folder (defaults to data/object_detection).

    Returns:
        Summary dictionary with copied files and paths.
    """
    staging_dir = Path(staging_dir)
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_dir}")

    target_split = target_split.lower()
    if target_split not in ("train", "val", "test"):
        raise ValueError(f"Invalid target split: {target_split}. Must be train, val, or test.")

    base_dir = base_data_dir or (_PROJECT_ROOT / "data" / "object_detection")
    img_dest_dir = base_dir / "images" / target_split
    lbl_dest_dir = base_dir / "labels" / target_split
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(staging_dir.glob("*.jpg")) + list(staging_dir.glob("*.png")))
    # Exclude contact sheets from dataset images
    image_files = [f for f in image_files if "contact_sheet" not in f.name]

    print(f"[DatasetPrep] Found {len(image_files)} image(s) in {staging_dir}")
    print(f"[DatasetPrep] Target split: {target_split.upper()}")
    print(f"[DatasetPrep] Images destination: {img_dest_dir}")
    print(f"[DatasetPrep] Labels destination: {lbl_dest_dir}")

    copied_images: List[str] = []
    created_labels: List[str] = []

    for img_path in image_files:
        dest_img = img_dest_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        copied_images.append(str(dest_img))

        if generate_templates:
            dest_lbl = lbl_dest_dir / f"{img_path.stem}.txt"
            if not dest_lbl.exists():
                # Write empty file ready for LabelImg / CVAT export
                dest_lbl.write_text("", encoding="utf-8")
                created_labels.append(str(dest_lbl))

    summary = {
        "source_staging_dir": str(staging_dir),
        "target_split": target_split,
        "images_copied": len(copied_images),
        "label_templates_created": len(created_labels),
        "classes": CLASS_NAMES_MAP,
    }

    print("\n" + "=" * 62)
    print("OBJECT DATASET STAGING COMPLETE")
    print("=" * 62)
    print(f"  Target Split     : {target_split}")
    print(f"  Images Copied    : {len(copied_images)}")
    print(f"  Label Templates  : {len(created_labels)}")
    print("=" * 62)

    return summary


def distribute_by_actor(
    raw_metadata_dir: Path,
    staging_root: Path,
    train_actors: List[str],
    val_actors: List[str],
    test_actors: List[str],
    base_data_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Distribute all staging sessions into train/val/test splits without subject leakage.

    Args:
        raw_metadata_dir: Path to data/raw/metadata/ containing session JSONs.
        staging_root: Path to data/object_detection/images/staging/.
        train_actors: List of actor IDs for training.
        val_actors: List of actor IDs for validation.
        test_actors: List of actor IDs for testing.
        base_data_dir: Target object detection directory.

    Returns:
        Distribution summary report.
    """
    raw_metadata_dir = Path(raw_metadata_dir)
    staging_root = Path(staging_root)

    # Check for actor overlap (leakage guard)
    s_train = set(train_actors)
    s_val = set(val_actors)
    s_test = set(test_actors)

    if s_train.intersection(s_val):
        raise ValueError(f"Actor leakage detected between Train and Val: {s_train.intersection(s_val)}")
    if s_train.intersection(s_test):
        raise ValueError(f"Actor leakage detected between Train and Test: {s_train.intersection(s_test)}")
    if s_val.intersection(s_test):
        raise ValueError(f"Actor leakage detected between Val and Test: {s_val.intersection(s_test)}")

    actor_split_map: Dict[str, str] = {}
    for a in train_actors:
        actor_split_map[a] = "train"
    for a in val_actors:
        actor_split_map[a] = "val"
    for a in test_actors:
        actor_split_map[a] = "test"

    meta_files = list(raw_metadata_dir.glob("*.json"))
    report: Dict[str, List[str]] = {"train": [], "val": [], "test": [], "unassigned": []}

    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            actor = meta.get("actor_id", "")
            session_id = meta.get("session_id", mf.stem)
            split = actor_split_map.get(actor)

            session_staging = staging_root / session_id
            if session_staging.exists() and split:
                prepare_staging_directory(
                    staging_dir=session_staging,
                    target_split=split,
                    base_data_dir=base_data_dir,
                )
                report[split].append(session_id)
            else:
                report["unassigned"].append(session_id)
        except Exception as exc:
            print(f"[DatasetPrep] Error reading {mf.name}: {exc}")

    return {
        "actor_split_map": actor_split_map,
        "sessions_distributed": report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Object Dataset Preparation & Staging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--staging_dir", default="", help="Path to extracted session image folder")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Target split")
    parser.add_argument("--distribute", action="store_true", help="Run automated subject-wise split distribution")
    parser.add_argument("--train_actors", nargs="+", default=["ACTOR_01", "ACTOR_02", "ACTOR_03", "ACTOR_04"])
    parser.add_argument("--val_actors", nargs="+", default=["ACTOR_05"])
    parser.add_argument("--test_actors", nargs="+", default=["ACTOR_06"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.distribute:
        distribute_by_actor(
            raw_metadata_dir=_PROJECT_ROOT / "data" / "raw" / "metadata",
            staging_root=_PROJECT_ROOT / "data" / "object_detection" / "images" / "staging",
            train_actors=args.train_actors,
            val_actors=args.val_actors,
            test_actors=args.test_actors,
        )
    elif args.staging_dir:
        prepare_staging_directory(
            staging_dir=Path(args.staging_dir),
            target_split=args.split,
        )
    else:
        print("[DatasetPrep] Specify --staging_dir <path> or --distribute. Use --help for guidance.")
