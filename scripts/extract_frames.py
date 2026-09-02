"""ISRO BAS HAR — Intelligent Video Frame Extractor.

Extracts representative, non-duplicate frames from experiment videos for
YOLO object detection annotation and dataset preparation.

Features
--------
- Configurable frame-stride interval or visual difference thresholding
- Preserves exact source frame indices and millisecond timestamps
- Generates optional thumbnail contact sheets for quick visual QA
- Deduplication filter: skips nearly identical static frames
- 100% Offline operation

Usage
-----
# Extract every 15th frame (2 fps from 30 fps video):
    python -m scripts.extract_frames --video data/raw/videos/session01.mp4 --interval 15

# Extract with visual motion difference threshold:
    python -m scripts.extract_frames --video data/raw/videos/session01.mp4 --diff_thresh 12.0

# Generate a 4x4 preview contact sheet:
    python -m scripts.extract_frames --video data/raw/videos/session01.mp4 --contact_sheet
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def extract_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
    interval_frames: int = 15,
    min_diff_threshold: float = 8.0,
    max_frames: int = 0,
    generate_contact_sheet: bool = True,
    contact_sheet_cols: int = 4,
) -> Dict[str, object]:
    """Extract filtered frames from a video file.

    Args:
        video_path: Path to input MP4/AVI video.
        output_dir: Destination folder for extracted JPEGs (default: data/object_detection/images/staging/<video_stem>/).
        interval_frames: Sample every N-th frame.
        min_diff_threshold: Minimum mean pixel difference between consecutive saved frames.
        max_frames: Max frames to extract (0 = all matching).
        generate_contact_sheet: If True, writes a summary grid image.
        contact_sheet_cols: Number of columns in contact sheet.

    Returns:
        Extraction summary dictionary.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = video_path.stem
    target_dir = output_dir or (_PROJECT_ROOT / "data" / "object_detection" / "images" / "staging" / video_stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file with OpenCV: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[FrameExtractor] Video: {video_path.name} | Total Frames: {total_video_frames} | FPS: {video_fps:.1f}")
    print(f"[FrameExtractor] Stride interval: {interval_frames} | Min diff threshold: {min_diff_threshold}")
    print(f"[FrameExtractor] Destination: {target_dir}")

    extracted_records: List[Dict[str, object]] = []
    saved_thumbnails: List[np.ndarray] = []
    last_saved_gray: Optional[np.ndarray] = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_sec = frame_idx / video_fps

        # Check stride interval
        if (frame_idx % interval_frames) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            should_save = False

            if last_saved_gray is None:
                should_save = True
            else:
                # Calculate mean absolute difference to filter stationary duplicate frames
                diff = float(np.mean(cv2.absdiff(gray, last_saved_gray)))
                if diff >= min_diff_threshold:
                    should_save = True

            if should_save:
                filename = f"{video_stem}_f{frame_idx:06d}_t{timestamp_sec:.2f}s.jpg"
                save_path = target_dir / filename
                cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                last_saved_gray = gray
                rec = {
                    "filename": filename,
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "resolution": [width, height],
                    "filepath": str(save_path),
                }
                extracted_records.append(rec)

                if generate_contact_sheet:
                    thumb = cv2.resize(frame, (240, int(240 * height / width)))
                    # Overlay timestamp badge on thumbnail
                    cv2.putText(
                        thumb,
                        f"F{frame_idx} | {timestamp_sec:.1f}s",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )
                    saved_thumbnails.append(thumb)

                if max_frames > 0 and len(extracted_records) >= max_frames:
                    print(f"[FrameExtractor] Reached max_frames limit ({max_frames}). Stopping.")
                    break

        frame_idx += 1

    cap.release()

    # Save metadata manifest
    manifest_path = target_dir / "extraction_manifest.json"
    summary = {
        "video_path": str(video_path),
        "total_source_frames": total_video_frames,
        "extracted_frames_count": len(extracted_records),
        "interval_frames": interval_frames,
        "min_diff_threshold": min_diff_threshold,
        "frames": extracted_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build contact sheet grid image
    contact_sheet_path: Optional[Path] = None
    if generate_contact_sheet and saved_thumbnails:
        contact_sheet_path = target_dir / f"{video_stem}_contact_sheet.jpg"
        _build_contact_sheet(saved_thumbnails, contact_sheet_cols, contact_sheet_path)
        print(f"[FrameExtractor] Contact sheet written: {contact_sheet_path.name}")

    print("\n" + "=" * 62)
    print("FRAME EXTRACTION COMPLETE")
    print("=" * 62)
    print(f"  Source Video     : {video_path.name}")
    print(f"  Total Video Frames: {total_video_frames}")
    print(f"  Frames Extracted : {len(extracted_records)}")
    print(f"  Extraction Dir   : {target_dir}")
    print("=" * 62)

    summary["contact_sheet_path"] = str(contact_sheet_path) if contact_sheet_path else None
    return summary


def _build_contact_sheet(
    thumbnails: List[np.ndarray],
    cols: int,
    output_path: Path,
) -> None:
    """Compose a multi-image grid contact sheet."""
    if not thumbnails:
        return

    th, tw = thumbnails[0].shape[:2]
    n = len(thumbnails)
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbnails):
        r = idx // cols
        c = idx % cols
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb

    cv2.imwrite(str(output_path), grid)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Intelligent Video Frame Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", default=None, help="Destination folder (optional)")
    parser.add_argument("--interval", type=int, default=15, help="Frame stride interval (default: 15)")
    parser.add_argument("--diff_thresh", type=float, default=8.0, help="Pixel difference threshold (default: 8.0)")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum frames to extract (0 = all)")
    parser.add_argument("--no_contact_sheet", action="store_true", help="Disable contact sheet creation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        interval_frames=args.interval,
        min_diff_threshold=args.diff_thresh,
        max_frames=args.max_frames,
        generate_contact_sheet=not args.no_contact_sheet,
    )
