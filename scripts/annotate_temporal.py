"""ISRO BAS HAR — Interactive Temporal Action Interval Annotation Tool.

Allows researchers and student annotators to review recorded experiment videos locally,
set frame-accurate start/end boundaries, tag action classes, and export clean JSON manifests.

100% Offline — No external cloud or web services.

Controls (Interactive Window)
-----------------------------
    SPACE       — Play / Pause
    D / RIGHT   — Step Forward 1 frame
    A / LEFT    — Step Backward 1 frame
    F / UP      — Jump Forward 30 frames (1 second)
    S / DOWN    — Jump Backward 30 frames (1 second)
    [  or  I    — Set Action Interval START boundary (t_start)
    ]  or  O    — Set Action Interval END boundary (t_end)
    0 – 9       — Select Action Class (0=IDLE, 1=SANITIZE, 2=HOLD_BOTTLE, 3=PLACE_BOTTLE, 4=HOLD_BOX, 5=OPEN_BOX, 6=PICK_OBJECT, 7=TRANSFER_OBJECT, 8=RETURN_OBJECT, 9=CLOSE_BOX)
    ENTER / C   — Commit current interval to manifest
    BACKSPACE   — Remove last committed interval
    W / S       — Save manifest JSON to disk
    Q / ESC     — Save and quit

Usage
-----
# Annotate an experiment video:
    python -m scripts.annotate_temporal --video data/raw/videos/session01.mp4 --actor ACTOR_01

# Load and edit an existing action manifest:
    python -m scripts.annotate_temporal --video data/raw/videos/session01.mp4 \
        --manifest data/temporal_har/manifests/session01_actions.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ACTION_TAXONOMY: Dict[int, str] = {
    0: "ACTION_IDLE",
    1: "ACTION_SANITIZE",
    2: "ACTION_HOLD_BOTTLE",
    3: "ACTION_PLACE_BOTTLE",
    4: "ACTION_HOLD_BOX",
    5: "ACTION_OPEN_BOX",
    6: "ACTION_PICK_OBJECT",
    7: "ACTION_TRANSFER_OBJECT",
    8: "ACTION_RETURN_OBJECT",
    9: "ACTION_CLOSE_BOX",
}

KEY_TO_ACTION: Dict[int, int] = {
    ord("0"): 0, ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4,
    ord("5"): 5, ord("6"): 6, ord("7"): 7, ord("8"): 8, ord("9"): 9,
}


def run_temporal_annotator(
    video_path: Path,
    actor_id: str = "ACTOR_01",
    session_id: str = "",
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show_window: bool = True,
    auto_save_and_exit: bool = False,
    mock_segments: Optional[List[Dict]] = None,
) -> Dict[str, object]:
    """Launch interactive video annotator and output action interval manifest.

    Args:
        video_path: Path to MP4 video.
        actor_id: Operator ID.
        session_id: Session ID (inferred from video stem if empty).
        manifest_path: Optional existing manifest to resume editing.
        output_dir: Destination folder for manifests (default: data/temporal_har/manifests/).
        show_window: Whether to open OpenCV display window.
        auto_save_and_exit: Test flag to save immediately and exit.
        mock_segments: Initial segment list for testing.

    Returns:
        Action manifest dictionary.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = video_path.stem
    session_id = session_id or video_stem

    out_dir = output_dir or (_PROJECT_ROOT / "data" / "temporal_har" / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest_path = manifest_path or (out_dir / f"{video_stem}_actions.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print(f"[Annotator] Video: {video_path.name} | {total_frames} frames ({duration_sec:.1f}s @ {fps:.1f} FPS)")
    print(f"[Annotator] Manifest output: {save_manifest_path}")

    # Load existing manifest if present
    segments: List[Dict[str, object]] = []
    if mock_segments:
        segments = list(mock_segments)
    elif save_manifest_path.exists():
        try:
            with open(save_manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                segments = data.get("segments", [])
                print(f"[Annotator] Loaded {len(segments)} existing segment(s) from {save_manifest_path.name}")
        except Exception as exc:
            print(f"[Annotator] Warning: could not parse existing manifest: {exc}")

    if auto_save_and_exit:
        cap.release()
        return _write_manifest(
            manifest_path=save_manifest_path,
            video_id=video_stem,
            actor_id=actor_id,
            session_id=session_id,
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_sec=duration_sec,
            resolution=(width, height),
            segments=segments,
        )

    current_frame_idx = 0
    is_playing = False
    mark_start_frame: Optional[int] = None
    mark_end_frame: Optional[int] = None
    selected_action_id: int = 0  # Default: ACTION_IDLE
    status_msg: str = "Ready. Press [ to set start, ] to set end."

    window_name = f"ISRO BAS HAR — Action Annotator [{video_stem}]"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                current_frame_idx = max(0, total_frames - 1)
                is_playing = False
                continue

            current_time = current_frame_idx / fps

            # Draw annotation HUD overlay
            canvas = frame.copy()
            _draw_annotator_ui(
                canvas,
                video_id=video_stem,
                actor_id=actor_id,
                current_frame=current_frame_idx,
                total_frames=total_frames,
                current_time=current_time,
                duration_sec=duration_sec,
                is_playing=is_playing,
                mark_start=mark_start_frame,
                mark_end=mark_end_frame,
                selected_action_id=selected_action_id,
                segments_count=len(segments),
                status_msg=status_msg,
                fps=fps,
            )

            if show_window:
                cv2.imshow(window_name, canvas)
                wait_ms = int(1000.0 / fps) if is_playing else 30
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in (ord("q"), 27):
                    print("[Annotator] Saving manifest and exiting...")
                    break
                elif key == ord(" "):
                    is_playing = not is_playing
                    status_msg = "Playing" if is_playing else "Paused"
                elif key in (ord("d"), 83):  # D or Right Arrow
                    current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
                    is_playing = False
                elif key in (ord("a"), 81):  # A or Left Arrow
                    current_frame_idx = max(0, current_frame_idx - 1)
                    is_playing = False
                elif key in (ord("f"), 82):  # F or Up Arrow (Jump 1s)
                    current_frame_idx = min(total_frames - 1, current_frame_idx + int(fps))
                    is_playing = False
                elif key in (ord("s"), 84):  # S or Down Arrow (Jump back 1s)
                    current_frame_idx = max(0, current_frame_idx - int(fps))
                    is_playing = False
                elif key in (ord("["), ord("i"), ord("I")):
                    mark_start_frame = current_frame_idx
                    status_msg = f"Start set: F{current_frame_idx} ({current_time:.2f}s)"
                elif key in (ord("]"), ord("o"), ord("O")):
                    mark_end_frame = current_frame_idx
                    status_msg = f"End set: F{current_frame_idx} ({current_time:.2f}s)"
                elif key in KEY_TO_ACTION:
                    selected_action_id = KEY_TO_ACTION[key]
                    status_msg = f"Selected: {ACTION_TAXONOMY[selected_action_id]}"
                elif key in (13, ord("c"), ord("C")):  # ENTER or C
                    if mark_start_frame is not None and mark_end_frame is not None:
                        sf, ef = min(mark_start_frame, mark_end_frame), max(mark_start_frame, mark_end_frame)
                        seg = {
                            "segment_id": len(segments) + 1,
                            "action_id": selected_action_id,
                            "action_name": ACTION_TAXONOMY[selected_action_id],
                            "start_frame": sf,
                            "end_frame": ef,
                            "start_time_sec": round(sf / fps, 3),
                            "end_time_sec": round(ef / fps, 3),
                            "duration_sec": round((ef - sf) / fps, 3),
                            "notes": "",
                        }
                        segments.append(seg)
                        status_msg = f"COMMITTED: {seg['action_name']} [{sf} -> {ef}]"
                        mark_start_frame = None
                        mark_end_frame = None
                    else:
                        status_msg = "Error: Set BOTH start [ and end ] marks first!"
                elif key in (8, ord("x"), ord("X")):  # Backspace
                    if segments:
                        removed = segments.pop()
                        status_msg = f"Removed: {removed['action_name']}"
                    else:
                        status_msg = "No segments to remove."
                elif key in (ord("w"), ord("W")):
                    _write_manifest(
                        manifest_path=save_manifest_path,
                        video_id=video_stem,
                        actor_id=actor_id,
                        session_id=session_id,
                        video_path=video_path,
                        total_frames=total_frames,
                        fps=fps,
                        duration_sec=duration_sec,
                        resolution=(width, height),
                        segments=segments,
                    )
                    status_msg = f"SAVED to {save_manifest_path.name}"

            if is_playing:
                current_frame_idx += 1
                if current_frame_idx >= total_frames:
                    current_frame_idx = total_frames - 1
                    is_playing = False

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    return _write_manifest(
        manifest_path=save_manifest_path,
        video_id=video_stem,
        actor_id=actor_id,
        session_id=session_id,
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
        resolution=(width, height),
        segments=segments,
    )


def _write_manifest(
    manifest_path: Path,
    video_id: str,
    actor_id: str,
    session_id: str,
    video_path: Path,
    total_frames: int,
    fps: float,
    duration_sec: float,
    resolution: Tuple[int, int],
    segments: List[Dict],
) -> Dict[str, object]:
    """Write action interval database to JSON file."""
    # Sort segments chronologically
    segments.sort(key=lambda s: s.get("start_frame", 0))
    for idx, s in enumerate(segments):
        s["segment_id"] = idx + 1

    manifest = {
        "manifest_version": "1.0.0",
        "video_id": video_id,
        "actor_id": actor_id,
        "session_id": session_id,
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 3),
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "segments_count": len(segments),
        "segments": segments,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Annotator] Manifest written with {len(segments)} segment(s) → {manifest_path}")
    return manifest


def _draw_annotator_ui(
    canvas: np.ndarray,
    video_id: str,
    actor_id: str,
    current_frame: int,
    total_frames: int,
    current_time: float,
    duration_sec: float,
    is_playing: bool,
    mark_start: Optional[int],
    mark_end: Optional[int],
    selected_action_id: int,
    segments_count: int,
    status_msg: str,
    fps: float,
) -> None:
    """Render the interactive timeline HUD on top of the frame."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bottom Control Console
    bar_h = 135
    bar_y = h - bar_h
    sub = canvas[bar_y:h, 0:w]
    dark = np.zeros(sub.shape, dtype=np.uint8)
    cv2.addWeighted(sub, 0.15, dark, 0.85, 0, sub)
    canvas[bar_y:h, 0:w] = sub

    cv2.line(canvas, (0, bar_y), (w, bar_y), (0, 200, 255), 2)

    # Timeline Scrubber Bar
    scrub_x1, scrub_x2 = 20, w - 20
    scrub_y = bar_y + 20
    cv2.line(canvas, (scrub_x1, scrub_y), (scrub_x2, scrub_y), (80, 80, 80), 4)

    progress = current_frame / float(max(1, total_frames - 1))
    playhead_x = int(scrub_x1 + progress * (scrub_x2 - scrub_x1))
    cv2.circle(canvas, (playhead_x, scrub_y), 7, (0, 255, 255), -1)

    # Start and End markers on scrubber
    if mark_start is not None:
        mx = int(scrub_x1 + (mark_start / max(1, total_frames - 1)) * (scrub_x2 - scrub_x1))
        cv2.circle(canvas, (mx, scrub_y), 5, (0, 255, 0), -1)
    if mark_end is not None:
        mx = int(scrub_x1 + (mark_end / max(1, total_frames - 1)) * (scrub_x2 - scrub_x1))
        cv2.circle(canvas, (mx, scrub_y), 5, (0, 0, 255), -1)

    # Information text lines
    action_name = ACTION_TAXONOMY.get(selected_action_id, "UNKNOWN")
    play_state = "▶ PLAYING" if is_playing else "❚❚ PAUSED"

    cv2.putText(canvas, f"{play_state}  |  Frame: {current_frame:05d}/{total_frames:05d} ({current_time:.2f}s / {duration_sec:.1f}s)", (20, bar_y + 48), font, 0.48, (255, 255, 255), 1)
    cv2.putText(canvas, f"Target Action: [{selected_action_id}] {action_name}", (20, bar_y + 72), font, 0.52, (0, 215, 255), 2)

    start_str = f"F{mark_start}" if mark_start is not None else "NONE"
    end_str = f"F{mark_end}" if mark_end is not None else "NONE"
    cv2.putText(canvas, f"Bounds: Start=[{start_str}]  End=[{end_str}]  |  Segments Committed: {segments_count}", (20, bar_y + 94), font, 0.44, (200, 200, 200), 1)
    cv2.putText(canvas, f"Status: {status_msg}", (20, bar_y + 118), font, 0.42, (0, 255, 120), 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Interactive Action Interval Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--actor", default="ACTOR_01", help="Operator ID")
    parser.add_argument("--session", default="", help="Session ID")
    parser.add_argument("--manifest", default=None, help="Existing manifest path to edit")
    parser.add_argument("--output_dir", default=None, help="Manifest output directory")
    parser.add_argument("--no_gui", action="store_true", help="Non-interactive headless save")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_temporal_annotator(
        video_path=Path(args.video),
        actor_id=args.actor,
        session_id=args.session,
        manifest_path=Path(args.manifest) if args.manifest else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        show_window=not args.no_gui,
        auto_save_and_exit=args.no_gui,
    )
