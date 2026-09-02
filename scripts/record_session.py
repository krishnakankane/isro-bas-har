"""ISRO BAS HAR — Experiment Session Video & Metadata Recording Utility.

Records high-fidelity experiment sessions from a local webcam or connected camera feed,
embeds local timestamp diagnostics, and saves raw MP4 video alongside structured JSON metadata.

100% Offline — No cloud uploads or external dependencies.

Usage
-----
# Interactive Recording (Webcam 0):
    python -m scripts.record_session --source 0 --actor ACTOR_01 --session SES_001

# With full metadata flags:
    python -m scripts.record_session --source 0 --actor ACTOR_01 --session SES_001 \
        --lighting DIFFUSE_LED --handedness RIGHT --protocol SYNTHETIC_DEMO_BIO_EXP_01 \
        --camera FRONT_45DEG --width 1280 --height 720 --fps 30

# Headless / Synthetic recording (for verification):
    python -m scripts.record_session --source synthetic --actor TEST_ACTOR --session TEST_01 \
        --auto_record_frames 90 --no_gui

Controls (Interactive Window)
-----------------------------
    SPACE / R  — Start / Stop recording segment
    S          — Save instant snapshot (JPEG)
    Q / ESC    — Save metadata and quit
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.camera_feed import CameraFeed


def record_session(
    source: str = "0",
    actor_id: str = "ACTOR_01",
    session_id: str = "",
    camera_id: str = "CAM_FRONT_45DEG",
    lighting: str = "DIFFUSE_LED",
    handedness: str = "RIGHT",
    protocol_id: str = "HOME_DEMO_PROTOCOL_01",
    notes: str = "",
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    output_dir: Optional[Path] = None,
    show_window: bool = True,
    auto_record_frames: int = 0,
) -> dict:
    """Record an experiment video session and write structured metadata.

    Args:
        source: Camera index ('0') or 'synthetic'.
        actor_id: Operator identifier (e.g. 'ACTOR_01').
        session_id: Session identifier (auto-generated timestamp if empty).
        camera_id: Camera viewpoint identifier.
        lighting: Lighting condition string.
        handedness: 'RIGHT', 'LEFT', or 'BIMANUAL'.
        protocol_id: Protocol specification ID.
        notes: User text notes.
        width: Video frame width.
        height: Video frame height.
        target_fps: Target recording FPS.
        output_dir: Base data directory (defaults to project data/raw).
        show_window: Whether to display OpenCV GUI window.
        auto_record_frames: If >0, automatically records N frames then exits.

    Returns:
        Metadata dict for the recorded session.
    """
    if not session_id:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{actor_id}_{timestamp_str}"

    base_dir = output_dir or (_PROJECT_ROOT / "data" / "raw")
    video_dir = base_dir / "videos"
    meta_dir = base_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{session_id}.mp4"
    meta_path = meta_dir / f"{session_id}.json"

    cam_source: object = int(source) if source.isdigit() else source

    print(f"[Recorder] Starting feed from source: {source!r}")
    print(f"[Recorder] Target resolution: {width}x{height} @ {target_fps} FPS")
    print(f"[Recorder] Output video: {video_path}")

    feed = CameraFeed(
        source=cam_source,
        target_fps=target_fps,
        loop_video=True,
        frame_width=width,
        frame_height=height,
    )
    feed.start()
    time.sleep(0.3)  # Warm up reader thread

    # Setup OpenCV VideoWriter with robust codec fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        float(target_fps),
        (width, height),
    )

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            float(target_fps),
            (width, height),
        )

    is_recording = (auto_record_frames > 0)
    recorded_frames = 0
    total_preview_frames = 0
    start_time: Optional[float] = None
    first_frame_recorded_time: Optional[float] = None

    window_name = f"ISRO BAS HAR — Session Recorder [{session_id}]"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    print("[Recorder] Ready. Press SPACE/R to toggle recording, Q to finish.")

    try:
        while True:
            success, frame, ts = feed.read_frame(timeout_sec=0.5)
            if not success or frame is None:
                time.sleep(0.01)
                continue

            total_preview_frames += 1
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # Handle interactive keys
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[Recorder] Stop requested by user.")
                    break
                elif key in (ord(" "), ord("r"), ord("R")):
                    is_recording = not is_recording
                    if is_recording and start_time is None:
                        start_time = time.time()
                    print(f"[Recorder] Recording state changed: {'RECORDING' if is_recording else 'PAUSED'}")
                elif key in (ord("s"), ord("S")):
                    snap_path = base_dir / f"snapshot_{session_id}_{total_preview_frames:05d}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    print(f"[Recorder] Snapshot saved → {snap_path}")

            # Write frame if actively recording
            if is_recording:
                if first_frame_recorded_time is None:
                    first_frame_recorded_time = time.time()
                writer.write(frame)
                recorded_frames += 1

                if auto_record_frames > 0 and recorded_frames >= auto_record_frames:
                    print(f"[Recorder] Auto-record limit reached ({recorded_frames} frames).")
                    break

            # Render preview HUD
            if show_window:
                preview = frame.copy()
                _draw_recorder_hud(
                    preview,
                    session_id=session_id,
                    actor_id=actor_id,
                    camera_id=camera_id,
                    is_recording=is_recording,
                    recorded_frames=recorded_frames,
                    fps=target_fps,
                )
                cv2.imshow(window_name, preview)

    except KeyboardInterrupt:
        print("\n[Recorder] Interrupted by Ctrl+C.")
    finally:
        feed.stop()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    end_time = time.time()
    effective_duration = (end_time - first_frame_recorded_time) if first_frame_recorded_time else 0.0

    # Build and write metadata manifest
    metadata = {
        "session_id": session_id,
        "actor_id": actor_id,
        "camera_id": camera_id,
        "lighting_condition": lighting,
        "handedness": handedness,
        "protocol_id": protocol_id,
        "notes": notes,
        "video_filename": f"{session_id}.mp4",
        "video_path": str(video_path),
        "recorded_frames": recorded_frames,
        "resolution": {"width": width, "height": height},
        "target_fps": target_fps,
        "effective_duration_sec": round(effective_duration, 3),
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "file_size_bytes": os.path.getsize(video_path) if video_path.exists() else 0,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 62)
    print("SESSION RECORDING COMPLETE")
    print("=" * 62)
    print(f"  Session ID       : {session_id}")
    print(f"  Actor ID         : {actor_id}")
    print(f"  Frames Written   : {recorded_frames}")
    print(f"  Video Output     : {video_path}")
    print(f"  Metadata Output  : {meta_path}")
    print(f"  File Size        : {metadata['file_size_bytes'] / (1024*1024):.2f} MB")
    print("=" * 62)

    return metadata


def _draw_recorder_hud(
    canvas: np.ndarray,
    session_id: str,
    actor_id: str,
    camera_id: str,
    is_recording: bool,
    recorded_frames: int,
    fps: int,
) -> None:
    """Draw recorder status badge and telemetry banner."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status Banner Top-Left
    banner_w, banner_h = 380, 85
    sub = canvas[10:10 + banner_h, 10:10 + banner_w]
    if sub.shape[0] == banner_h and sub.shape[1] == banner_w:
        dark = np.zeros(sub.shape, dtype=np.uint8)
        cv2.addWeighted(sub, 0.3, dark, 0.7, 0, sub)
        canvas[10:10 + banner_h, 10:10 + banner_w] = sub

    cv2.rectangle(canvas, (10, 10), (10 + banner_w, 10 + banner_h), (0, 200, 255), 1)

    # Flashing REC Indicator
    if is_recording:
        blink = int(time.time() * 2) % 2 == 0
        rec_color = (0, 0, 255) if blink else (0, 0, 150)
        cv2.circle(canvas, (28, 28), 7, rec_color, -1)
        cv2.putText(canvas, "REC", (42, 33), font, 0.55, (0, 0, 255), 2)
    else:
        cv2.circle(canvas, (28, 28), 7, (120, 120, 120), -1)
        cv2.putText(canvas, "PAUSED (Press SPACE to record)", (42, 33), font, 0.42, (180, 180, 180), 1)

    duration_sec = recorded_frames / float(fps) if fps > 0 else 0.0
    cv2.putText(canvas, f"Session: {session_id} | Actor: {actor_id}", (18, 54), font, 0.40, (255, 255, 255), 1)
    cv2.putText(canvas, f"Cam: {camera_id} | Frames: {recorded_frames:05d} ({duration_sec:.1f}s)", (18, 74), font, 0.40, (0, 255, 150), 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Session Video & Metadata Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default="0", help="Camera index ('0') or 'synthetic'")
    parser.add_argument("--actor", default="ACTOR_01", help="Actor/Operator ID (e.g. ACTOR_01)")
    parser.add_argument("--session", default="", help="Session ID (auto-generated if omitted)")
    parser.add_argument("--camera", default="CAM_FRONT_45DEG", help="Camera angle ID")
    parser.add_argument("--lighting", default="DIFFUSE_LED", help="Lighting condition")
    parser.add_argument("--handedness", default="RIGHT", choices=["RIGHT", "LEFT", "BIMANUAL"])
    parser.add_argument("--protocol", default="HOME_DEMO_PROTOCOL_01", help="Protocol ID")
    parser.add_argument("--notes", default="", help="Optional session notes")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--no_gui", action="store_true", help="Run without opening OpenCV GUI window")
    parser.add_argument("--auto_record_frames", type=int, default=0, help="Auto-record N frames and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    record_session(
        source=args.source,
        actor_id=args.actor,
        session_id=args.session,
        camera_id=args.camera,
        lighting=args.lighting,
        handedness=args.handedness,
        protocol_id=args.protocol,
        notes=args.notes,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        show_window=not args.no_gui,
        auto_record_frames=args.auto_record_frames,
    )
