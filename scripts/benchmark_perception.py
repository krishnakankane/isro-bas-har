"""Perception Pipeline Hardware Benchmark for ISRO BAS HAR.

Measures real inference latency for MediaPipe Pose Tasks, Hand Tasks, and YOLOv8 on the current hardware.
"""

import time
import psutil
import numpy as np
import torch
from pathlib import Path

from src.ingestion.camera_feed import CameraFeed
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.hand_tracker import HandTracker


def run_benchmark(iterations: int = 50, resolution=(1280, 720)):
    w, h = resolution
    print("=" * 70)
    print("ISRO BAS HAR -- PERCEPTION HARDWARE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Target Resolution: {w}x{h} | Iterations: {iterations}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:        {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability:   {torch.cuda.get_device_capability(0)}")

    feed = CameraFeed(source="synthetic", frame_width=w, frame_height=h)
    feed.start()
    time.sleep(0.3)
    success, frame, _ = feed.read_frame()
    if not success or frame is None:
        raise RuntimeError("Failed to capture frame from synthetic feed.")

    # Initialize perception components
    yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ObjectDetector(model_name_or_path="yolov8n.pt", device=yolo_device)
    pose_lite = PoseEstimator(model_path="models/mediapipe/pose_landmarker_lite.task")
    pose_full = PoseEstimator(model_path="models/mediapipe/pose_landmarker_full.task")
    hand = HandTracker(model_path="models/mediapipe/hand_landmarker.task")

    # Warm-up runs
    print("\nExecuting warm-up passes (10 iterations)...")
    for _ in range(10):
        detector.detect(frame)
        pose_lite.estimate(frame)
        pose_full.estimate(frame)
        hand.track(frame)

    detector_times = []
    pose_lite_times = []
    pose_full_times = []
    hand_times = []
    total_pipeline_times = []

    cpu_usages = []
    print(f"Executing {iterations} timed iterations...")

    for i in range(iterations):
        cpu_usages.append(psutil.cpu_percent(interval=None))
        t_pipe_start = time.perf_counter()

        # 1. YOLOv8
        t0 = time.perf_counter()
        _ = detector.detect(frame)
        detector_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. MediaPipe Pose Lite
        t0 = time.perf_counter()
        _ = pose_lite.estimate(frame)
        pose_lite_times.append((time.perf_counter() - t0) * 1000.0)

        # 3. MediaPipe Hands
        t0 = time.perf_counter()
        _ = hand.track(frame)
        hand_times.append((time.perf_counter() - t0) * 1000.0)

        total_pipeline_times.append((time.perf_counter() - t_pipe_start) * 1000.0)

        # 4. MediaPipe Pose Full (isolated comparison)
        t0 = time.perf_counter()
        _ = pose_full.estimate(frame)
        pose_full_times.append((time.perf_counter() - t0) * 1000.0)

    feed.stop()
    pose_lite.close()
    pose_full.close()
    hand.close()

    def calc_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    s_det = calc_stats(detector_times)
    s_pl = calc_stats(pose_lite_times)
    s_pf = calc_stats(pose_full_times)
    s_h = calc_stats(hand_times)
    s_pipe = calc_stats(total_pipeline_times)

    print("\n" + "=" * 70)
    print("REAL MEASURED BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n[1] Object Detection (YOLOv8n on {detector.device.upper()}):")
    print(f"    Mean Latency: {s_det['mean']:6.2f} ms (± {s_det['std']:.2f} ms)")
    print(f"    P50 / P95:    {s_det['p50']:6.2f} ms / {s_det['p95']:.2f} ms")
    print(f"    Min / Max:    {s_det['min']:6.2f} ms / {s_det['max']:.2f} ms")

    print(f"\n[2] Human Pose Estimation - MediaPipe Tasks PoseLandmarker Lite (5.5 MB):")
    print(f"    Mean Latency: {s_pl['mean']:6.2f} ms (± {s_pl['std']:.2f} ms)")
    print(f"    P50 / P95:    {s_pl['p50']:6.2f} ms / {s_pl['p95']:.2f} ms")
    print(f"    Min / Max:    {s_pl['min']:6.2f} ms / {s_pl['max']:.2f} ms")

    print(f"\n[3] Human Pose Estimation - MediaPipe Tasks PoseLandmarker Full (9.0 MB):")
    print(f"    Mean Latency: {s_pf['mean']:6.2f} ms (± {s_pf['std']:.2f} ms)")
    print(f"    P50 / P95:    {s_pf['p50']:6.2f} ms / {s_pf['p95']:.2f} ms")
    print(f"    Min / Max:    {s_pf['min']:6.2f} ms / {s_pf['max']:.2f} ms")

    print(f"\n[4] Dual Hand Tracking - MediaPipe Tasks HandLandmarker (7.5 MB):")
    print(f"    Mean Latency: {s_h['mean']:6.2f} ms (± {s_h['std']:.2f} ms)")
    print(f"    P50 / P95:    {s_h['p50']:6.2f} ms / {s_h['p95']:.2f} ms")
    print(f"    Min / Max:    {s_h['min']:6.2f} ms / {s_h['max']:.2f} ms")

    print(f"\n[5] Synchronous End-to-End Perception Pipeline (YOLO + Pose Lite + Dual Hands):")
    print(f"    Total Mean Latency:  {s_pipe['mean']:6.2f} ms (± {s_pipe['std']:.2f} ms)")
    print(f"    Total P95 Latency:   {s_pipe['p95']:6.2f} ms")
    print(f"    Effective Throughput: {1000.0 / s_pipe['mean']:6.1f} FPS")

    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        gpu_res_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
        print(f"\n[6] GPU Memory (NVIDIA GTX 1650):")
        print(f"    Allocated VRAM: {gpu_mem_mb:.2f} MB")
        print(f"    Reserved VRAM:  {gpu_res_mb:.2f} MB")

    mean_cpu = np.mean([u for u in cpu_usages if u > 0] or [psutil.cpu_percent()])
    print(f"\n[7] CPU Utilization:")
    print(f"    Observed Process/System CPU: ~{mean_cpu:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(iterations=50, resolution=(1280, 720))
