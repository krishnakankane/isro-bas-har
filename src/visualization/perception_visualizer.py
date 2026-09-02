"""High-Performance OpenCV Visualizer for Development Perception Testing.

Renders bounding boxes, skeletal wireframes, 21-landmark hand graphs,
HOI state overlay, and real-time HUD telemetry.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from ..core.models import (
    DetectedObject,
    PoseEstimationResult,
    HandTrackingResult,
    FramePerceptionResult,
    InteractionEvent,
    InteractionState,
)


# MediaPipe Skeletal Connection Pairs
POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left Arm
    (12, 14), (14, 16),  # Right Arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left Leg
    (24, 26), (26, 28),  # Right Leg
]

# MediaPipe Hand Landmark Finger Connection Pairs
HAND_CONNECTIONS = [
    # Palm / Base
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]

# HOI state → display colour (BGR)
_HOI_STATE_COLORS: Dict[InteractionState, Tuple[int, int, int]] = {
    InteractionState.NONE:         (80,  80,  80),
    InteractionState.UNKNOWN:      (120, 120, 120),
    InteractionState.APPROACHING:  (0,   215, 255),   # Gold
    InteractionState.GRASPING:     (0,   165, 255),   # Orange
    InteractionState.MANIPULATING: (0,   255, 50),    # Bright green
    InteractionState.RELEASED:     (180, 180, 255),   # Lavender
}


class PerceptionVisualizer:
    """OpenCV Overlay Renderer for bounding boxes, pose, hands, HOI, and telemetry HUD."""

    def __init__(self, show_hud: bool = True):
        self.show_hud = show_hud
        # Palette for object classes
        self.colors = [
            (0, 215, 255),    # Gold / Yellow
            (255, 105, 180),  # Pink
            (50, 205, 50),    # Lime Green
            (0, 140, 255),    # Orange
            (238, 130, 238),  # Violet
            (255, 69, 0),     # Red-Orange
            (0, 255, 127),    # Spring Green
            (30, 144, 255),   # Dodger Blue
        ]

    def render(
        self,
        frame: np.ndarray,
        objects: List[DetectedObject],
        pose: Optional[PoseEstimationResult],
        left_hand: Optional[HandTrackingResult],
        right_hand: Optional[HandTrackingResult],
        metrics: Optional[Dict[str, float]] = None,
        interactions: Optional[List[InteractionEvent]] = None,
    ) -> np.ndarray:
        """Render complete perception annotations on a copy of the frame.

        Args:
            frame:        (H, W, 3) BGR image.
            objects:      List of detected objects.
            pose:         Estimated body pose result.
            left_hand:    Left hand tracking result.
            right_hand:   Right hand tracking result.
            metrics:      Dictionary containing latency and FPS measurements.
            interactions: List of HOI events for this frame.

        Returns:
            Annotated BGR frame.
        """
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Render Object Bounding Boxes
        self._draw_objects(canvas, objects, w, h)

        # 2. Render Human Pose Skeleton
        if pose and pose.is_detected:
            self._draw_pose(canvas, pose, w, h)

        # 3. Render Hand Landmarks
        if left_hand and left_hand.is_detected:
            self._draw_hand(canvas, left_hand, w, h, color=(255, 255, 0), tag="Left")
        if right_hand and right_hand.is_detected:
            self._draw_hand(canvas, right_hand, w, h, color=(0, 255, 255), tag="Right")

        # 4. Render HOI Interaction State Overlay
        if interactions:
            self._draw_hoi_overlay(canvas, interactions, w, h)

        # 5. Render Diagnostic Telemetry HUD
        if self.show_hud and metrics:
            self._draw_hud(canvas, metrics, w, h)
            if "har_action" in metrics:
                self._draw_har_banner(canvas, metrics, w, h)

        return canvas

    def _draw_objects(self, canvas: np.ndarray, objects: List[DetectedObject], w: int, h: int) -> None:
        """Draw bounding boxes and class badges."""
        for obj in objects:
            color = self.colors[obj.class_id % len(self.colors)]
            x1, y1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
            x2, y2 = int(obj.bbox.xmax), int(obj.bbox.ymax)

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Box outline
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label badge
            label = f"{obj.class_name} {obj.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1
            badge_x2 = min(w - 1, x1 + tw + 6)

            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(canvas, label, (x1 + 3, badge_y2 - baseline - 2), font, font_scale, (0, 0, 0), thickness)

    def _draw_pose(self, canvas: np.ndarray, pose: PoseEstimationResult, w: int, h: int) -> None:
        """Draw 33-landmark skeletal graph."""
        keypoints = pose.landmarks
        if len(keypoints) < 29:
            return

        coords = {}
        for lm in keypoints:
            px = int(lm.x * w)
            py = int(lm.y * h)
            coords[lm.id] = (px, py, lm.visibility)

        # Draw bones
        for p1, p2 in POSE_CONNECTIONS:
            if p1 in coords and p2 in coords:
                x1, y1, v1 = coords[p1]
                x2, y2, v2 = coords[p2]
                if v1 > 0.4 and v2 > 0.4:
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw joints
        for lm_id, (px, py, vis) in coords.items():
            if vis > 0.4 and 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)
                cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1)

    def _draw_hand(self, canvas: np.ndarray, hand: HandTrackingResult, w: int, h: int, color: Tuple[int, int, int], tag: str) -> None:
        """Draw 21 hand landmarks and connection lines."""
        landmarks = hand.landmarks
        if len(landmarks) < 21:
            return

        coords = {}
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            coords[lm.id] = (px, py)

        # Draw finger links
        for p1, p2 in HAND_CONNECTIONS:
            if p1 in coords and p2 in coords:
                cv2.line(canvas, coords[p1], coords[p2], color, 1)

        # Draw hand joints
        for lm_id, (px, py) in coords.items():
            radius = 4 if lm_id in (4, 8, 12, 16, 20) else 2
            joint_col = (0, 255, 0) if lm_id in (4, 8) and hand.is_grasping else color
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), radius, joint_col, -1)

        # Wrist label
        wx, wy = coords[0]
        grasp_txt = " [GRASP]" if hand.is_grasping else ""
        cv2.putText(canvas, f"{tag}{grasp_txt}", (wx - 20, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_hoi_overlay(
        self,
        canvas: np.ndarray,
        interactions: List[InteractionEvent],
        w: int,
        h: int,
    ) -> None:
        """Draw HOI interaction state panel (bottom-left corner)."""
        if not interactions:
            return

        panel_w, row_h = 360, 22
        panel_h = 28 + row_h * len(interactions)
        x1 = 15
        y1 = h - panel_h - 15
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        # Semi-transparent background
        if 0 <= y1 < h and 0 <= x1 < w:
            sub = canvas[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if sub.size > 0:
                dark = np.zeros(sub.shape, dtype=np.uint8)
                blended = cv2.addWeighted(sub, 0.2, dark, 0.8, 0)
                canvas[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = blended

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 120), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "HOI INTERACTION STATE", (x1 + 8, y1 + 16), font, 0.42, (0, 200, 120), 1)

        for i, evt in enumerate(interactions):
            row_y = y1 + 26 + i * row_h
            color = _HOI_STATE_COLORS.get(evt.state, (200, 200, 200))
            state_name = evt.state.name
            dist_str = f"{evt.proximity_distance_px:.3f}"
            dwell_str = f"{evt.duration_frames}f"
            text = f"{evt.hand_side[:1]}H ► {evt.target_object_name[:12]}  [{state_name}]  d={dist_str} t={dwell_str}"
            cv2.putText(canvas, text, (x1 + 8, row_y + 14), font, 0.36, color, 1)

    def _draw_hud(self, canvas: np.ndarray, metrics: Dict[str, object], w: int, h: int) -> None:
        """Draw semi-transparent HUD telemetry box with latency and throughput metrics."""
        hud_w, hud_h = 360, 215
        x1, y1 = 15, 15
        x2, y2 = x1 + hud_w, y1 + hud_h

        # Semi-transparent dark background
        if 0 <= y1 < h and 0 <= x1 < w:
            sub_img = canvas[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if sub_img.shape[0] == hud_h and sub_img.shape[1] == hud_w:
                black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0, sub_img)
                canvas[y1:y2, x1:x2] = sub_img

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 1)

        # Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "ISRO BAS HAR -- PERCEPTION HUD", (x1 + 10, y1 + 20), font, 0.46, (0, 215, 255), 1)
        cv2.line(canvas, (x1 + 10, y1 + 26), (x2 - 10, y1 + 26), (80, 80, 80), 1)

        # Telemetry fields
        fps_act   = float(metrics.get("fps_actual", 0.0))
        det_ms    = float(metrics.get("detector_ms", 0.0))
        pose_ms   = float(metrics.get("pose_ms", 0.0))
        hand_ms   = float(metrics.get("hand_ms", 0.0))
        hoi_ms    = float(metrics.get("hoi_ms", 0.0))
        har_ms    = float(metrics.get("har_latency_ms", 0.0))
        total_ms  = float(metrics.get("total_latency_ms", 0.0))
        res_txt   = str(metrics.get("resolution", f"{w}x{h}"))
        dev_txt   = str(metrics.get("device", "N/A"))
        buf_fill  = int(metrics.get("buffer_fill", 0))
        buf_cap   = int(metrics.get("buffer_cap", 60))
        hoi_state = str(metrics.get("hoi_state", "—"))
        buf_status = "[FULL]" if buf_fill >= buf_cap else f"[{buf_fill}/{buf_cap}]"

        cv2.putText(canvas, f"Throughput: {fps_act:.1f} FPS | Res: {res_txt}", (x1 + 10, y1 + 44), font, 0.40, (255, 255, 255), 1)
        cv2.putText(canvas, f"Device: {dev_txt} | Pipeline: {total_ms:.1f} ms", (x1 + 10, y1 + 62), font, 0.40, (0, 255, 150), 1)
        cv2.putText(canvas, f"• Object Detection (YOLO):  {det_ms:.1f} ms", (x1 + 10, y1 + 82),  font, 0.36, (200, 200, 200), 1)
        cv2.putText(canvas, f"• Pose Estimation (MP):     {pose_ms:.1f} ms", (x1 + 10, y1 + 100), font, 0.36, (200, 200, 200), 1)
        cv2.putText(canvas, f"• Hand Tracking (MP):       {hand_ms:.1f} ms", (x1 + 10, y1 + 118), font, 0.36, (200, 200, 200), 1)
        cv2.putText(canvas, f"• HOI Analysis:             {hoi_ms:.2f} ms", (x1 + 10, y1 + 136), font, 0.36, (200, 200, 200), 1)
        cv2.putText(canvas, f"• Temporal HAR (BiGRU):     {har_ms:.2f} ms", (x1 + 10, y1 + 154), font, 0.36, (0, 255, 255), 1)
        cv2.putText(canvas, f"Buffer: {buf_fill}/{buf_cap} {buf_status}", (x1 + 10, y1 + 174), font, 0.36, (0, 215, 255), 1)
        cv2.putText(canvas, f"HOI: {hoi_state[:36]}", (x1 + 10, y1 + 194), font, 0.34, (200, 255, 200), 1)

    def _draw_har_banner(
        self,
        canvas: np.ndarray,
        metrics: Dict[str, object],
        w: int,
        h: int,
    ) -> None:
        """Draw prominent Protocol Validation & HAR Banner at top-center of the screen."""
        action_name = str(metrics.get("har_action", "ACTION_IDLE"))
        conf = float(metrics.get("har_confidence", 1.0))
        is_conf = bool(metrics.get("har_is_confident", True))

        proto_id = str(metrics.get("protocol_id", "HOME_DEMO_PROTOCOL_01"))
        fsm_step = str(metrics.get("fsm_step", "Step 1/10: IDLE"))
        expected_act = str(metrics.get("expected_action", "ACTION_IDLE"))
        status_txt = str(metrics.get("fsm_status", "OK"))
        anom_score = float(metrics.get("anomaly_score", 0.0))
        anomaly_msg = str(metrics.get("fsm_anomaly", ""))

        banner_w, banner_h = 480, 85
        x1 = max(10, (w - banner_w) // 2)
        y1 = 15
        x2 = min(w - 10, x1 + banner_w)
        y2 = y1 + banner_h

        # Semi-transparent background
        if 0 <= y1 < h and 0 <= x1 < w:
            sub = canvas[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if sub.shape[0] == banner_h and sub.shape[1] == (x2 - x1):
                dark = np.zeros(sub.shape, dtype=np.uint8)
                canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.2, dark, 0.8, 0)

        # Border & badge color
        if status_txt == "COMPLETED":
            border_col = (255, 200, 0)    # Cyan / Gold Completed
        elif status_txt == "ANOMALY":
            border_col = (0, 0, 255)      # Red Anomaly
        elif status_txt == "UNCERTAIN":
            border_col = (0, 165, 255)    # Orange Uncertain
        else:
            border_col = (0, 255, 128)    # Green OK

        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Header line: Protocol ID and Step
        cv2.putText(canvas, f"Protocol: {proto_id}  |  {fsm_step}", (x1 + 10, y1 + 18), font, 0.40, (220, 220, 220), 1)

        # Line 2: Observed Action & Expected Action
        conf_pct = conf * 100.0
        cv2.putText(canvas, f"Observed: {action_name} ({conf_pct:.1f}%)", (x1 + 10, y1 + 42), font, 0.50, border_col, 2)
        cv2.putText(canvas, f"Expected: {expected_act}", (x1 + 10, y1 + 60), font, 0.38, (180, 180, 180), 1)

        # Line 3: Status and Anomaly Score
        score_pct = int(round(anom_score * 100))
        if status_txt == "ANOMALY" and anomaly_msg:
            status_line = f"Status: {status_txt} (Score: {score_pct}%) -- {anomaly_msg[:30]}"
        else:
            status_line = f"Status: {status_txt}  |  Anomaly Score: {score_pct}%"
        cv2.putText(canvas, status_line, (x1 + 10, y1 + 76), font, 0.36, border_col, 1)

