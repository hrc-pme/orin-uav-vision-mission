#!/usr/bin/env python3
"""Real-time Orin server for mission-target vehicle geolocation.

This module performs visual fiducial marker detection and target geolocation,
operating independently of the general vehicle tracker.  It detects a chromatic
target marker and verifies its placement on the target vehicle surface, then
computes the geodetic coordinates of the target centroid via D455 depth camera
and MAVLink telemetry.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO
from scipy.optimize import linear_sum_assignment


def _find_orin_dir() -> Path:
    """Find the Orin code root without assuming one fixed deployment depth."""
    env_root = os.environ.get("ORIN_CODE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "orin_d455i_precision_tracker.py").exists():
            return parent
        if (parent / "orin" / "orin_d455i_precision_tracker.py").exists():
            return parent / "orin"
    return here.parents[2] / "orin"


ORIN_DIR = _find_orin_dir()
sys.path.insert(0, str(ORIN_DIR))
sys.path.insert(0, str(ORIN_DIR / "red_white"))

from detect_red_white_x import Detection as MarkerDetection  # noqa: E402
from detect_red_white_x import detect_markers  # noqa: E402
from detect_red_white_x import white_mask_bgr  # noqa: E402
from orin_d455i_precision_tracker import (  # noqa: E402
    CameraSample,
    D455Camera,
    FlightTelemetry,
    Intrinsics,
    JsonlWriter,
    ROS2Publisher,
    TargetGeolocator,
)


class MissionROS2Publisher(ROS2Publisher):
    """ROS 2 publisher tuned for the G3P mission camera.

    The base publisher already records compressed annotated frames for the GUI
    and mission review.  This subclass adds an unannotated original-camera JPEG
    topic at the same throttled image rate, so bags can be replayed for detector
    tuning without storing massive 1080P raw frames.
    """

    def __init__(self, compressed_only: bool = True, image_fps: float = 1.0, jpeg_quality: int = 85):
        super().__init__(compressed_only, image_fps, jpeg_quality)
        from sensor_msgs.msg import CompressedImage

        self.original_compressed_pub = self.node.create_publisher(
            CompressedImage,
            "/g3p/color/image_raw/compressed",
            20,
        )
        self.g3p_annotated_compressed_pub = self.node.create_publisher(
            CompressedImage,
            "/g3p/color/image_annotated/compressed",
            20,
        )
        self._last_original_publish = 0.0

    def publish_frame(
        self,
        sample: CameraSample,
        annotated: np.ndarray,
        detections: list[Dict[str, Any]],
        flight: Dict[str, Any],
        frame_index: int,
    ) -> None:
        now = time.monotonic()
        if now - self._last_original_publish >= self.image_interval_s:
            self._last_original_publish = now
            msg = self._compressed_message(sample.color, "g3p_color", self._stamp())
            if msg is not None:
                self.original_compressed_pub.publish(msg)
            annotated_msg = self._compressed_message(annotated, "g3p_color", self._stamp())
            if annotated_msg is not None:
                self.g3p_annotated_compressed_pub.publish(annotated_msg)
        super().publish_frame(sample, annotated, detections, flight, frame_index)


EARTH_RADIUS_M = 6_378_137.0
DEFAULT_VEHICLE_MODEL = "/home/hrc/Downloads/VisDrone_train1_5090best.pt"
log = logging.getLogger("red_white_target")
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
)


@socketio.on("connect", namespace="/events")
def events_connect(_auth=None):
    return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


TAIWAN_TZ = timezone(timedelta(hours=8))


def taiwan_now() -> str:
    return datetime.now(TAIWAN_TZ).isoformat()


def to_taiwan_iso(timestamp: Any) -> str:
    if not timestamp:
        return taiwan_now()
    try:
        text = str(timestamp)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TAIWAN_TZ).isoformat()
    except Exception:
        return taiwan_now()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def wgs84_to_twd97(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert WGS84 geodetic coordinates to TWD97 TM2 projected easting/northing (metres).

    Uses the GRS80 ellipsoid (TWD97 datum), central meridian 121蝪,
    scale factor k0 = 0.9999, and false easting 250 000 m.
    Returns (easting_m, northing_m).
    """
    a = 6_378_137.0
    f = 1.0 / 298.257222101
    e2 = 2 * f - f * f
    ep2 = e2 / (1 - e2)
    k0 = 0.9999
    lon0 = math.radians(121.0)
    FE = 250_000.0

    lat = math.radians(lat_deg)
    dlon = math.radians(lon_deg) - lon0
    sl, cl = math.sin(lat), math.cos(lat)

    N = a / math.sqrt(1 - e2 * sl * sl)
    T = math.tan(lat) ** 2
    C = ep2 * cl * cl
    A = cl * dlon

    e2_2 = e2 * e2
    e2_3 = e2_2 * e2
    M = a * (
        (1 - e2 / 4 - 3 * e2_2 / 64 - 5 * e2_3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2_2 / 32 + 45 * e2_3 / 1024) * math.sin(2 * lat)
        + (15 * e2_2 / 256 + 45 * e2_3 / 1024) * math.sin(4 * lat)
        - (35 * e2_3 / 3072) * math.sin(6 * lat)
    )
    easting = k0 * N * (
        A + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120
    ) + FE
    northing = k0 * (
        M + N * math.tan(lat) * (
            A * A / 2
            + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
            + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720
        )
    )
    return easting, northing


def gps_distance_m(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    if not a.get("latitude") or not b.get("latitude"):
        return float("inf")
    lat = math.radians((float(a["latitude"]) + float(b["latitude"])) * 0.5)
    north = math.radians(float(a["latitude"]) - float(b["latitude"])) * EARTH_RADIUS_M
    east = (
        math.radians(float(a["longitude"]) - float(b["longitude"]))
        * EARTH_RADIUS_M
        * max(math.cos(lat), 1e-6)
    )
    return math.hypot(north, east)


def expand_xyxy(
    bbox_xywh: Sequence[float],
    width: int,
    height: int,
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w * 0.5, y + h * 0.5
    nw, nh = w * scale_x, h * scale_y
    x1 = max(0, int(round(cx - nw * 0.5)))
    y1 = max(0, int(round(cy - nh * 0.5)))
    x2 = min(width - 1, int(round(cx + nw * 0.5)))
    y2 = min(height - 1, int(round(cy + nh * 0.5)))
    return x1, y1, x2, y2


def vehicle_bbox_from_marker(frame: np.ndarray, marker: MarkerDetection) -> tuple[float, tuple[int, int, int, int]]:
    """Return vehicle surface photometric confidence and the full vehicle bounding box.

    The chromatic fiducial marker is the target indicator only.  The reported
    bounding region encompasses the enclosing vehicle surface containing the marker.
    """
    h, w = frame.shape[:2]
    search_box = expand_xyxy(marker.bbox, w, h, scale_x=5.5, scale_y=3.8)
    x1, y1, x2, y2 = search_box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, search_box
    white = white_mask_bgr(roi)

    mx, my, mw, mh = marker.bbox
    rx1 = max(0, int(mx - x1))
    ry1 = max(0, int(my - y1))
    rx2 = min(white.shape[1], int(mx + mw - x1))
    ry2 = min(white.shape[0], int(my + mh - y1))

    white_no_marker = white.copy()
    if rx2 > rx1 and ry2 > ry1:
        white_no_marker[ry1:ry2, rx1:rx2] = 0
    white_ratio = cv2.countNonZero(white_no_marker) / float(white.size)

    close_w = min(45, max(15, int(max(mw, mh) * 0.35)))
    close_w = close_w + 1 if close_w % 2 == 0 else close_w
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_w, close_w))
    vehicle_mask = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_cx = float(marker.center[0] - x1)
    marker_cy = float(marker.center[1] - y1)
    best: Optional[tuple[float, tuple[int, int, int, int]]] = None
    marker_area = max(1.0, float(mw * mh))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < marker_area * 1.2:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        contains_marker = bx <= marker_cx <= bx + bw and by <= marker_cy <= by + bh
        near_marker = (
            bx - mw * 0.7 <= marker_cx <= bx + bw + mw * 0.7
            and by - mh * 0.7 <= marker_cy <= by + bh + mh * 0.7
        )
        if not (contains_marker or near_marker):
            continue
        aspect = bw / max(float(bh), 1.0)
        aspect_score = 1.0 - min(abs(aspect - 2.0) / 2.0, 0.7)
        score = area * max(aspect_score, 0.3)
        full_box = (
            max(0, x1 + bx),
            max(0, y1 + by),
            min(w - 1, x1 + bx + bw),
            min(h - 1, y1 + by + bh),
        )
        if best is None or score > best[0]:
            best = (score, full_box)

    if best is not None:
        vehicle_box = best[1]
    else:
        vehicle_box = expand_xyxy(marker.bbox, w, h, scale_x=4.2, scale_y=2.8)
    return float(white_ratio), vehicle_box


def marker_bbox_xyxy(marker: MarkerDetection) -> list[float]:
    x, y, w, h = marker.bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def offset_marker(marker: MarkerDetection, dx: int, dy: int) -> MarkerDetection:
    x, y, w, h = marker.bbox
    quad = marker.quad.copy()
    quad[:, 0] += dx
    quad[:, 1] += dy
    return MarkerDetection(
        bbox=(int(x + dx), int(y + dy), int(w), int(h)),
        quad=quad,
        center=(float(marker.center[0] + dx), float(marker.center[1] + dy)),
        score=float(marker.score),
        red_ratio=float(marker.red_ratio),
        white_ratio=float(marker.white_ratio),
        x_score=float(marker.x_score),
        estimated_range_m=marker.estimated_range_m,
    )


def scale_marker(marker: MarkerDetection, scale: float) -> MarkerDetection:
    if scale <= 0:
        return marker
    x, y, w, h = marker.bbox
    quad = marker.quad.astype(np.float32) / scale
    return MarkerDetection(
        bbox=(
            int(round(x / scale)),
            int(round(y / scale)),
            max(1, int(round(w / scale))),
            max(1, int(round(h / scale))),
        ),
        quad=quad,
        center=(float(marker.center[0] / scale), float(marker.center[1] / scale)),
        score=float(marker.score),
        red_ratio=float(marker.red_ratio),
        white_ratio=float(marker.white_ratio),
        x_score=float(marker.x_score),
        estimated_range_m=marker.estimated_range_m,
    )


def vehicle_center_marker(vehicle_box: Sequence[int], *, score: float, white_ratio: float) -> MarkerDetection:
    x1, y1, x2, y2 = [float(v) for v in vehicle_box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(8, int(round(min(abs(x2 - x1), abs(y2 - y1)) * 0.25)))
    x = int(round(cx - side * 0.5)); y = int(round(cy - side * 0.5))
    quad = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float32)
    return MarkerDetection(bbox=(x, y, side, side), quad=quad, center=(float(cx), float(cy)), score=float(score), red_ratio=0.0, white_ratio=float(white_ratio), x_score=0.0, estimated_range_m=None)


def marker_vehicle_geometry_ok(
    marker: MarkerDetection,
    vehicle_box: Sequence[int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    vx1, vy1, vx2, vy2 = [float(v) for v in vehicle_box]
    vehicle_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
    mx, my, mw, mh = [float(v) for v in marker.bbox]
    marker_area = max(1.0, mw * mh)
    area_ratio = marker_area / vehicle_area
    aspect = max(mw, mh) / max(min(mw, mh), 1.0)
    cx, cy = marker.center
    inside = vx1 <= cx <= vx2 and vy1 <= cy <= vy2
    return inside and aspect <= 1.35 and min_area_ratio <= area_ratio <= max_area_ratio


def xyxy_list(box: Sequence[int]) -> list[float]:
    return [float(v) for v in box]


def clamp_xyxy(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_xyxy_abs(
    box: Sequence[float],
    width: int,
    height: int,
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return clamp_xyxy((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5), width, height)


def box_contains_point(box: Sequence[float], point: Sequence[float], margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    x, y = [float(v) for v in point]
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


@dataclass
class VehicleCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int
    source: str = "yolo"


class KalmanTrack:
    """Single-target Kalman tracker with constant-velocity model (SORT-style).

    State vector: [cx, cy, vx, vy]
    Measurement:  [cx, cy]
    """

    _next_id: int = 1

    def __init__(
        self,
        center: tuple[float, float],
        marker: MarkerDetection,
        vehicle_box: tuple[int, int, int, int],
        vehicle_candidate: Optional[VehicleCandidate],
        white_ratio: float,
        target_location: Dict[str, Any],
        *,
        confirm_hits: int,
        max_age: int,
    ) -> None:
        cx, cy = float(center[0]), float(center[1])
        self._x = np.array([cx, cy, 0.0, 0.0])
        self._P = np.diag([50.0, 50.0, 500.0, 500.0])
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._Q = np.diag([2.0, 2.0, 20.0, 20.0])
        self._R = np.diag([16.0, 16.0])  # ~4 px measurement std-dev

        self.track_id = f"RW{KalmanTrack._next_id:03d}"
        KalmanTrack._next_id += 1
        self.hits = 1
        self.time_since_update = 0
        self.confirm_hits = confirm_hits
        self.max_age = max_age

        self.marker = marker
        self.vehicle_box = vehicle_box
        self.vehicle_candidate = vehicle_candidate
        self.white_ratio = white_ratio
        self.target_location = target_location
        self.last_update_monotonic = time.monotonic()

    # ------------------------------------------------------------------
    def predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.time_since_update += 1

    def update(
        self,
        center: tuple[float, float],
        marker: MarkerDetection,
        vehicle_box: tuple[int, int, int, int],
        vehicle_candidate: Optional[VehicleCandidate],
        white_ratio: float,
        target_location: Dict[str, Any],
    ) -> None:
        z = np.array([float(center[0]), float(center[1])])
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._H @ self._x)
        self._P = (np.eye(4) - K @ self._H) @ self._P
        self.hits += 1
        self.time_since_update = 0
        self.marker = marker
        self.vehicle_box = vehicle_box
        self.vehicle_candidate = vehicle_candidate
        self.white_ratio = white_ratio
        self.target_location = target_location
        self.last_update_monotonic = time.monotonic()

    # ------------------------------------------------------------------
    @property
    def center(self) -> tuple[float, float]:
        return (float(self._x[0]), float(self._x[1]))

    @property
    def predicted_vehicle_box(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.vehicle_box
        box_cx = (x1 + x2) * 0.5
        box_cy = (y1 + y2) * 0.5
        dx = float(self._x[0]) - box_cx
        dy = float(self._x[1]) - box_cy
        return (
            int(round(x1 + dx)),
            int(round(y1 + dy)),
            int(round(x2 + dx)),
            int(round(y2 + dy)),
        )

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.confirm_hits

    @property
    def is_dead(self) -> bool:
        return self.time_since_update > self.max_age


class MarkerTracker:
    """Multi-target marker tracker: Kalman filter + Hungarian assignment (SORT).

    Each detected marker is matched to the nearest predicted track using the
    Hungarian algorithm (globally optimal, no greedy bias).  Unmatched
    detections spawn new tentative tracks; tracks not seen for ``max_age``
    consecutive frames are dropped.  A track is only reported once it has
    been matched for ``confirm_hits`` frames, suppressing single-frame
    false positives.
    """

    def __init__(
        self,
        *,
        confirm_hits: int = 3,
        max_age: int = 8,
        max_dist_px: float = 120.0,
        min_iou: float = 0.05,
        geo_break_m: float = 4.0,
    ) -> None:
        self.tracks: list[KalmanTrack] = []
        self.confirm_hits = confirm_hits
        self.max_age = max_age
        self.max_dist_px = max_dist_px
        self.min_iou = min_iou
        self.geo_break_m = geo_break_m

    def _assign(
        self,
        det_centers: list[tuple[float, float]],
        det_boxes: list[tuple[int, int, int, int]],
        det_targets: list[Dict[str, Any]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not det_centers:
            return [], list(range(len(det_centers))), list(range(len(self.tracks)))
        cost = np.full((len(det_centers), len(self.tracks)), 1e6, dtype=np.float64)
        for d_idx, center in enumerate(det_centers):
            det_target = det_targets[d_idx] if d_idx < len(det_targets) else {}
            for t_idx, trk in enumerate(self.tracks):
                dist = math.hypot(center[0] - trk.center[0], center[1] - trk.center[1])
                iou = box_iou(det_boxes[d_idx], trk.predicted_vehicle_box)

                if dist > self.max_dist_px and iou < self.min_iou:
                    continue

                old_target = trk.target_location or {}
                if det_target.get("valid") and old_target.get("valid"):
                    if gps_distance_m(det_target, old_target) > self.geo_break_m:
                        continue

                # SORT commonly matches by IoU.  The center term keeps small
                # marker boxes stable when the full vehicle box jitters.
                cost[d_idx, t_idx] = (1.0 - iou) + 0.45 * min(dist / self.max_dist_px, 2.0)
        rows, cols = linear_sum_assignment(cost)
        matched: list[tuple[int, int]] = []
        unmatched_d = list(range(len(det_centers)))
        unmatched_t = list(range(len(self.tracks)))
        for r, c in zip(rows, cols):
            if cost[r, c] < 1e6:
                matched.append((r, c))
                unmatched_d.remove(r)
                unmatched_t.remove(c)
        return matched, unmatched_d, unmatched_t

    def step(
        self,
        markers: list[MarkerDetection],
        vehicle_data: list[
            tuple[float, tuple[int, int, int, int], Optional[VehicleCandidate], Dict[str, Any]]
        ],
    ) -> list[KalmanTrack]:
        """Update all tracks and return confirmed tracks that are still alive.

        Keeping unmatched confirmed tracks alive for ``max_age`` frames is the
        practical SORT behavior that prevents flicker and ID churn when the
        marker is briefly missed by exposure blur or motion.
        """
        for trk in self.tracks:
            trk.predict()

        centers = [tuple(float(v) for v in m.center) for m in markers]
        det_boxes = [item[1] for item in vehicle_data]
        det_targets = [item[3] for item in vehicle_data]
        matched, unmatched_d, _ = self._assign(centers, det_boxes, det_targets)

        for d_idx, t_idx in matched:
            wr, vbox, vcand, tloc = vehicle_data[d_idx]
            self.tracks[t_idx].update(centers[d_idx], markers[d_idx], vbox, vcand, wr, tloc)

        for d_idx in unmatched_d:
            wr, vbox, vcand, tloc = vehicle_data[d_idx]
            self.tracks.append(
                KalmanTrack(
                    centers[d_idx], markers[d_idx], vbox, vcand, wr, tloc,
                    confirm_hits=self.confirm_hits,
                    max_age=self.max_age,
                )
            )

        self.tracks = [t for t in self.tracks if not t.is_dead]
        confirmed = [t for t in self.tracks if t.confirmed]
        return confirmed


class YoloVehicleDetector:
    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(args.vehicle_model and not args.disable_vehicle_model)
        self.model = None
        self.device = "cpu"
        self.use_half = False
        self.class_ids: list[int] = []
        self.class_names: Dict[int, str] = {}
        self.args = args
        if not self.enabled:
            log.info("vehicle model disabled; using marker-derived vehicle boxes")
            return

        import torch
        from ultralytics import YOLO

        model_path = Path(args.vehicle_model).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"vehicle model not found: {model_path}")
        cuda_ready = torch.cuda.is_available()
        self.device = "0" if args.vehicle_device == "auto" and cuda_ready else args.vehicle_device
        if self.device != "cpu" and not cuda_ready:
            raise RuntimeError(f"CUDA device {self.device} was requested, but CUDA is unavailable")
        self.use_half = bool(args.vehicle_half and self.device != "cpu")
        self.model = YOLO(str(model_path))
        names = self.model.names
        wanted = {name.strip().lower() for name in args.vehicle_classes.split(",") if name.strip()}
        self.class_ids = [
            int(idx) for idx, name in names.items() if str(name).strip().lower() in wanted
        ]
        if not self.class_ids:
            raise RuntimeError(f"vehicle classes {sorted(wanted)} not found in model names: {names}")
        self.class_names = {int(idx): str(names[idx]) for idx in self.class_ids}
        warm = np.zeros((args.vehicle_imgsz, args.vehicle_imgsz, 3), dtype=np.uint8)
        self.model.predict(
            warm,
            conf=args.vehicle_conf,
            iou=args.vehicle_iou,
            classes=self.class_ids,
            imgsz=args.vehicle_imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )
        if cuda_ready and self.device != "cpu":
            torch.cuda.synchronize()
        log.info(
            "vehicle model ready: %s classes=%s device=%s half=%s tiles=%s",
            model_path,
            self.class_names,
            self.device,
            self.use_half,
            int(getattr(args, "vehicle_tiles", 1)),
        )

    def _predict_tile(
        self,
        tile: np.ndarray,
        *,
        offset_x: int,
        offset_y: int,
        full_w: int,
        full_h: int,
    ) -> list[VehicleCandidate]:
        result = self.model.predict(
            tile,
            conf=self.args.vehicle_conf,
            iou=self.args.vehicle_iou,
            classes=self.class_ids,
            imgsz=self.args.vehicle_imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        out: list[VehicleCandidate] = []
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().tolist()
            out.append(
                VehicleCandidate(
                    bbox=clamp_xyxy(
                        [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                        full_w,
                        full_h,
                    ),
                    confidence=float(result.boxes.conf[i].item()),
                    class_name=self.class_names.get(cls_id, str(cls_id)),
                    class_id=cls_id,
                )
            )
        return out

    def _nms(self, candidates: list[VehicleCandidate]) -> list[VehicleCandidate]:
        kept: list[VehicleCandidate] = []
        for cand in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if all(box_iou(cand.bbox, old.bbox) < self.args.vehicle_tile_nms_iou for old in kept):
                kept.append(cand)
        return kept

    def detect(self, frame: np.ndarray) -> list[VehicleCandidate]:
        if not self.enabled or self.model is None:
            return []
        h, w = frame.shape[:2]
        candidates = self._predict_tile(frame, offset_x=0, offset_y=0, full_w=w, full_h=h)
        tiles = max(1, int(getattr(self.args, "vehicle_tiles", 1)))
        if tiles > 1:
            overlap = max(0.0, min(0.8, float(getattr(self.args, "vehicle_tile_overlap", 0.25))))
            step_x = max(1, int(math.ceil(w / tiles * (1.0 - overlap))))
            step_y = max(1, int(math.ceil(h / tiles * (1.0 - overlap))))
            tile_w = max(64, int(math.ceil(w / tiles * (1.0 + overlap))))
            tile_h = max(64, int(math.ceil(h / tiles * (1.0 + overlap))))
            xs = sorted(set(list(range(0, max(1, w - tile_w + 1), step_x)) + [max(0, w - tile_w)]))
            ys = sorted(set(list(range(0, max(1, h - tile_h + 1), step_y)) + [max(0, h - tile_h)]))
            for y1 in ys:
                for x1 in xs:
                    crop = frame[y1:min(h, y1 + tile_h), x1:min(w, x1 + tile_w)]
                    if crop.size == 0:
                        continue
                    candidates.extend(
                        self._predict_tile(crop, offset_x=x1, offset_y=y1, full_w=w, full_h=h)
                    )
        return self._nms(candidates)


def white_vehicle_ratio(frame: np.ndarray, box: Sequence[int], marker: MarkerDetection) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_xyxy(box, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask = white_mask_bgr(roi)
    mx, my, mw, mh = marker.bbox
    rx1 = max(0, int(mx - x1))
    ry1 = max(0, int(my - y1))
    rx2 = min(mask.shape[1], int(mx + mw - x1))
    ry2 = min(mask.shape[0], int(my + mh - y1))
    if rx2 > rx1 and ry2 > ry1:
        mask[ry1:ry2, rx1:rx2] = 0
    return cv2.countNonZero(mask) / float(mask.size)


def vehicle_body_white_ratio(frame: np.ndarray, box: Sequence[int]) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_xyxy(box, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask = white_mask_bgr(roi)
    return cv2.countNonZero(mask) / float(mask.size)


def roof_red_marker_from_vehicle(
    frame: np.ndarray,
    box: Sequence[int],
    *,
    roi_scale: float,
    min_red_ratio: float,
    min_red_pixels: int,
    red_h1_max: int,
    red_h2_min: int,
    red_s_min: int,
    red_v_min: int,
    min_marker_score: float,
    min_x_score: float,
    min_x_white_ratio: float,
) -> Optional[tuple[MarkerDetection, float, tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = expand_xyxy_abs(box, w, h, scale=roi_scale)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_h1_max = int(np.clip(red_h1_max, 0, 40))
    red_h2_min = int(np.clip(red_h2_min, 120, 179))
    red_s_min = int(np.clip(red_s_min, 0, 255))
    red_v_min = int(np.clip(red_v_min, 0, 255))
    red1 = cv2.inRange(hsv, np.array([0, red_s_min, red_v_min]), np.array([red_h1_max, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([red_h2_min, red_s_min, red_v_min]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_pixels = int(cv2.countNonZero(red_mask))
    red_ratio = red_pixels / float(max(1, red_mask.size))
    if red_pixels < min_red_pixels or red_ratio < min_red_ratio:
        return None

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw * bh < min_red_pixels:
        return None

    side = max(6, int(max(bw, bh)))
    cx = rx1 + x + bw * 0.5
    cy = ry1 + y + bh * 0.5
    mx = int(round(cx - side * 0.5))
    my = int(round(cy - side * 0.5))
    mx = max(0, min(w - side, mx))
    my = max(0, min(h - side, my))
    quad = np.array(
        [[mx, my], [mx + side, my], [mx + side, my + side], [mx, my + side]],
        dtype=np.float32,
    )
    score = max(float(min_marker_score), min(1.0, red_ratio / max(min_red_ratio, 1e-6)))
    marker = MarkerDetection(
        bbox=(mx, my, side, side),
        quad=quad,
        center=(float(cx), float(cy)),
        score=score,
        red_ratio=float(red_ratio),
        white_ratio=max(float(min_x_white_ratio), float(red_ratio)),
        x_score=max(float(min_x_score), score),
        estimated_range_m=None,
    )
    return marker, float(red_ratio), (rx1, ry1, rx2, ry2)


def match_vehicle_candidate(
    frame: np.ndarray,
    marker: MarkerDetection,
    vehicles: Sequence[VehicleCandidate],
    *,
    min_white_ratio: float,
) -> tuple[float, tuple[int, int, int, int], Optional[VehicleCandidate]]:
    h, w = frame.shape[:2]
    marker_box = marker_bbox_xyxy(marker)
    marker_size = max(marker.bbox[2], marker.bbox[3])
    best: Optional[tuple[float, float, tuple[int, int, int, int], VehicleCandidate]] = None
    for vehicle in vehicles:
        expanded = expand_xyxy_abs(vehicle.bbox, w, h, scale=1.12)
        if not box_contains_point(expanded, marker.center, margin=marker_size * 0.35):
            if box_iou(expanded, marker_box) <= 0.0:
                continue
        white_ratio = white_vehicle_ratio(frame, vehicle.bbox, marker)
        if white_ratio < min_white_ratio:
            continue
        score = vehicle.confidence * 0.55 + marker.score * 0.35 + min(white_ratio / 0.20, 1.0) * 0.10
        if best is None or score > best[0]:
            best = (score, white_ratio, vehicle.bbox, vehicle)
    if best is None:
        white_ratio, vehicle_box = vehicle_bbox_from_marker(frame, marker)
        return white_ratio, vehicle_box, None
    return best[1], best[2], best[3]


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._revision = 0
        self._last_preview_publish = 0.0
        self.flight: Dict[str, Any] = {}
        self.last_target: Optional[Dict[str, Any]] = None
        self.last_detections: list[Dict[str, Any]] = []
        self.mission: Dict[str, Any] = {}

    def publish_frame(self, frame: np.ndarray, quality: int, scale: float = 1.0, max_fps: float = 0.0) -> int:
        now = time.monotonic()
        if max_fps > 0:
            min_interval = 1.0 / max_fps
            with self._lock:
                if now - self._last_preview_publish < min_interval:
                    return self._revision
                self._last_preview_publish = now
        out = frame
        if 0.05 < scale < 1.0:
            h, w = frame.shape[:2]
            out = cv2.resize(
                frame,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return self._revision
        with self._lock:
            self._jpeg = encoded.tobytes()
            self._revision += 1
            return self._revision

    def get_jpeg(self) -> tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._revision

    def update_status(
        self,
        flight: Dict[str, Any],
        detections: list[Dict[str, Any]],
        target: Optional[Dict[str, Any]],
        mission: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self.flight = copy.deepcopy(flight)
            self.last_detections = copy.deepcopy(detections)
            if target is not None:
                self.last_target = copy.deepcopy(target)
            if mission is not None:
                self.mission = copy.deepcopy(mission)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "device": "orin-mission-target",
                "gps": copy.deepcopy(self.flight),
                "last_detection": copy.deepcopy(self.last_target),
                "detections": copy.deepcopy(self.last_detections),
                "mission": copy.deepcopy(self.mission),
                "image_revision": self._revision,
            }


class CsvSummaryWriter:
    """Live mission summary for fast post-flight inspection in a spreadsheet."""

    FIELDS = [
        "timestamp",
        "timestamp_utc",
        "row_type",
        "target_id",
        "target_latitude",
        "target_longitude",
        "target_altitude_m",
        "target_twd97_e",
        "target_twd97_n",
        "horizontal_sigma_m",
        "slant_range_m",
        "target_valid",
        "target_reason",
        "detection_confidence",
        "white_vehicle_ratio",
        "vehicle_bbox",
        "marker_bbox",
        "uav_latitude",
        "uav_longitude",
        "uav_altitude_msl",
        "uav_relative_altitude_m",
        "uav_rangefinder_m",
        "gps_fix_type",
        "satellites",
        "hdop",
        "flight_mode",
        "armed",
        "heading_deg",
        "groundspeed_mps",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "battery_pct",
        "voltage_v",
        "rtl_enabled",
        "rtl_sent",
        "rtl_mode_confirmed",
        "rtl_command",
        "rtl_reason",
        "last_command_ack_command",
        "last_command_ack_result",
        "last_command_ack_progress",
        "last_command_ack_result_param2",
    ]

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._stream = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._stream.flush()

    @staticmethod
    def _flight_fields(flight: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "uav_latitude": flight.get("latitude"),
            "uav_longitude": flight.get("longitude"),
            "uav_altitude_msl": flight.get("altitude"),
            "uav_relative_altitude_m": flight.get("relative_altitude"),
            "uav_rangefinder_m": flight.get("rangefinder_m"),
            "gps_fix_type": flight.get("fix_type"),
            "satellites": flight.get("satellites"),
            "hdop": flight.get("hdop"),
            "flight_mode": flight.get("flight_mode"),
            "armed": flight.get("armed"),
            "heading_deg": flight.get("heading"),
            "groundspeed_mps": flight.get("groundspeed"),
            "roll_rad": flight.get("roll"),
            "pitch_rad": flight.get("pitch"),
            "yaw_rad": flight.get("yaw"),
            "battery_pct": flight.get("battery_remaining_pct"),
            "voltage_v": flight.get("voltage_v"),
            "last_command_ack_command": flight.get("last_command_ack_command"),
            "last_command_ack_result": flight.get("last_command_ack_result"),
            "last_command_ack_progress": flight.get("last_command_ack_progress"),
            "last_command_ack_result_param2": flight.get("last_command_ack_result_param2"),
        }

    def write_flight(self, flight: Dict[str, Any], rtl_status: Dict[str, Any]) -> None:
        timestamp_utc = flight.get("timestamp") or utc_now()
        row = {
            "timestamp": to_taiwan_iso(timestamp_utc),
            "timestamp_utc": timestamp_utc,
            "row_type": "flight",
            "rtl_enabled": rtl_status.get("enabled"),
            "rtl_sent": rtl_status.get("sent"),
            "rtl_mode_confirmed": rtl_status.get("mode_confirmed"),
            "rtl_command": rtl_status.get("command"),
            "rtl_reason": rtl_status.get("reason"),
        }
        row.update(self._flight_fields(flight))
        self._write(row)

    def write_target(
        self,
        event: Dict[str, Any],
        flight: Dict[str, Any],
        rtl_status: Dict[str, Any],
    ) -> None:
        detections = event.get("detections") or [{}]
        det = detections[0] if detections else {}
        target = event.get("target_location") or det.get("target_location") or {}
        timestamp_utc = event.get("timestamp") or utc_now()
        row = {
            "timestamp": to_taiwan_iso(timestamp_utc),
            "timestamp_utc": timestamp_utc,
            "row_type": "target",
            "target_id": event.get("target_id") or det.get("track_id"),
            "target_latitude": target.get("latitude"),
            "target_longitude": target.get("longitude"),
            "target_altitude_m": target.get("altitude"),
            "target_twd97_e": target.get("twd97_x"),
            "target_twd97_n": target.get("twd97_y"),
            "horizontal_sigma_m": target.get("estimated_horizontal_sigma_m"),
            "slant_range_m": target.get("slant_range_m"),
            "target_valid": target.get("valid"),
            "target_reason": target.get("reason"),
            "detection_confidence": det.get("confidence"),
            "white_vehicle_ratio": det.get("white_vehicle_ratio"),
            "vehicle_bbox": json.dumps(det.get("vehicle_bbox"), separators=(",", ":")),
            "marker_bbox": json.dumps(det.get("marker_bbox"), separators=(",", ":")),
            "rtl_enabled": rtl_status.get("enabled"),
            "rtl_sent": rtl_status.get("sent"),
            "rtl_mode_confirmed": rtl_status.get("mode_confirmed"),
            "rtl_command": rtl_status.get("command"),
            "rtl_reason": rtl_status.get("reason"),
        }
        row.update(self._flight_fields(flight))
        self._write(row)

    def _write(self, row: Dict[str, Any]) -> None:
        clean = {key: row.get(key) for key in self.FIELDS}
        with self._lock:
            self._writer.writerow(clean)
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()


class AutoReturnController:
    """Send MAV_CMD_DO_SET_MODE after the first mission target is detected."""

    def __init__(self, flight: FlightTelemetry, args: argparse.Namespace):
        self.flight = flight
        self.args = args
        self._lock = threading.RLock()
        self._sent = False
        self._last_reason = "waiting"
        self._last_command = None
        self._last_ack_command = None
        self._last_ack_result = None
        self._last_ack_progress = None
        self._last_ack_result_param2 = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            flight = self.flight.get()
            mode = str(flight.get("flight_mode") or "")
            ack_command = flight.get("last_command_ack_command")
            ack_result = flight.get("last_command_ack_result")
            if ack_command is not None:
                self._last_ack_command = ack_command
                self._last_ack_result = ack_result
                self._last_ack_progress = flight.get("last_command_ack_progress")
                self._last_ack_result_param2 = flight.get("last_command_ack_result_param2")
            confirmed = self._sent and mode.upper() == "RTL"
            ack_accepted = (
                self._sent
                and int(self._last_ack_command or -1) == 176
                and int(self._last_ack_result or -1) == 0
            )
            reason = "mode_confirmed_rtl" if confirmed else self._last_reason
            return {
                "enabled": bool(self.args.auto_rtl),
                "sent": self._sent,
                "mode_confirmed": confirmed,
                "ack_accepted": ack_accepted,
                "command": self._last_command,
                "reason": reason,
                "last_command_ack_command": self._last_ack_command,
                "last_command_ack_result": self._last_ack_result,
                "last_command_ack_progress": self._last_ack_progress,
                "last_command_ack_result_param2": self._last_ack_result_param2,
            }

    def trigger_for_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        target = event.get("target_location") or {}
        with self._lock:
            if not self.args.auto_rtl:
                self._last_reason = "auto_rtl_disabled"
                return self.status()
            if self._sent:
                self._last_reason = "already_sent"
                return self.status()
            if self.args.auto_rtl_require_valid_target and not target.get("valid"):
                self._last_reason = str(target.get("reason") or "target_location_invalid")
                return self.status()
            self._sent = True
            self._last_reason = "sending"
            self._last_ack_command = None
            self._last_ack_result = None
            self._last_ack_progress = None
            self._last_ack_result_param2 = None

        threading.Thread(
            target=self._send_command_thread,
            name="auto-rtl-command",
            daemon=True,
        ).start()
        return self.status()

    def _send_command_thread(self) -> None:
        from pymavlink import mavutil

        command_name = "MAV_CMD_DO_SET_MODE"
        try:
            conn = getattr(self.flight, "_conn", None)
            if conn is None:
                raise RuntimeError("mavlink_connection_not_ready")
            target_system = int(getattr(conn, "target_system", 0) or self.args.rtl_target_system)
            target_component = int(
                getattr(conn, "target_component", 0) or self.args.rtl_target_component
            )
            command = getattr(mavutil.mavlink, command_name)
            self._last_command = (
                f"{command_name}(176) param1={self.args.rtl_mode_param1} "
                f"param2={self.args.rtl_custom_mode} + SET_MODE "
                f"base_mode={self.args.rtl_mode_param1} custom_mode={self.args.rtl_custom_mode}"
            )
            deadline = time.monotonic() + max(0.5, float(self.args.rtl_confirm_timeout_s))
            repeat = max(1, int(self.args.rtl_command_repeats))
            attempt = 0
            while attempt < repeat or time.monotonic() < deadline:
                attempt += 1
                conn.mav.command_long_send(
                    target_system,
                    target_component,
                    command,
                    0,
                    float(self.args.rtl_mode_param1),
                    float(self.args.rtl_custom_mode),
                    0, 0, 0, 0, 0,
                )
                conn.mav.set_mode_send(
                    target_system,
                    int(self.args.rtl_mode_param1),
                    int(self.args.rtl_custom_mode),
                )
                time.sleep(max(0.05, float(self.args.rtl_command_interval_s)))
                status = self.status()
                if status.get("mode_confirmed") or status.get("ack_accepted"):
                    break
            with self._lock:
                status = self.status()
                if status.get("mode_confirmed"):
                    self._last_reason = "mode_confirmed_rtl"
                elif status.get("ack_accepted"):
                    self._last_reason = "command_ack_accepted"
                else:
                    self._last_reason = "sent_waiting_confirmation"
            log.warning(
                "AUTO RTL command sent: %s target=%d/%d attempts=%d status=%s",
                self._last_command,
                target_system,
                target_component,
                attempt,
                self.status(),
            )
        except Exception as exc:
            with self._lock:
                self._sent = False
                self._last_reason = f"send_failed:{exc}"
            log.error("AUTO RTL command failed: %s", exc)


class FakeMarkerCamera:
    """Small synthetic camera for desktop verification without D455."""

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._sample: Optional[CameraSample] = None

    def start(self) -> None:
        self._running.set()
        threading.Thread(target=self._loop, name="fake-marker-camera", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()

    def get(self) -> Optional[CameraSample]:
        with self._lock:
            if self._sample is None:
                return None
            s = self._sample
            return CameraSample(
                s.color.copy(),
                None if s.depth_m is None else s.depth_m.copy(),
                s.intrinsics,
                s.camera_timestamp_ms,
                copy.deepcopy(s.imu),
            )

    def _loop(self) -> None:
        intr = Intrinsics(self.width, self.height, 610.0, 610.0, self.width / 2, self.height / 2)
        started = time.monotonic()
        while self._running.is_set():
            t = time.monotonic() - started
            img = np.full((self.height, self.width, 3), (50, 70, 55), np.uint8)
            cx = int(self.width * 0.5 + math.sin(t * 0.7) * self.width * 0.22)
            cy = int(self.height * 0.52)
            car_w, car_h = 230, 110
            x1, y1 = cx - car_w // 2, cy - car_h // 2
            x2, y2 = cx + car_w // 2, cy + car_h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (238, 238, 232), -1)
            cv2.rectangle(img, (x1 + 25, y1 + 18), (x2 - 25, y2 - 18), (248, 248, 244), -1)
            cv2.circle(img, (x1 + 45, y2), 16, (20, 20, 20), -1)
            cv2.circle(img, (x2 - 45, y2), 16, (20, 20, 20), -1)
            m = 62
            mx1, my1 = cx - m // 2, cy - m // 2
            mx2, my2 = cx + m // 2, cy + m // 2
            cv2.rectangle(img, (mx1, my1), (mx2, my2), (0, 0, 255), -1)
            thick = 16
            margin = 13
            cv2.line(img, (mx1 + margin, my1 + margin), (mx2 - margin, my2 - margin), (255, 255, 255), thick)
            cv2.line(img, (mx2 - margin, my1 + margin), (mx1 + margin, my2 - margin), (255, 255, 255), thick)
            depth = np.full((self.height, self.width), 30.0, np.float32)
            depth[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)] = 28.0
            with self._lock:
                self._sample = CameraSample(img, depth, intr, t * 1000.0, {"source": "fake"})
            time.sleep(1.0 / max(self.fps, 1))


class OpenCVCamera:
    """Camera adapter for non-RealSense USB/V4L2/GStreamer cameras.

    It intentionally returns no depth map.  The existing geolocator will fall
    back to attitude-compensated ground-ray estimation from aircraft altitude,
    which is the same RGB-only mode used by the competition RTL script.
    """

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        *,
        backend: int = cv2.CAP_ANY,
        fx: float = 0.0,
        fy: float = 0.0,
        ppx: float = 0.0,
        ppy: float = 0.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._sample: Optional[CameraSample] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self._running.set()
        threading.Thread(target=self._loop, name="opencv-camera", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        cap = self._cap
        if cap is not None:
            cap.release()

    def get(self) -> Optional[CameraSample]:
        with self._lock:
            if self._sample is None:
                return None
            s = self._sample
            return CameraSample(
                s.color.copy(),
                None,
                s.intrinsics,
                s.camera_timestamp_ms,
                copy.deepcopy(s.imu),
            )

    def _open(self) -> cv2.VideoCapture:
        source: Any = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source, self.backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera source: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _intrinsics(self, frame: np.ndarray) -> Intrinsics:
        h, w = frame.shape[:2]
        fx = self.fx if self.fx > 0 else 0.9 * float(w)
        fy = self.fy if self.fy > 0 else fx
        ppx = self.ppx if self.ppx > 0 else float(w) * 0.5
        ppy = self.ppy if self.ppy > 0 else float(h) * 0.5
        return Intrinsics(w, h, fx, fy, ppx, ppy)

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                self._cap = self._open()
                log.info("OpenCV camera opened: source=%s backend=%s", self.source, self.backend)
                while self._running.is_set():
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("camera read failed")
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    ts_ms = time.monotonic() * 1000.0
                    sample = CameraSample(
                        frame,
                        None,
                        self._intrinsics(frame),
                        ts_ms,
                        {"source": "opencv", "camera_source": self.source},
                    )
                    with self._lock:
                        self._sample = sample
            except Exception as exc:
                log.warning("OpenCV camera error: %s; retrying", exc)
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(1.0)


class GStreamerJpegCamera:
    """Read JPEG frames from gst-launch stdout.

    This supports Jetson builds where cv2 has no GStreamer backend.  The
    pipeline may end with raw BGR/BGRx/NV12 video; this class appends jpegenc
    and fdsink unless the caller already provided an fdsink.
    """

    def __init__(
        self,
        pipeline: str,
        width: int,
        height: int,
        fps: int,
        *,
        fx: float = 0.0,
        fy: float = 0.0,
        ppx: float = 0.0,
        ppy: float = 0.0,
    ):
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.fps = fps
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._sample: Optional[CameraSample] = None
        self._proc: Any = None
        self._failure_count = 0
        self._last_viewfinder_reset = 0.0

    def start(self) -> None:
        self._running.set()
        threading.Thread(target=self._loop, name="gst-jpeg-camera", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def get(self) -> Optional[CameraSample]:
        with self._lock:
            if self._sample is None:
                return None
            s = self._sample
            return CameraSample(
                s.color.copy(),
                None,
                s.intrinsics,
                s.camera_timestamp_ms,
                copy.deepcopy(s.imu),
            )

    def _command(self) -> str:
        pipeline = self.pipeline.strip()
        if "fdsink" not in pipeline:
            pipeline = f"{pipeline} ! jpegenc quality=85 ! fdsink fd=1"
        return pipeline

    def _intrinsics(self, frame: np.ndarray) -> Intrinsics:
        h, w = frame.shape[:2]
        fx = self.fx if self.fx > 0 else 0.9 * float(w)
        fy = self.fy if self.fy > 0 else fx
        ppx = self.ppx if self.ppx > 0 else float(w) * 0.5
        ppy = self.ppy if self.ppy > 0 else float(h) * 0.5
        return Intrinsics(w, h, fx, fy, ppx, ppy)

    def _loop(self) -> None:
        import subprocess
        import shlex

        while self._running.is_set():
            pipeline = self._command()
            cmd = ["gst-launch-1.0", "-q", *shlex.split(pipeline)]
            try:
                log.info("starting GStreamer JPEG camera: gst-launch-1.0 -q %s", pipeline)
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                assert self._proc.stdout is not None
                buf = b""
                while self._running.is_set() and self._proc.poll() is None:
                    chunk = self._proc.stdout.read(8192)
                    if not chunk:
                        time.sleep(0.02)
                        continue
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9", start + 2)
                        if start < 0 or end < 0:
                            if start > 0:
                                buf = buf[start:]
                            break
                        jpg, buf = buf[start:end + 2], buf[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                        sample = CameraSample(
                            frame,
                            None,
                            self._intrinsics(frame),
                            time.monotonic() * 1000.0,
                            {"source": "gstreamer-jpeg"},
                        )
                        with self._lock:
                            self._sample = sample
                err = b""
                if self._proc.stderr is not None:
                    try:
                        err = self._proc.stderr.read(4096)
                    except Exception:
                        err = b""
                if err:
                    log.warning("GStreamer camera stderr: %s", err.decode(errors="replace").strip())
            except Exception as exc:
                log.warning("GStreamer JPEG camera error: %s; retrying", exc)
            finally:
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                self._proc = None
                time.sleep(1.0)


class RtspFfmpegCamera:
    """Low-latency RTSP camera reader using ffmpeg rawvideo pipe."""

    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        fps: int,
        *,
        fx: float = 0.0,
        fy: float = 0.0,
        ppx: float = 0.0,
        ppy: float = 0.0,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._sample: Optional[CameraSample] = None
        self._proc: Any = None
        self._failure_count = 0
        self._last_viewfinder_reset = 0.0

    def start(self) -> None:
        self._running.set()
        threading.Thread(target=self._loop, name="rtsp-ffmpeg-camera", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def get(self) -> Optional[CameraSample]:
        with self._lock:
            if self._sample is None:
                return None
            s = self._sample
            return CameraSample(
                s.color.copy(),
                None,
                s.intrinsics,
                s.camera_timestamp_ms,
                copy.deepcopy(s.imu),
            )

    def _intrinsics(self, frame: np.ndarray) -> Intrinsics:
        h, w = frame.shape[:2]
        fx = self.fx if self.fx > 0 else 0.9 * float(w)
        fy = self.fy if self.fy > 0 else fx
        ppx = self.ppx if self.ppx > 0 else float(w) * 0.5
        ppy = self.ppy if self.ppy > 0 else float(h) * 0.5
        return Intrinsics(w, h, fx, fy, ppx, ppy)

    def _frame_size(self) -> int:
        return int(self.width) * int(self.height) * 3

    def _read_exact(self, pipe: Any, size: int, timeout_s: float = 2.0) -> Optional[bytes]:
        import os as _os
        import select

        fd = pipe.fileno()
        chunks: list[bytes] = []
        remaining = size
        deadline = time.monotonic() + timeout_s
        while remaining > 0 and self._running.is_set():
            wait_s = deadline - time.monotonic()
            if wait_s <= 0:
                return None
            readable, _, _ = select.select([fd], [], [], min(wait_s, 0.25))
            if not readable:
                continue
            chunk = _os.read(fd, remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _maybe_restart_viewfinder(self) -> bool:
        every = max(1, int(os.environ.get("G3P_VIEWFINDER_RESET_EVERY_N_FAILURES", "1")))
        cooldown = max(1.0, float(os.environ.get("G3P_VIEWFINDER_RESET_COOLDOWN_S", "12")))
        if self._failure_count % every != 0:
            return False
        now = time.monotonic()
        if now - self._last_viewfinder_reset < cooldown:
            return False
        control_script = Path(__file__).resolve().with_name("start_g3p_viewfinder.py")
        if not control_script.exists():
            return False
        camera_ip = os.environ.get("G3P_CAMERA_IP", "192.168.144.135")
        control_port = os.environ.get("G3P_CONTROL_PORT", "7878")
        cmd = [
            sys.executable,
            str(control_script),
            "--host",
            camera_ip,
            "--port",
            str(control_port),
            "--soft-fail",
        ]
        self._last_viewfinder_reset = now
        try:
            log.warning(
                "RTSP has failed %d times; requesting G3P viewfinder restart",
                self._failure_count,
            )
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(8.0, float(os.environ.get("G3P_VIEWFINDER_RESET_TIMEOUT_S", "20"))),
                check=False,
            )
            text = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            if text:
                log.warning("G3P viewfinder restart output: %s", text[-1200:])
            return True
        except Exception as exc:
            log.warning("G3P viewfinder restart failed: %s", exc)
            return True

    def _loop(self) -> None:
        import subprocess

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self.rtsp_url,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={max(1, int(round(self.fps)))},scale={self.width}:{self.height}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        frame_size = self._frame_size()
        while self._running.is_set():
            restarted_viewfinder = False
            try:
                log.info("starting RTSP ffmpeg camera: %s", self.rtsp_url)
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                assert self._proc.stdout is not None
                while self._running.is_set() and self._proc.poll() is None:
                    raw = self._read_exact(self._proc.stdout, frame_size, timeout_s=5.0)
                    if raw is None or len(raw) != frame_size:
                        raise RuntimeError("ffmpeg frame read timed out")
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                    sample = CameraSample(
                        frame,
                        None,
                        self._intrinsics(frame),
                        time.monotonic() * 1000.0,
                        {"source": "rtsp", "rtsp_url": self.rtsp_url},
                    )
                    with self._lock:
                        self._sample = sample
                    self._failure_count = 0
                err = b""
                if self._proc.stderr is not None:
                    try:
                        err = self._proc.stderr.read(4096)
                    except Exception:
                        err = b""
                if err:
                    log.warning("RTSP ffmpeg stderr: %s", err.decode(errors="replace").strip())
            except Exception as exc:
                self._failure_count += 1
                log.warning("RTSP ffmpeg camera error: %s; retrying", exc)
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
                self._proc = None
                restarted_viewfinder = self._maybe_restart_viewfinder()
            finally:
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                self._proc = None
                if restarted_viewfinder:
                    settle_s = max(1.0, float(os.environ.get("G3P_VIEWFINDER_SETTLE_S", "12")))
                    log.warning("waiting %.1fs for G3P RTSP encoder to settle", settle_s)
                    time.sleep(settle_s)
                else:
                    time.sleep(1.0)


class RedWhiteTargetApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.shared = SharedState()
        self.running = threading.Event()
        self.running.set()
        self.flight = FlightTelemetry(args.mavlink, args.baud, mock=args.mock_flight)
        if args.fake_camera:
            self.camera = FakeMarkerCamera(args.width, args.height, args.fps)
        elif args.camera_source == "d455":
            self.camera = D455Camera(
                args.camera_serial,
                args.width,
                args.height,
                args.fps,
                fake=False,
                enable_depth=not args.color_only,
            )
        elif args.camera_source == "gstreamer":
            self.camera = GStreamerJpegCamera(
                args.camera_pipeline,
                args.width,
                args.height,
                args.fps,
                fx=args.camera_fx,
                fy=args.camera_fy,
                ppx=args.camera_ppx,
                ppy=args.camera_ppy,
            )
        elif args.camera_source == "rtsp":
            self.camera = RtspFfmpegCamera(
                args.rtsp_url,
                args.width,
                args.height,
                args.fps,
                fx=args.camera_fx,
                fy=args.camera_fy,
                ppx=args.camera_ppx,
                ppy=args.camera_ppy,
            )
        else:
            self.camera = OpenCVCamera(
                args.video_device,
                args.width,
                args.height,
                args.fps,
                backend=cv2.CAP_V4L2,
                fx=args.camera_fx,
                fy=args.camera_fy,
                ppx=args.camera_ppx,
                ppy=args.camera_ppy,
            )
        self.geolocator = TargetGeolocator(
            [args.camera_roll_deg, args.camera_pitch_deg, args.camera_yaw_deg],
            [args.camera_x_m, args.camera_y_m, args.camera_z_m],
            args.min_depth,
            args.max_depth,
            args.attitude_sigma_deg,
        )
        self.vehicle_detector = YoloVehicleDetector(args)
        self.log_dir = (
            Path(args.session_dir).expanduser()
            if args.session_dir
            else Path(args.record_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.log_dir.mkdir(parents=True, exist_ok=bool(args.session_dir))
        self.event_log = JsonlWriter(self.log_dir / "events.jsonl")
        self.csv_log = CsvSummaryWriter(self.log_dir / "mission_summary.csv")
        self.ros2 = (
            MissionROS2Publisher(args.ros2_compressed_only, args.ros_image_fps, args.ros_jpeg_quality)
            if args.ros2
            else None
        )
        self.auto_return = AutoReturnController(self.flight, args)
        max_dist_px = args.track_gate_ratio * max(args.width, args.height)
        max_age = max(3, int(args.track_ttl_s * args.fps))
        self.tracker = MarkerTracker(
            confirm_hits=args.confirm_hits,
            max_age=max_age,
            max_dist_px=max_dist_px,
            min_iou=args.track_min_iou,
            geo_break_m=args.track_geo_break_m,
        )
        self.emitted_ids: set[str] = set()
        self.target_memory: list[Dict[str, Any]] = []
        self.mission_locked = False
        self.mission_target_id: Optional[str] = None
        self.mission_event: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        self.flight.start()
        self.camera.start()
        threading.Thread(target=self._loop, name="red-white-loop", daemon=True).start()

    def stop(self) -> None:
        self.running.clear()
        self.camera.stop()
        self.flight.stop()
        self.event_log.close()
        self.csv_log.close()
        if self.ros2 is not None:
            self.ros2.stop()

    def _should_emit_once(self, target_id: str) -> bool:
        return target_id not in self.emitted_ids

    def _mission_status(self) -> Dict[str, Any]:
        rtl = self.auto_return.status()
        return {
            "locked": self.mission_locked,
            "target_id": self.mission_target_id,
            "target_event": copy.deepcopy(self.mission_event),
            "auto_rtl": rtl,
            "message": self._mission_message(rtl),
        }

    def _mission_message(self, rtl: Dict[str, Any]) -> str:
        if not self.mission_locked:
            return "waiting for mission target"
        if rtl.get("mode_confirmed"):
            return "target detected; flight controller reports RTL"
        if rtl.get("ack_accepted"):
            return "target detected; RTL command accepted"
        if rtl.get("sent"):
            return "target detected; waiting for RTL confirmation"
        return "target detected; preparing RTL command"
    def _lock_mission(self, event: Dict[str, Any]) -> None:
        self.mission_locked = True
        self.mission_target_id = str(event.get("target_id") or "")
        self.mission_event = copy.deepcopy(event)

    def _stabilize_track_id(self, trk: KalmanTrack) -> str:
        """Reuse an old ID when a temporarily lost target is rediscovered.

        The main tracker handles short misses.  This memory handles longer
        misses by comparing GPS distance when valid, then falling back to pixel
        distance in the current camera frame.
        """
        if not (self.args.enable_gps_reid or self.args.enable_pixel_reid):
            return trk.track_id

        target = trk.target_location or {}
        center = trk.center
        for item in self.target_memory:
            old_target = item.get("target_location") or {}
            if self.args.enable_gps_reid and target.get("valid") and old_target.get("valid"):
                if gps_distance_m(target, old_target) <= self.args.dedupe_radius_m:
                    trk.track_id = str(item["target_id"])
                    item["target_location"] = copy.deepcopy(target)
                    item["center_px"] = center
                    item["last_seen"] = time.monotonic()
                    return trk.track_id
            # Pixel-only re-ID is intentionally disabled by default because it
            # can glue a new physical target to an old ID after the operator
            # moves the camera to the next vehicle.  Use GPS-based memory for
            # once-per-target semantics; the live tracker handles short misses.
            if self.args.enable_pixel_reid:
                old_center = item.get("center_px")
                if old_center is not None:
                    pixel_gate = self.args.reid_pixel_gate_ratio * max(self.args.width, self.args.height)
                    if math.hypot(center[0] - old_center[0], center[1] - old_center[1]) <= pixel_gate:
                        trk.track_id = str(item["target_id"])
                        item["target_location"] = copy.deepcopy(target)
                        item["center_px"] = center
                        item["last_seen"] = time.monotonic()
                        return trk.track_id

        self.target_memory.append(
            {
                "target_id": trk.track_id,
                "target_location": copy.deepcopy(target),
                "center_px": center,
                "last_seen": time.monotonic(),
            }
        )
        self.target_memory = [
            item
            for item in self.target_memory
            if time.monotonic() - float(item.get("last_seen", 0.0)) <= self.args.reid_memory_s
        ]
        return trk.track_id

    def _build_event(
        self,
        target_id: str,
        marker: MarkerDetection,
        vehicle_box: Sequence[int],
        white_ratio: float,
        vehicle_candidate: Optional[VehicleCandidate],
        target: Dict[str, Any],
        flight: Dict[str, Any],
        sample: CameraSample,
    ) -> Dict[str, Any]:
        det = {
            "id": target_id,
            "track_id": target_id,
            "class": "target_vehicle",
            "confidence": round(float(marker.score), 5),
            "bbox": [round(v, 2) for v in xyxy_list(vehicle_box)],
            "vehicle_bbox": [int(v) for v in vehicle_box],
            "marker_bbox": [round(v, 2) for v in marker_bbox_xyxy(marker)],
            "marker_center_px": [round(v, 2) for v in marker.center],
            "red_ratio": round(marker.red_ratio, 4),
            "white_x_ratio": round(marker.white_ratio, 4),
            "x_score": round(marker.x_score, 4),
            "white_vehicle_ratio": round(white_ratio, 4),
            "vehicle_detector": (
                {
                    "source": vehicle_candidate.source,
                    "class": "target_vehicle",
                    "model_class": vehicle_candidate.class_name,
                    "class_id": vehicle_candidate.class_id,
                    "confidence": round(vehicle_candidate.confidence, 5),
                }
                if vehicle_candidate is not None
                else {"source": "marker_geometry", "confidence": 0.0}
            ),
            "target_location": target,
        }
        enriched_target = dict(target)
        if target.get("valid"):
            lat = target.get("latitude")
            lon = target.get("longitude")
            if lat is not None and lon is not None:
                tx, ty = wgs84_to_twd97(float(lat), float(lon))
                enriched_target["twd97_x"] = round(tx, 4)
                enriched_target["twd97_y"] = round(ty, 4)
        det["target_location"] = enriched_target
        return {
            "type": "target_detection",
            "event_id": f"{int(time.time() * 1000)}_{target_id}",
            "target_id": target_id,
            "timestamp": utc_now(),
            "camera_timestamp_ms": sample.camera_timestamp_ms,
            "gps": flight,
            "target_location": enriched_target,
            "detections": [det],
        }

    def _annotate(
        self,
        frame: np.ndarray,
        flight: Dict[str, Any],
        records: list[Dict[str, Any]],
        vehicle_candidates: Optional[list[VehicleCandidate]] = None,
        target_candidates: Optional[list[Dict[str, Any]]] = None,
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        out = frame.copy()
        if self.args.draw_vehicle_candidates and target_candidates:
            for rec in target_candidates:
                marker = rec["marker"]
                x1, y1, x2, y2 = rec["vehicle_box"]
                passed = bool(rec.get("passed_filters"))
                if not passed and not self.args.draw_rejected_marker_candidates:
                    continue
                if (
                    not passed
                    and float(rec.get("white_ratio", 0.0))
                    < self.args.debug_min_vehicle_white_ratio
                ):
                    continue
                color = (255, 220, 80) if passed else (255, 0, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.polylines(out, [marker.quad.astype(np.int32)], True, (0, 255, 0), 2)
                label = (
                    f"{'target cand' if passed else 'marker cand'} "
                    f"M={marker.score:.2f} X={marker.x_score:.2f} "
                    f"R={marker.red_ratio:.2f} Wx={marker.white_ratio:.2f} "
                    f"Wcar={float(rec.get('white_ratio', 0.0)):.2f}"
                )
                cv2.putText(
                    out,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 220, 80),
                    2,
                    cv2.LINE_AA,
                )
        if self.args.draw_raw_vehicle_candidates and vehicle_candidates:
            frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
            debug_candidates = [
                vehicle
                for vehicle in vehicle_candidates
                if vehicle.confidence >= self.args.debug_vehicle_conf
                and vehicle_body_white_ratio(frame, vehicle.bbox)
                >= self.args.debug_min_vehicle_white_ratio
                and self.args.debug_min_vehicle_area_ratio
                <= (
                    max(1.0, float((vehicle.bbox[2] - vehicle.bbox[0]) * (vehicle.bbox[3] - vehicle.bbox[1])))
                    / frame_area
                )
                <= self.args.debug_max_vehicle_area_ratio
            ]
            for vehicle in debug_candidates[: self.args.max_debug_vehicle_boxes]:
                x1, y1, x2, y2 = vehicle.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 170, 0), 2)
                white_ratio = vehicle_body_white_ratio(frame, vehicle.bbox)
                label = f"YOLO {vehicle.class_name} {vehicle.confidence:.2f} W{white_ratio:.2f}"
                cv2.putText(
                    out,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 170, 0),
                    2,
                    cv2.LINE_AA,
                )
        for rec in records:
            marker = rec["marker"]
            vehicle_box = rec["vehicle_box"]
            x1, y1, x2, y2 = vehicle_box
            predicted = int(rec.get("time_since_update", 0)) > 0
            box_color = (180, 220, 255) if predicted else (255, 255, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            cv2.polylines(out, [marker.quad.astype(np.int32)], True, (0, 255, 0), 3)
            cx, cy = [int(v) for v in marker.center]
            cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            target_loc = rec.get("target_location") or {}
            lx = max(5, x1)
            if target_loc.get("valid"):
                lat = target_loc.get("latitude")
                lon = target_loc.get("longitude")
                tx = target_loc.get("twd97_x")
                ty = target_loc.get("twd97_y")
                if tx is None and lat is not None and lon is not None:
                    tx, ty = wgs84_to_twd97(float(lat), float(lon))
                suffix = " tracking" if predicted else ""
                line1 = f"{rec.get('target_id', 'TGT')}  WGS84 {lat:.7f}, {lon:.7f}{suffix}"
                cv2.putText(out, line1, (lx, max(46, y1 - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                if tx is not None:
                    line2 = f"TWD97  E {tx:.2f}  N {ty:.2f}"
                    cv2.putText(out, line2, (lx, max(68, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2, cv2.LINE_AA)
            else:
                suffix = " tracking" if predicted else ""
                label = f"{rec.get('target_id', 'TGT')}  conf={marker.score:.2f}{suffix}"
                cv2.putText(out, label, (lx, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)

        gps_text = "UAV GPS NO FIX"
        if flight.get("latitude") is not None:
            gps_text = (
                f"UAV {flight['latitude']:.7f},{flight['longitude']:.7f} "
                f"ALT {float(flight.get('relative_altitude') or 0):.1f}m "
                f"FIX {flight.get('fix_type', 0)} SAT {flight.get('satellites', 0)}"
            )
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(out, gps_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        if self.args.draw_vehicle_candidates and debug_stats:
            stat_text = (
                f"YOLO vehicles {debug_stats.get('vehicles', 0)} | "
                f"marker candidates {debug_stats.get('marker_candidates', 0)} | "
                f"target candidates {debug_stats.get('passed_candidates', 0)} | "
                f"confirmed {debug_stats.get('confirmed_tracks', 0)}"
            )
            cv2.rectangle(out, (0, 34), (out.shape[1], 64), (0, 0, 0), -1)
            cv2.putText(
                out,
                stat_text,
                (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )
        bottom_y = out.shape[0]
        for rec in reversed(records):
            t = rec.get("target_location") or {}
            if t.get("valid"):
                lat, lon = t.get("latitude"), t.get("longitude")
                tx = t.get("twd97_x")
                ty = t.get("twd97_y")
                if tx is None and lat is not None and lon is not None:
                    tx, ty = wgs84_to_twd97(float(lat), float(lon))
                msg = f"{rec['target_id']}  WGS84 {lat:.7f},{lon:.7f}"
                if tx is not None:
                    msg += f"  |  TWD97 E {tx:.2f} N {ty:.2f}"
            else:
                msg = f"{rec['target_id']}  {t.get('reason', 'geolocation unavailable')}"
            bottom_y -= 36
            cv2.rectangle(out, (0, bottom_y), (out.shape[1], bottom_y + 36), (0, 0, 0), -1)
            cv2.putText(out, msg, (8, bottom_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def _loop(self) -> None:
        last_flight_emit = 0.0
        frame_index = 0
        while self.running.is_set():
            started = time.monotonic()
            sample = self.camera.get()
            if sample is None:
                time.sleep(0.03)
                continue
            frame_index += 1
            flight = self.flight.get()
            vehicles = self.vehicle_detector.detect(sample.color)

            # Filter markers and compute per-detection vehicle data.
            valid_markers: list[MarkerDetection] = []
            valid_vehicle_data: list[
                tuple[float, tuple[int, int, int, int], Optional[VehicleCandidate], Dict[str, Any]]
            ] = []
            candidate_items: list[tuple[MarkerDetection, float, tuple[int, int, int, int], Optional[VehicleCandidate]]] = []
            if self.args.vehicle_roi_first and self.vehicle_detector.enabled:
                h, w = sample.color.shape[:2]
                for vehicle in vehicles[: self.args.max_vehicle_rois]:
                    vx1, vy1, vx2, vy2 = expand_xyxy_abs(
                        vehicle.bbox,
                        w,
                        h,
                        scale=self.args.vehicle_roi_expand,
                    )
                    roi = sample.color[vy1:vy2, vx1:vx2]
                    if roi.size == 0:
                        continue
                    marker_roi = roi
                    marker_scale = max(1.0, float(self.args.marker_roi_upscale))
                    if marker_scale > 1.0:
                        marker_roi = cv2.resize(
                            roi,
                            None,
                            fx=marker_scale,
                            fy=marker_scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    roi_min_area = max(
                        self.args.min_roi_marker_area,
                        int(self.args.min_marker_area * self.args.roi_marker_area_scale),
                    )
                    roi_markers = detect_markers(
                        marker_roi,
                        min_area=roi_min_area,
                        focal_px=sample.intrinsics.fx * marker_scale,
                    )
                    for roi_marker in roi_markers[: self.args.max_markers_per_vehicle]:
                        marker = offset_marker(scale_marker(roi_marker, marker_scale), vx1, vy1)
                        white_ratio = white_vehicle_ratio(sample.color, vehicle.bbox, marker)
                        candidate_items.append((marker, white_ratio, vehicle.bbox, vehicle))
            else:
                markers = detect_markers(
                    sample.color,
                    min_area=self.args.min_marker_area,
                    focal_px=sample.intrinsics.fx,
                )
                for marker in markers[: self.args.max_targets_per_frame]:
                    white_ratio, vehicle_box, vehicle_candidate = match_vehicle_candidate(
                        sample.color,
                        marker,
                        vehicles,
                        min_white_ratio=self.args.min_white_vehicle_ratio,
                    )
                    candidate_items.append((marker, white_ratio, vehicle_box, vehicle_candidate))

            debug_target_candidates: list[Dict[str, Any]] = [
                {
                    "marker": marker,
                    "white_ratio": white_ratio,
                    "vehicle_box": vehicle_box,
                    "vehicle_candidate": vehicle_candidate,
                    "passed_filters": False,
                }
                for marker, white_ratio, vehicle_box, vehicle_candidate in candidate_items[
                    : self.args.max_targets_per_frame
                ]
            ]

            for marker, white_ratio, vehicle_box, vehicle_candidate in candidate_items[: self.args.max_targets_per_frame]:
                if not marker_vehicle_geometry_ok(
                    marker,
                    vehicle_box,
                    min_area_ratio=self.args.min_marker_vehicle_area_ratio,
                    max_area_ratio=self.args.max_marker_vehicle_area_ratio,
                ):
                    continue
                if marker.score < self.args.min_marker_score:
                    continue
                if marker.x_score < self.args.min_x_score:
                    continue
                if marker.red_ratio < self.args.min_red_ratio:
                    continue
                if marker.white_ratio < self.args.min_x_white_ratio:
                    continue
                if white_ratio < self.args.min_white_vehicle_ratio:
                    continue
                # For 60 m RGB-only geolocation, use the red-white marker center
                # as the precision fiducial. The vehicle box still validates and
                # displays the target car, but YOLO box center is less stable.
                target = self.geolocator.locate(
                    sample.depth_m,
                    sample.intrinsics,
                    marker_bbox_xyxy(marker),
                    flight,
                )
                if target.get("valid") and int(flight.get("fix_type") or 0) < self.args.min_gps_fix:
                    target = {**target, "valid": False, "reason": "gps_fix_below_gate"}
                valid_markers.append(marker)
                valid_vehicle_data.append((white_ratio, vehicle_box, vehicle_candidate, target))
                for item in debug_target_candidates:
                    if item["marker"] is marker:
                        item["passed_filters"] = True
                        item["target_location"] = target
                        break

            if False and self.args.white_vehicle_fallback and not valid_markers:
                fallback_items: list[tuple[float, MarkerDetection, float, tuple[int, int, int, int], VehicleCandidate, Dict[str, Any]]] = []
                frame_area = float(sample.color.shape[0] * sample.color.shape[1])
                for vehicle in vehicles[: self.args.max_vehicle_rois]:
                    white_ratio = vehicle_body_white_ratio(sample.color, vehicle.bbox)
                    x1, y1, x2, y2 = [float(v) for v in vehicle.bbox]
                    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(frame_area, 1.0)
                    if vehicle.confidence < self.args.white_vehicle_fallback_conf:
                        continue
                    if white_ratio < self.args.white_vehicle_fallback_white_ratio:
                        continue
                    if area_ratio < self.args.white_vehicle_fallback_min_area_ratio:
                        continue
                    if area_ratio > self.args.white_vehicle_fallback_max_area_ratio:
                        continue
                    marker = vehicle_center_marker(vehicle.bbox, score=vehicle.confidence, white_ratio=white_ratio)
                    target = self.geolocator.locate(sample.depth_m, sample.intrinsics, xyxy_list(vehicle.bbox), flight)
                    if target.get("valid") and int(flight.get("fix_type") or 0) < self.args.min_gps_fix:
                        target = {**target, "valid": False, "reason": "gps_fix_below_gate"}
                    score = float(vehicle.confidence) * 0.65 + min(float(white_ratio) / self.args.white_vehicle_fallback_white_ratio, 1.0) * 0.25 + min(area_ratio / 0.01, 1.0) * 0.10
                    fallback_items.append((score, marker, white_ratio, vehicle.bbox, vehicle, target))
                fallback_items.sort(key=lambda item: item[0], reverse=True)
                for _, marker, white_ratio, vehicle_box, vehicle_candidate, target in fallback_items[:1]:
                    valid_markers.append(marker)
                    valid_vehicle_data.append((white_ratio, vehicle_box, vehicle_candidate, target))
                    debug_target_candidates.append({"marker": marker, "white_ratio": white_ratio, "vehicle_box": vehicle_box, "vehicle_candidate": vehicle_candidate, "passed_filters": True, "target_location": target, "fallback": "white_vehicle"})

            if self.args.roof_red_fallback and not valid_markers:
                roof_items: list[tuple[float, MarkerDetection, float, tuple[int, int, int, int], VehicleCandidate, Dict[str, Any]]] = []
                for vehicle in vehicles[: self.args.max_vehicle_rois]:
                    if vehicle.confidence < self.args.roof_red_vehicle_conf:
                        continue
                    white_ratio = vehicle_body_white_ratio(sample.color, vehicle.bbox)
                    if white_ratio < self.args.min_white_vehicle_ratio:
                        continue
                    roof_result = roof_red_marker_from_vehicle(
                        sample.color,
                        vehicle.bbox,
                        roi_scale=self.args.roof_red_roi_scale,
                        min_red_ratio=self.args.roof_red_min_ratio,
                        min_red_pixels=self.args.roof_red_min_pixels,
                        red_h1_max=self.args.roof_red_h1_max,
                        red_h2_min=self.args.roof_red_h2_min,
                        red_s_min=self.args.roof_red_s_min,
                        red_v_min=self.args.roof_red_v_min,
                        min_marker_score=self.args.min_marker_score,
                        min_x_score=self.args.min_x_score,
                        min_x_white_ratio=self.args.min_x_white_ratio,
                    )
                    if roof_result is None:
                        continue
                    marker, red_ratio, _roof_roi = roof_result
                    target = self.geolocator.locate(
                        sample.depth_m,
                        sample.intrinsics,
                        marker_bbox_xyxy(marker),
                        flight,
                    )
                    if target.get("valid") and int(flight.get("fix_type") or 0) < self.args.min_gps_fix:
                        target = {**target, "valid": False, "reason": "gps_fix_below_gate"}
                    score = (
                        float(vehicle.confidence) * 0.50
                        + min(float(white_ratio) / max(self.args.min_white_vehicle_ratio, 1e-6), 1.0) * 0.20
                        + min(float(red_ratio) / max(self.args.roof_red_min_ratio, 1e-6), 1.0) * 0.30
                    )
                    roof_items.append((score, marker, white_ratio, vehicle.bbox, vehicle, target))
                roof_items.sort(key=lambda item: item[0], reverse=True)
                for _, marker, white_ratio, vehicle_box, vehicle_candidate, target in roof_items[:1]:
                    valid_markers.append(marker)
                    valid_vehicle_data.append((white_ratio, vehicle_box, vehicle_candidate, target))
                    debug_target_candidates.append({
                        "marker": marker,
                        "white_ratio": white_ratio,
                        "vehicle_box": vehicle_box,
                        "vehicle_candidate": vehicle_candidate,
                        "passed_filters": True,
                        "target_location": target,
                        "fallback": "roof_red",
                    })

            # Kalman + Hungarian tracker returns only confirmed tracks.
            confirmed_tracks = self.tracker.step(valid_markers, valid_vehicle_data)
            if not self.args.draw_predicted_tracks:
                display_tracks = [trk for trk in confirmed_tracks if trk.time_since_update == 0]
            elif valid_markers:
                display_tracks = [
                    trk
                    for trk in confirmed_tracks
                    if trk.time_since_update <= self.args.display_hold_frames_with_detection
                ]
            else:
                display_tracks = [
                    trk for trk in confirmed_tracks if trk.time_since_update <= self.args.display_hold_frames
                ]

            records: list[Dict[str, Any]] = []
            current_event = None
            for trk in display_tracks:
                stable_id = self._stabilize_track_id(trk)
                if (
                    self.args.mission_latch_once
                    and self.mission_locked
                    and stable_id != self.mission_target_id
                ):
                    continue
                display_box = trk.vehicle_box if trk.time_since_update == 0 else trk.predicted_vehicle_box
                record = {
                    "target_id": stable_id,
                    "marker": trk.marker,
                    "vehicle_box": display_box,
                    "vehicle_candidate": trk.vehicle_candidate,
                    "white_ratio": trk.white_ratio,
                    "target_location": trk.target_location,
                    "time_since_update": trk.time_since_update,
                }
                records.append(record)
                event = self._build_event(
                    stable_id,
                    trk.marker,
                    trk.vehicle_box,
                    trk.white_ratio,
                    trk.vehicle_candidate,
                    trk.target_location,
                    flight,
                    sample,
                )
                current_event = self.mission_event if self.mission_locked else event
                if self.args.emit_updates or self._should_emit_once(stable_id):
                    if self.args.mission_latch_once and self.mission_locked:
                        continue
                    if self.args.require_vehicle_model_match and trk.vehicle_candidate is None:
                        continue
                    # Strict race mode: RTL/event is allowed only for a real red-background white-X marker on a white vehicle.
                    # This prevents white-only vehicles from becoming mission targets at venues with many white cars.
                    if trk.marker.red_ratio < self.args.min_red_ratio:
                        continue
                    if trk.marker.white_ratio < self.args.min_x_white_ratio:
                        continue
                    if trk.marker.x_score < self.args.min_x_score:
                        continue
                    if trk.white_ratio < self.args.min_white_vehicle_ratio:
                        continue
                    if trk.hits < self.args.mission_confirm_hits:
                        continue
                    self.emitted_ids.add(stable_id)
                    rtl_status = self.auto_return.trigger_for_event(event)
                    event["auto_rtl"] = rtl_status
                    if self.args.mission_latch_once:
                        self._lock_mission(event)
                        current_event = self.mission_event
                        socketio.emit("mission_status", self._mission_status(), namespace="/events")
                    self.event_log.write(event)
                    if self.ros2 is not None:
                        self.ros2.publish_event(event)
                    self.csv_log.write_target(event, flight, rtl_status)
                    socketio.emit("target", event, namespace="/events")
                    socketio.emit("detection", event, namespace="/events")
                    log.info(
                        "target %s marker=%.2f vehicle_white=%.2f gps=%s",
                        stable_id,
                        trk.marker.score,
                        trk.white_ratio,
                        (
                            f"{trk.target_location.get('latitude'):.7f},"
                            f"{trk.target_location.get('longitude'):.7f}"
                            if trk.target_location.get("valid")
                            else trk.target_location.get("reason")
                        ),
                    )

            annotated = self._annotate(
                sample.color,
                flight,
                records,
                vehicles,
                debug_target_candidates,
                {
                    "vehicles": len(vehicles),
                    "marker_candidates": len(candidate_items),
                    "passed_candidates": len(valid_markers),
                    "confirmed_tracks": len(display_tracks),
                },
            )
            self.shared.publish_frame(
                annotated,
                self.args.jpeg_quality,
                self.args.ground_preview_scale,
                self.args.ground_preview_fps,
            )
            status_records = [
                {
                    "target_id": rec["target_id"],
                    "bbox": list(rec["vehicle_box"]),
                    "marker_bbox": marker_bbox_xyxy(rec["marker"]),
                    "vehicle_bbox": list(rec["vehicle_box"]),
                    "score": rec["marker"].score,
                    "white_vehicle_ratio": rec["white_ratio"],
                    "vehicle_detector": (
                        asdict(rec["vehicle_candidate"])
                        if rec.get("vehicle_candidate") is not None
                        else {"source": "marker_geometry"}
                    ),
                    "track_age_frames": int(rec.get("time_since_update", 0)),
                    "target_location": rec["target_location"],
                }
                for rec in records
            ]
            self.shared.update_status(
                flight,
                status_records,
                current_event,
                self._mission_status(),
            )
            if time.monotonic() - last_flight_emit >= 1.0:
                last_flight_emit = time.monotonic()
                self.csv_log.write_flight(flight, self.auto_return.status())
                socketio.emit("gps", flight, namespace="/events")
                socketio.emit("mission_status", self._mission_status(), namespace="/events")
            if self.ros2 is not None:
                self.ros2.publish_frame(sample, annotated, status_records, flight, frame_index)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, 1.0 / max(self.args.fps, 1) - elapsed))


SHARED: Optional[SharedState] = None


def mjpeg_stream() -> Iterable[bytes]:
    seen = -1
    while True:
        if SHARED is None:
            time.sleep(0.1)
            continue
        data, revision = SHARED.get_jpeg()
        if data and revision != seen:
            seen = revision
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        else:
            time.sleep(0.03)


@app.route("/video")
@app.route("/video/cam0")
def video():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(SHARED.status() if SHARED else {"ok": False})


@app.route("/health")
def health():
    return jsonify({"ok": True, "timestamp": utc_now()})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-white X target geolocation server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--camera-serial", default="234322304658")
    p.add_argument(
        "--camera-source",
        choices=("d455", "v4l2", "gstreamer", "rtsp"),
        default="d455",
        help="d455 keeps the old RealSense path; rtsp is for G3P/network cameras",
    )
    p.add_argument("--video-device", default="/dev/video0", help="V4L2 device or numeric OpenCV camera index")
    p.add_argument("--rtsp-url", default="rtsp://192.168.144.135/live")
    p.add_argument("--camera-pipeline", default="", help="GStreamer pipeline when --camera-source=gstreamer")
    p.add_argument("--camera-fx", type=float, default=0.0, help="optional focal length fx in pixels for non-depth camera")
    p.add_argument("--camera-fy", type=float, default=0.0, help="optional focal length fy in pixels for non-depth camera")
    p.add_argument("--camera-ppx", type=float, default=0.0, help="optional principal point x in pixels")
    p.add_argument("--camera-ppy", type=float, default=0.0, help="optional principal point y in pixels")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fake-camera", action="store_true")
    p.add_argument("--color-only", action="store_true")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--ground-preview-scale", type=float, default=1.0)
    p.add_argument("--ground-preview-fps", type=float, default=0.0)
    p.add_argument("--vehicle-model", default=DEFAULT_VEHICLE_MODEL)
    p.add_argument("--disable-vehicle-model", action="store_true")
    p.add_argument("--vehicle-classes", default="car,van,truck,bus")
    p.add_argument("--vehicle-conf", type=float, default=0.18)
    p.add_argument("--vehicle-iou", type=float, default=0.55)
    p.add_argument("--vehicle-imgsz", type=int, default=1280)
    p.add_argument("--vehicle-device", default="0")
    p.add_argument("--vehicle-half", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vehicle-tiles", type=int, default=1)
    p.add_argument("--vehicle-tile-overlap", type=float, default=0.25)
    p.add_argument("--vehicle-tile-nms-iou", type=float, default=0.45)
    p.add_argument("--draw-vehicle-candidates", action="store_true")
    p.add_argument("--draw-rejected-marker-candidates", action="store_true")
    p.add_argument("--draw-raw-vehicle-candidates", action="store_true")
    p.add_argument("--debug-vehicle-conf", type=float, default=0.80)
    p.add_argument("--debug-min-vehicle-white-ratio", type=float, default=0.55)
    p.add_argument("--debug-min-vehicle-area-ratio", type=float, default=0.0002)
    p.add_argument("--debug-max-vehicle-area-ratio", type=float, default=0.25)
    p.add_argument("--max-debug-vehicle-boxes", type=int, default=20)
    p.add_argument("--min-marker-area", type=int, default=260)
    p.add_argument("--min-marker-score", type=float, default=0.68)
    p.add_argument("--min-x-score", type=float, default=0.55)
    p.add_argument("--min-red-ratio", type=float, default=0.42)
    p.add_argument("--min-x-white-ratio", type=float, default=0.14)
    p.add_argument("--min-white-vehicle-ratio", type=float, default=0.30)
    p.add_argument("--white-vehicle-fallback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--white-vehicle-fallback-conf", type=float, default=0.45)
    p.add_argument("--white-vehicle-fallback-white-ratio", type=float, default=0.55)
    p.add_argument("--white-vehicle-fallback-min-area-ratio", type=float, default=0.0005)
    p.add_argument("--white-vehicle-fallback-max-area-ratio", type=float, default=0.08)
    p.add_argument("--roof-red-fallback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--roof-red-vehicle-conf", type=float, default=0.60)
    p.add_argument("--roof-red-roi-scale", type=float, default=1.80)
    p.add_argument("--roof-red-min-ratio", type=float, default=0.015)
    p.add_argument("--roof-red-min-pixels", type=int, default=8)
    p.add_argument("--roof-red-h1-max", type=int, default=18)
    p.add_argument("--roof-red-h2-min", type=int, default=155)
    p.add_argument("--roof-red-s-min", type=int, default=22)
    p.add_argument("--roof-red-v-min", type=int, default=35)
    p.add_argument("--min-marker-vehicle-area-ratio", type=float, default=0.025)
    p.add_argument("--max-marker-vehicle-area-ratio", type=float, default=0.45)
    p.add_argument("--vehicle-roi-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vehicle-roi-expand", type=float, default=1.20)
    p.add_argument("--max-vehicle-rois", type=int, default=8)
    p.add_argument("--max-markers-per-vehicle", type=int, default=4)
    p.add_argument("--min-roi-marker-area", type=int, default=20)
    p.add_argument("--roi-marker-area-scale", type=float, default=0.25)
    p.add_argument("--marker-roi-upscale", type=float, default=4.5)
    p.add_argument("--confirm-hits", type=int, default=1)
    p.add_argument("--mission-confirm-hits", type=int, default=2)
    p.add_argument("--require-vehicle-model-match", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track-gate-ratio", type=float, default=0.12)
    p.add_argument("--track-min-iou", type=float, default=0.05)
    p.add_argument("--track-geo-break-m", type=float, default=4.0)
    p.add_argument("--track-ttl-s", type=float, default=0.35)
    p.add_argument("--display-hold-s", type=float, default=0.0)
    p.add_argument("--display-hold-with-detection-s", type=float, default=0.0)
    p.add_argument("--draw-predicted-tracks", action="store_true")
    p.add_argument("--mission-latch-once", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reid-memory-s", type=float, default=90.0)
    p.add_argument("--reid-pixel-gate-ratio", type=float, default=0.20)
    p.add_argument("--enable-gps-reid", action="store_true")
    p.add_argument("--enable-pixel-reid", action="store_true")
    p.add_argument("--max-targets-per-frame", type=int, default=3)
    p.add_argument("--emit-updates", action="store_true", help="emit target every frame instead of first sighting only")
    p.add_argument("--dedupe-radius-m", type=float, default=3.0)
    p.add_argument("--ros2", action="store_true", help="publish ROS 2 topics for official ros2 bag record")
    p.add_argument("--ros2-compressed-only", action="store_true", default=True)
    p.add_argument("--ros-image-fps", type=float, default=1.0)
    p.add_argument("--ros-jpeg-quality", type=int, default=60)
    p.add_argument("--ros2-warmup", type=float, default=3.0)
    p.add_argument("--auto-rtl", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--auto-rtl-require-valid-target", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rtl-mode-param1", type=float, default=1.0)
    p.add_argument("--rtl-custom-mode", type=float, default=6.0)
    p.add_argument("--rtl-target-system", type=int, default=1)
    p.add_argument("--rtl-target-component", type=int, default=1)
    p.add_argument("--rtl-command-repeats", type=int, default=3)
    p.add_argument("--rtl-command-interval-s", type=float, default=0.20)
    p.add_argument("--rtl-confirm-timeout-s", type=float, default=10.0)
    p.add_argument(
        "--min-gps-fix",
        type=int,
        default=5,
        help="minimum MAVLink GPS fix_type for valid target geolocation; 5=RTK FLOAT, 6=RTK FIXED",
    )
    p.add_argument("--mavlink", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mock-flight", action="store_true")
    p.add_argument("--min-depth", type=float, default=0.4)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--attitude-sigma-deg", type=float, default=1.0)
    p.add_argument("--camera-roll-deg", type=float, default=0.0)
    p.add_argument("--camera-pitch-deg", type=float, default=-90.0)
    p.add_argument("--camera-yaw-deg", type=float, default=0.0)
    p.add_argument("--camera-x-m", type=float, default=0.0)
    p.add_argument("--camera-y-m", type=float, default=0.0)
    p.add_argument("--camera-z-m", type=float, default=0.0)
    p.add_argument("--record-dir", default="/home/hrc/recordings/red_white_target")
    p.add_argument("--session-dir", default="")
    args = p.parse_args()
    args.display_hold_frames = max(0, int(round(args.display_hold_s * args.fps)))
    args.display_hold_frames_with_detection = max(
        0, int(round(args.display_hold_with_detection_s * args.fps))
    )
    return args


def main() -> int:
    global SHARED
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app_state = RedWhiteTargetApp(args)
    SHARED = app_state.shared

    def shutdown(_signum, _frame):
        log.info("shutdown requested")
        app_state.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if args.ros2 and args.ros2_warmup > 0:
        log.info("waiting %.1f seconds for ROS 2 recorder discovery", args.ros2_warmup)
        time.sleep(args.ros2_warmup)
    app_state.start()
    log.info("mission-target server listening on %s:%d", args.host, args.port)
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
