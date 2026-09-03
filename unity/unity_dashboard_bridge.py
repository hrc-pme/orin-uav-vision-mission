#!/usr/bin/env python3
"""ROS 2 data bridge for the Unity UAV Dashboard.

In live outdoor mode, a confirmed target may switch the selected autopilot to
GUIDED and send a global position target after all configured safety checks.
"""

import argparse
import base64
import copy
import glob
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import (
    BatteryState,
    CameraInfo,
    CompressedImage,
    Image,
    Imu,
    NavSatFix,
    NavSatStatus,
)
from std_msgs.msg import String


DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "live",
    "publish_rate_hz": 5,
    "camera_index": 0,
    "camera_source": "realsense",
    "use_realsense": True,
    "realsense_serial": "",
    "rtsp_url": "rtsp://192.168.144.135/live",
    "camera_fx": 0.0,
    "camera_fy": 0.0,
    "camera_ppx": 0.0,
    "camera_ppy": 0.0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "enable_depth": False,
    "publish_depth": False,
    "publish_camera_info": False,
    "publish_camera_info_when_subscribed": False,
    "publish_d455_imu": False,
    "depth_publish_rate_hz": 1.0,
    "engine_path": "/home/hrc/Downloads/VisDrone_train1_5090best.engine",
    "fallback_model_path": "/home/hrc/Downloads/VisDrone_train1_5090best.pt",
    "detector_backend": "ultralytics_engine",
    "tracker": "botsort.yaml",
    "confidence_threshold": 0.25,
    "max_candidates": 3,
    # "white" preserves the original white-car mission.  "any" accepts every
    # detection whose model class is listed in target_classes.
    "target_color_mode": "white",
    "target_classes": ["car"],
    # 0 means reuse white_confirm_frames for backward compatibility.
    "target_confirm_frames": 0,
    "target_streak_hold_s": 5.0,
    "target_streak_match_iou": 0.35,
    "candidate_id_prefix": "white-car",
    "continuous_candidate_geolocation": False,
    "white_saturation_max": 65,
    "white_value_min": 165,
    "white_rgb_min": 145,
    "white_rgb_delta_max": 55,
    "white_roi_mean_saturation_max": 95,
    "white_roi_mean_value_min": 95,
    "white_min_pixels": 60,
    "red_veto_ratio_threshold": 0.06,
    "white_ratio_max": 0.82,
    "car_aspect_ratio_min": 0.35,
    "car_aspect_ratio_max": 3.2,
    "car_min_edge_density": 0.025,
    "white_ratio_threshold": 0.30,
    "white_confirm_frames": 3,
    "draw_non_white_detections": False,
    "detection_display_max_age_s": 0.7,
    # Visual-only persistence smooths brief detector misses. Held boxes are
    # never used as fresh navigation measurements.
    "visual_detection_hold_s": 15.0,
    "visual_detection_deduplicate_iou": 0.35,
    "visual_tracking_min_score": 0.55,
    "visual_tracking_search_expansion": 1.0,
    "visual_tracking_miss_frames": 2,
    "visual_display_max_detections": 3,
    "draw_navigation_safe_region": False,
    "auto_verification_min_edge_margin_ratio": 0.0,
    "candidate_jpeg_quality": 70,
    "include_candidate_image_base64": True,
    "candidate_image_max_width": 320,
    "candidate_image_max_height": 240,
    "candidate_image_resend_interval_s": 5.0,
    "jpeg_quality": 80,
    "publish_image_max_width": 400,
    "publish_image_max_height": 225,
    "publish_raw_compressed": True,
    "publish_raw_compressed_when_subscribed": False,
    "raw_jpeg_quality": 70,
    "raw_publish_image_max_width": 0,
    "raw_publish_image_max_height": 0,
    "publish_annotated_compressed": True,
    "image_qos_depth": 10,
    "default_latitude": 24.7998907,
    "default_longitude": 121.0296557,
    "default_altitude": 52.7,
    "meters_per_pixel_at_center": 0.08,
    "mock_gps": False,
    "mock_imu": False,
    "mavlink_enabled": True,
    "mavlink_port": "auto",
    "mavlink_baud": 115200,
    "mavlink_bauds": [921600, 115200, 57600],
    "mavlink_heartbeat_timeout_s": 2.0,
    "mavlink_request_message_rates": True,
    # Zero means auto-select.  The selected heartbeat must advertise
    # CUSTOM_MODE_ENABLED; this avoids AP6's second, non-controlling heartbeat.
    "mavlink_target_system": 0,
    "mavlink_target_component": 0,
    "gps_min_fix_type": 3,
    "gps_stale_timeout_s": 3.0,
    "gps_display_hold_s": 10.0,
    "camera_mount_rpy_deg": [0.0, -90.0, 0.0],
    "camera_offset_frd_m": [0.0, 0.0, 0.0],
    "target_min_depth_m": 0.4,
    "target_max_depth_m": 20.0,
    "attitude_error_deg": 3.0,
    "geolocation_filter_window": 5,
    "enable_flight_control": False,
    "simulate_flight_commands": False,
    "allowed_flight_modes": ["GUIDED"],
    "auto_switch_to_guided": False,
    "guided_mode_name": "GUIDED",
    "auto_switch_guided_from_modes": ["AUTO", "LOITER", "POSHOLD", "GUIDED"],
    "mode_change_timeout_s": 5.0,
    "minimum_flight_altitude_m": 5.0,
    "minimum_gps_satellites": 8,
    "maximum_gps_eph_m": 5.0,
    "maximum_target_error_m": 15.0,
    "maximum_goto_distance_m": 100.0,
    "target_max_age_s": 3.0,
    "candidate_archive_age_s": 6.0,
    "target_confirm_seconds": 3.0,
    "candidate_display_hold_s": 10.0,
    "target_position_require_gps": True,
    "command_target_match_distance_m": 5.0,
    "goto_ground_speed_mps": 3.0,
    "goto_relative_altitude_m": 20.0,
    "arrival_radius_m": 3.0,
    "arrival_altitude_tolerance_m": 3.0,
    "arrival_confirm_seconds": 2,
    "goto_timeout_s": 120.0,
    "class_name_override": "car",
}


def white_vehicle_score(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], config: Dict[str, Any]
) -> Tuple[float, int, float, float, float, float]:
    """Return robust white-body score for a detected car bbox.

    The mask requires both HSV low saturation/high value and near-neutral RGB
    channels.  This rejects most red/black cars with specular highlights while
    still accepting real white body paint in a small aerial bbox.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0

    margin_x = max(1, int(bw * 0.18))
    margin_y = max(1, int(bh * 0.18))
    xa, xb = x1 + margin_x, x2 - margin_x
    ya, yb = y1 + margin_y, y2 - margin_y
    if xb <= xa or yb <= ya:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0

    roi = frame[ya:yb, xa:xb]
    if roi.size < 9:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    edges = cv2.Canny(gray, 40, 120)
    edge_density = int(np.count_nonzero(edges)) / float(edges.size)

    b, g, r = cv2.split(roi)
    min_bgr = np.minimum(np.minimum(b, g), r)
    max_bgr = np.maximum(np.maximum(b, g), r)
    neutral_mask = (
        (min_bgr >= int(config["white_rgb_min"]))
        & ((max_bgr - min_bgr) <= int(config["white_rgb_delta_max"]))
    )
    hsv_mask = (
        (s <= int(config["white_saturation_max"]))
        & (v >= int(config["white_value_min"]))
    )
    white_mask = hsv_mask & neutral_mask

    red_mask = (
        (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)))
        & (s >= 80)
        & (v >= 60)
    )

    white_pixels = int(np.count_nonzero(white_mask))
    white_ratio = white_pixels / float(white_mask.size)
    red_ratio = int(np.count_nonzero(red_mask)) / float(red_mask.size)
    return white_ratio, white_pixels, mean_s, mean_v, red_ratio, edge_density


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, ...]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotation_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def wgs84_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_378_137.0
    mean_lat = math.radians((lat1 + lat2) * 0.5)
    north = math.radians(lat2 - lat1) * radius
    east = math.radians(lon2 - lon1) * radius * max(math.cos(mean_lat), 1e-6)
    return math.hypot(north, east)


def wgs84_to_twd97(latitude: float, longitude: float) -> Tuple[float, float]:
    """Convert WGS84 to Taiwan TWD97 / TM2 zone 121 (EPSG:3826)."""
    a = 6378137.0
    b = 6356752.314245
    lon0 = math.radians(121.0)
    k0, false_easting = 0.9999, 250000.0
    e2 = 1.0 - (b * b) / (a * a)
    ep2 = e2 / (1.0 - e2)
    phi, lam = math.radians(latitude), math.radians(longitude)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    tan_phi = math.tan(phi)
    n = a / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = ep2 * cos_phi * cos_phi
    aa = cos_phi * (lam - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024)
        * math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
        - (35 * e2**3 / 3072) * math.sin(6 * phi)
    )
    easting = false_easting + k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    )
    northing = k0 * (
        m
        + n
        * tan_phi
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return easting, northing


class RealSenseSource:
    """Continuously captures the latest D455i RGB/depth/IMU sample."""

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._pipeline = None
        self._frame: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._intrinsics: Optional[Dict[str, float]] = None
        self._imu: Dict[str, Tuple[float, float, float]] = {}
        self.connected = False

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="d455i", daemon=True)
        self._thread.start()

    @staticmethod
    def _motion(frame: Any) -> Tuple[float, float, float]:
        value = frame.as_motion_frame().get_motion_data()
        return float(value.x), float(value.y), float(value.z)

    def _run(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            self.logger.error(f"pyrealsense2 unavailable: {exc}")
            return
        want_imu = bool(self.config.get("publish_d455_imu", False))
        hidraw_devices = glob.glob("/dev/hidraw*")
        if (
            want_imu
            and hidraw_devices
            and not any(os.access(path, os.R_OK) for path in hidraw_devices)
        ):
            self.logger.error(
                "D455i HID/IMU permission denied; run ./install_realsense_permissions.sh "
                "once, then unplug/replug the camera"
            )
        while self._running.is_set():
            pipeline = rs.pipeline()
            cfg = rs.config()
            serial = str(self.config.get("realsense_serial", "")).strip()
            if serial:
                cfg.enable_device(serial)
            width = int(self.config["camera_width"])
            height = int(self.config["camera_height"])
            fps = int(self.config["camera_fps"])
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if bool(self.config["enable_depth"]):
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            imu_enabled = want_imu
            try:
                if imu_enabled:
                    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
                    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                profile = pipeline.start(cfg)
            except Exception as exc:
                self.logger.warning(f"D455i stream open failed ({exc}); retry RGB only")
                try:
                    pipeline.stop()
                except Exception:
                    pass
                pipeline, cfg = rs.pipeline(), rs.config()
                if serial:
                    cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                if bool(self.config["enable_depth"]):
                    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                try:
                    profile = pipeline.start(cfg)
                except Exception as camera_exc:
                    self.logger.error(f"D455i open failed: {camera_exc}; retry in 2 s")
                    time.sleep(2)
                    continue
                imu_enabled = False
            self._pipeline = pipeline
            align = rs.align(rs.stream.color) if bool(self.config["enable_depth"]) else None
            depth_scale = (
                profile.get_device().first_depth_sensor().get_depth_scale()
                if bool(self.config["enable_depth"])
                else None
            )
            self.connected = True
            self.logger.info(
                f"D455i connected: {width}x{height}@{fps}, "
                f"depth={depth_scale is not None}, imu={imu_enabled}"
            )
            try:
                while self._running.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=3000)
                    processed = align.process(frames) if align is not None else frames
                    color = processed.get_color_frame()
                    depth = processed.get_depth_frame() if align is not None else None
                    if not color:
                        continue
                    frame = np.asanyarray(color.get_data()).copy()
                    depth_m = None
                    if depth and depth_scale is not None:
                        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
                    intr = color.profile.as_video_stream_profile().intrinsics
                    imu_update: Dict[str, Tuple[float, float, float]] = {}
                    if imu_enabled:
                        accel = frames.first_or_default(rs.stream.accel)
                        gyro = frames.first_or_default(rs.stream.gyro)
                        if accel:
                            imu_update["linear_acceleration"] = self._motion(accel)
                        if gyro:
                            imu_update["angular_velocity"] = self._motion(gyro)
                    with self._lock:
                        self._frame = frame
                        self._depth = depth_m
                        self._intrinsics = {
                            "width": intr.width,
                            "height": intr.height,
                            "fx": intr.fx,
                            "fy": intr.fy,
                            "ppx": intr.ppx,
                            "ppy": intr.ppy,
                        }
                        self._imu.update(imu_update)
            except Exception as exc:
                if self._running.is_set():
                    self.logger.error(f"D455i capture lost: {exc}; reconnecting")
            finally:
                self.connected = False
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]], Dict[str, Any]]:
        with self._lock:
            return (
                None if self._frame is None else self._frame.copy(),
                None if self._depth is None else self._depth.copy(),
                copy.deepcopy(self._intrinsics),
                copy.deepcopy(self._imu),
            )

    def stop(self) -> None:
        self._running.clear()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        if hasattr(self, "_thread"):
            self._thread.join(timeout=4)


class RtspFfmpegSource:
    """Continuously captures RGB frames from the G3P RTSP viewfinder."""

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._process: Any = None
        self._frame: Optional[np.ndarray] = None
        self._intrinsics: Optional[Dict[str, float]] = None
        self.connected = False

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="g3p-rtsp", daemon=True)
        self._thread.start()

    def _frame_size(self) -> int:
        return int(self.config["camera_width"]) * int(self.config["camera_height"]) * 3

    def _intrinsics_for(self, frame: np.ndarray) -> Dict[str, float]:
        height, width = frame.shape[:2]
        fx = float(self.config.get("camera_fx", 0.0) or 0.0)
        fy = float(self.config.get("camera_fy", 0.0) or 0.0)
        ppx = float(self.config.get("camera_ppx", 0.0) or 0.0)
        ppy = float(self.config.get("camera_ppy", 0.0) or 0.0)
        if fx <= 0:
            fx = 0.9 * float(width)
        if fy <= 0:
            fy = fx
        if ppx <= 0:
            ppx = 0.5 * float(width)
        if ppy <= 0:
            ppy = 0.5 * float(height)
        return {"width": width, "height": height, "fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy}

    def _read_exact(self, pipe: Any, size: int, timeout_s: float = 5.0) -> Optional[bytes]:
        import select

        fd = pipe.fileno()
        chunks: List[bytes] = []
        remaining = size
        deadline = time.monotonic() + timeout_s
        while remaining > 0 and self._running.is_set():
            wait_s = deadline - time.monotonic()
            if wait_s <= 0:
                return None
            readable, _, _ = select.select([fd], [], [], min(wait_s, 0.25))
            if not readable:
                continue
            chunk = os.read(fd, remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _run(self) -> None:
        import subprocess

        url = str(self.config.get("rtsp_url", "")).strip()
        width = int(self.config["camera_width"])
        height = int(self.config["camera_height"])
        fps = int(self.config["camera_fps"])
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-avioflags",
            "direct",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-rtbufsize",
            "100k",
            "-i",
            url,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={max(1, fps)},scale={width}:{height}",
            "-vsync",
            "drop",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        frame_size = self._frame_size()
        while self._running.is_set():
            try:
                self.logger.info(f"G3P RTSP capture starting: {url}")
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                assert self._process.stdout is not None
                self.connected = True
                while self._running.is_set() and self._process.poll() is None:
                    raw = self._read_exact(self._process.stdout, frame_size)
                    if raw is None or len(raw) != frame_size:
                        raise RuntimeError("ffmpeg frame read timed out")
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                    with self._lock:
                        self._frame = frame
                        self._intrinsics = self._intrinsics_for(frame)
            except Exception as exc:
                if self._running.is_set():
                    self.logger.error(f"G3P RTSP capture lost: {exc}; reconnecting")
            finally:
                self.connected = False
                proc = self._process
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                self._process = None
                time.sleep(1)

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]], Dict[str, Any]]:
        with self._lock:
            return (
                None if self._frame is None else self._frame.copy(),
                None,
                copy.deepcopy(self._intrinsics),
                {},
            )

    def stop(self) -> None:
        self._running.clear()
        proc = self._process
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=4)


class MavlinkTelemetryReader:
    """Reads telemetry and sends flight commands to one selected autopilot."""

    MESSAGE_RATES = {
        "GLOBAL_POSITION_INT": 10,
        "GPS_RAW_INT": 2,
        "GPS2_RAW": 2,
        "ATTITUDE": 20,
        "HIGHRES_IMU": 20,
        "VFR_HUD": 5,
        "SYS_STATUS": 2,
        "BATTERY_STATUS": 2,
    }

    def __init__(self, config: Dict[str, Any], logger: Any, message_callback: Any = None):
        self.config = config
        self.logger = logger
        self._message_callback = message_callback
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._connection = None
        self._target_system: Optional[int] = None
        self._target_component: Optional[int] = None
        self._state: Dict[str, Any] = {
            "connected": False,
            "port": None,
            "baud": None,
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "relative_altitude": None,
            "fix_type": 0,
            "satellites": 0,
            "eph_m": None,
            "epv_m": None,
            "gps_update_monotonic": 0.0,
            "gps1": {
                "fix_type": 0,
                "satellites": 0,
                "eph_m": None,
                "epv_m": None,
                "latitude": None,
                "longitude": None,
                "altitude": None,
            },
            "gps2": {
                "fix_type": 0,
                "satellites": 0,
                "eph_m": None,
                "epv_m": None,
                "latitude": None,
                "longitude": None,
                "altitude": None,
            },
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "rollspeed": 0.0,
            "pitchspeed": 0.0,
            "yawspeed": 0.0,
            "xacc": None,
            "yacc": None,
            "zacc": None,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "voltage_v": None,
            "current_a": None,
            "battery_remaining_pct": None,
            "armed": False,
            "flight_mode": None,
            "target_system": None,
            "target_component": None,
            "last_command_ack_command": None,
            "last_command_ack_result": None,
            "last_command_ack_progress": None,
            "last_command_ack_monotonic": 0.0,
            "last_status_text": None,
            "last_status_text_monotonic": 0.0,
        }

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="mavlink", daemon=True)
        self._thread.start()

    def _ports(self) -> List[str]:
        configured = str(self.config.get("mavlink_port", "auto"))
        if configured.lower() != "auto":
            return [configured]
        ports = glob.glob("/dev/serial/by-id/*")
        ports += glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        ports += glob.glob("/dev/ttyTHS*") + glob.glob("/dev/ttyAMA*")
        return list(dict.fromkeys(sorted(ports)))

    def _bauds(self) -> List[int]:
        configured = self.config.get("mavlink_bauds")
        if isinstance(configured, list) and configured:
            return [int(value) for value in configured]
        return [int(self.config["mavlink_baud"])]

    def _request_rates(self, connection: Any, mavutil: Any) -> None:
        if not bool(self.config.get("mavlink_request_message_rates", True)):
            return
        for name, hz in self.MESSAGE_RATES.items():
            message_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if message_id is None:
                continue
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                int(1_000_000 / hz),
                0,
                0,
                0,
                0,
                0,
            )

    @staticmethod
    def _message_source(message: Any) -> Tuple[int, int]:
        try:
            return int(message.get_srcSystem()), int(message.get_srcComponent())
        except Exception:
            return 0, 0

    def _heartbeat_is_control_autopilot(self, heartbeat: Any, mavutil: Any) -> bool:
        source_system, source_component = self._message_source(heartbeat)
        configured_system = int(self.config.get("mavlink_target_system", 0) or 0)
        configured_component = int(self.config.get("mavlink_target_component", 0) or 0)
        if configured_system > 0 and source_system != configured_system:
            return False
        if configured_component > 0 and source_component != configured_component:
            return False
        if int(getattr(heartbeat, "autopilot", -1)) in (
            int(mavutil.mavlink.MAV_AUTOPILOT_INVALID),
        ):
            return False
        # The bad AP6 heartbeat in the recorded flight was base_mode=128,
        # custom_mode=0.  It cannot report a usable flight mode and must never
        # become the destination of SET_MODE or GUIDED position targets.
        return bool(
            int(getattr(heartbeat, "base_mode", 0))
            & int(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
        )

    def _wait_for_control_heartbeat(self, connection: Any, mavutil: Any) -> Any:
        deadline = time.monotonic() + float(self.config["mavlink_heartbeat_timeout_s"])
        rejected_sources = set()
        while self._running.is_set() and time.monotonic() < deadline:
            heartbeat = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if heartbeat is None or heartbeat.get_type() == "BAD_DATA":
                continue
            self._publish_mavlink_message(heartbeat)
            source = self._message_source(heartbeat)
            if not self._heartbeat_is_control_autopilot(heartbeat, mavutil):
                rejected_sources.add(source)
                continue
            self._target_system, self._target_component = source
            connection.target_system = self._target_system
            connection.target_component = self._target_component
            with self._lock:
                self._state["target_system"] = self._target_system
                self._state["target_component"] = self._target_component
            if rejected_sources:
                self.logger.warning(
                    "Ignored non-controlling MAVLink heartbeat source(s): "
                    + ", ".join(
                        f"system={system} component={component}"
                        for system, component in sorted(rejected_sources)
                    )
                )
            return heartbeat
        return None

    def _publish_mavlink_message(self, message: Any) -> None:
        if self._message_callback is None:
            return
        try:
            source_system, source_component = self._message_source(message)
            payload = {
                "timestamp": time.time(),
                "message_type": message.get_type(),
                "source_system": source_system,
                "source_component": source_component,
                "fields": message.to_dict(),
            }
            self._message_callback(payload)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            from pymavlink import mavutil
        except Exception as exc:
            self.logger.error(f"pymavlink unavailable: {exc}")
            return
        no_port_logged = False
        while self._running.is_set():
            ports = self._ports()
            if not ports:
                if not no_port_logged:
                    self.logger.warning("No MAVLink serial port found; waiting for flight controller")
                    no_port_logged = True
                time.sleep(2)
                continue
            no_port_logged = False
            connected = False
            for port in ports:
                if not self._running.is_set():
                    break
                for baud in self._bauds():
                    if not self._running.is_set():
                        break
                    try:
                        self.logger.info(f"Trying MAVLink {port} @ {baud}")
                        connection = mavutil.mavlink_connection(
                            port, baud=baud, autoreconnect=False
                        )
                        heartbeat = self._wait_for_control_heartbeat(connection, mavutil)
                        if heartbeat is None:
                            connection.close()
                            continue
                        self._connection = connection
                        # Consume the selected heartbeat only after the target
                        # IDs are locked.  It was already written to raw logging.
                        self._consume(heartbeat, mavutil, publish_raw=False)
                        self._request_rates(connection, mavutil)
                        with self._lock:
                            self._state.update(
                                {"connected": True, "port": port, "baud": baud}
                            )
                        self.logger.info(
                            f"Flight controller connected on {port} @ {baud}; "
                            f"target system={self._target_system} "
                            f"component={self._target_component}"
                        )
                        connected = True
                        while self._running.is_set():
                            message = connection.recv_match(blocking=True, timeout=1)
                            if message is not None and message.get_type() != "BAD_DATA":
                                self._consume(message, mavutil)
                        break
                    except Exception as exc:
                        if self._running.is_set():
                            self.logger.warning(
                                f"MAVLink {port} @ {baud} unavailable: {exc}"
                            )
                    finally:
                        if self._connection is not None:
                            try:
                                self._connection.close()
                            except Exception:
                                pass
                        self._connection = None
                        self._target_system = None
                        self._target_component = None
                        with self._lock:
                            self._state["connected"] = False
                            self._state["target_system"] = None
                            self._state["target_component"] = None
                    if connected:
                        break
                if connected:
                    break
            if connected and self._running.is_set():
                self.logger.warning("Flight controller disconnected; reconnecting")
            time.sleep(2)

    def _consume(self, message: Any, mavutil: Any, publish_raw: bool = True) -> None:
        message_type = message.get_type()
        if publish_raw:
            self._publish_mavlink_message(message)
        source_system, source_component = self._message_source(message)
        if self._target_system is not None and source_system != self._target_system:
            return
        if (
            message_type == "HEARTBEAT"
            and self._target_component is not None
            and source_component != self._target_component
        ):
            return
        with self._lock:
            state = self._state
            if message_type == "GLOBAL_POSITION_INT":
                if abs(message.lat) > 1 and abs(message.lon) > 1:
                    state["latitude"] = message.lat / 1e7
                    state["longitude"] = message.lon / 1e7
                    state["altitude"] = message.alt / 1000.0
                    state["gps_update_monotonic"] = time.monotonic()
                state["relative_altitude"] = message.relative_alt / 1000.0
                state["vx"], state["vy"], state["vz"] = (
                    message.vx / 100.0,
                    message.vy / 100.0,
                    message.vz / 100.0,
                )
            elif message_type == "GPS_RAW_INT":
                fix_type = int(message.fix_type)
                satellites = int(getattr(message, "satellites_visible", 0))
                eph, epv = getattr(message, "eph", 65535), getattr(message, "epv", 65535)
                eph_m = None if eph == 65535 else eph / 100.0
                epv_m = None if epv == 65535 else epv / 100.0
                state["fix_type"] = fix_type
                state["satellites"] = satellites
                state["eph_m"] = eph_m
                state["epv_m"] = epv_m
                state["gps1"].update(
                    {
                        "fix_type": fix_type,
                        "satellites": satellites,
                        "eph_m": eph_m,
                        "epv_m": epv_m,
                    }
                )
                if abs(message.lat) > 1 and abs(message.lon) > 1:
                    latitude = message.lat / 1e7
                    longitude = message.lon / 1e7
                    altitude = message.alt / 1000.0
                    state["latitude"] = latitude
                    state["longitude"] = longitude
                    state["altitude"] = altitude
                    state["gps1"].update(
                        {
                            "latitude": latitude,
                            "longitude": longitude,
                            "altitude": altitude,
                        }
                    )
                    state["gps_update_monotonic"] = time.monotonic()
            elif message_type == "GPS2_RAW":
                eph, epv = getattr(message, "eph", 65535), getattr(message, "epv", 65535)
                update = {
                    "fix_type": int(message.fix_type),
                    "satellites": int(getattr(message, "satellites_visible", 0)),
                    "eph_m": None if eph == 65535 else eph / 100.0,
                    "epv_m": None if epv == 65535 else epv / 100.0,
                    "latitude": None,
                    "longitude": None,
                    "altitude": None,
                }
                if abs(message.lat) > 1 and abs(message.lon) > 1:
                    update["latitude"] = message.lat / 1e7
                    update["longitude"] = message.lon / 1e7
                    update["altitude"] = message.alt / 1000.0
                state["gps2"].update(update)
            elif message_type == "ATTITUDE":
                for key in ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"):
                    state[key] = float(getattr(message, key))
            elif message_type == "HIGHRES_IMU":
                state["xacc"] = float(message.xacc)
                state["yacc"] = float(message.yacc)
                state["zacc"] = float(message.zacc)
            elif message_type == "VFR_HUD":
                state["groundspeed"] = float(message.groundspeed)
                state["airspeed"] = float(message.airspeed)
            elif message_type == "SYS_STATUS":
                voltage, current = message.voltage_battery, message.current_battery
                state["voltage_v"] = None if voltage < 0 else voltage / 1000.0
                state["current_a"] = None if current < 0 else current / 100.0
                remaining = int(message.battery_remaining)
                state["battery_remaining_pct"] = None if remaining < 0 else remaining
            elif message_type == "BATTERY_STATUS":
                remaining = int(getattr(message, "battery_remaining", -1))
                if remaining >= 0:
                    state["battery_remaining_pct"] = remaining
            elif message_type == "HEARTBEAT":
                state["armed"] = bool(
                    message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                state["flight_mode"] = mavutil.mode_string_v10(message)
            elif message_type == "COMMAND_ACK":
                state["last_command_ack_command"] = int(getattr(message, "command", -1))
                state["last_command_ack_result"] = int(getattr(message, "result", -1))
                progress = int(getattr(message, "progress", 255))
                state["last_command_ack_progress"] = None if progress == 255 else progress
                state["last_command_ack_monotonic"] = time.monotonic()
            elif message_type == "STATUSTEXT":
                state["last_status_text"] = str(getattr(message, "text", "")).strip()
                state["last_status_text_monotonic"] = time.monotonic()

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def send_reposition(
        self, latitude: float, longitude: float, relative_altitude_m: float
    ) -> Tuple[bool, str]:
        connection = self._connection
        if connection is None:
            return False, "MAVLink is not connected"
        try:
            from pymavlink import mavutil
            hold_yaw_rad = float(self.get().get("yaw") or 0.0)

            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            with self._send_lock:
                connection.mav.set_position_target_global_int_send(
                    int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
                    connection.target_system,
                    connection.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    type_mask,
                    int(round(float(latitude) * 1e7)),
                    int(round(float(longitude) * 1e7)),
                    float(relative_altitude_m),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    hold_yaw_rad,
                    0.0,
                )
            self.logger.warning(
                f"AP6 GUIDED global target sent to {latitude:.7f}, {longitude:.7f} "
                f"at {relative_altitude_m:.1f} m above home"
            )
            return True, "SET_POSITION_TARGET_GLOBAL_INT sent"
        except Exception as exc:
            return False, f"MAVLink global target failed: {exc}"

    def send_body_velocity(
        self,
        forward_mps: float,
        right_mps: float,
        down_mps: float,
        yaw_rad: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Send a GUIDED velocity target in aircraft body axes.

        Positive axes follow MAVLink BODY_NED: forward, right and down.  This
        is used by the D455 visual servo so image error can be corrected
        without repeatedly converting a bounding box into a noisy GPS point.
        """
        connection = self._connection
        if connection is None:
            return False, "MAVLink is not connected"
        try:
            from pymavlink import mavutil

            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            if yaw_rad is None:
                type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            with self._send_lock:
                connection.mav.set_position_target_local_ned_send(
                    int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
                    connection.target_system,
                    connection.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,
                    type_mask,
                    0.0,
                    0.0,
                    0.0,
                    float(forward_mps),
                    float(right_mps),
                    float(down_mps),
                    0.0,
                    0.0,
                    0.0,
                    float(yaw_rad) if yaw_rad is not None else 0.0,
                    0.0,
                )
            return True, "SET_POSITION_TARGET_LOCAL_NED body velocity sent"
        except Exception as exc:
            return False, f"MAVLink body velocity target failed: {exc}"

    def set_flight_mode(self, mode_name: str) -> Tuple[bool, str]:
        connection = self._connection
        if connection is None:
            return False, "MAVLink is not connected"
        mode_name = str(mode_name).upper()
        try:
            from pymavlink import mavutil

            mapping = connection.mode_mapping() or {}
            if mode_name not in mapping:
                available = ",".join(sorted(mapping.keys())) if mapping else "none"
                return False, f"Mode {mode_name} is not available; modes={available}"
            mode_id = int(mapping[mode_name])
            command_id = int(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
            started = time.monotonic()
            with self._lock:
                self._state["last_command_ack_command"] = None
                self._state["last_command_ack_result"] = None
                self._state["last_command_ack_progress"] = None
                self._state["last_command_ack_monotonic"] = 0.0
            with self._send_lock:
                connection.mav.command_long_send(
                    connection.target_system,
                    connection.target_component,
                    command_id,
                    0,
                    float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                    float(mode_id),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            deadline = time.monotonic() + float(self.config["mode_change_timeout_s"])
            while time.monotonic() < deadline:
                state = self.get()
                if str(state.get("flight_mode") or "").upper() == mode_name:
                    return True, f"Flight mode changed to {mode_name}"
                ack_time = float(state.get("last_command_ack_monotonic") or 0.0)
                ack_command = state.get("last_command_ack_command")
                ack_result = state.get("last_command_ack_result")
                if ack_time >= started and ack_command == command_id and ack_result not in (
                    int(mavutil.mavlink.MAV_RESULT_ACCEPTED),
                    int(mavutil.mavlink.MAV_RESULT_IN_PROGRESS),
                ):
                    return False, (
                        f"Flight controller rejected {mode_name}: "
                        f"{self._mav_result_name(mavutil, int(ack_result))}"
                    )
                time.sleep(0.1)
            state = self.get()
            details = []
            ack_time = float(state.get("last_command_ack_monotonic") or 0.0)
            if ack_time >= started and state.get("last_command_ack_command") == command_id:
                details.append(
                    "ACK "
                    + self._mav_result_name(
                        mavutil, int(state.get("last_command_ack_result") or 0)
                    )
                )
            status_time = float(state.get("last_status_text_monotonic") or 0.0)
            if status_time >= started and state.get("last_status_text"):
                details.append(str(state["last_status_text"]))
            suffix = f"; {'; '.join(details)}" if details else "; no mode ACK received"
            return False, (
                f"Requested {mode_name}, current mode is "
                f"{state.get('flight_mode') or 'UNKNOWN'}{suffix}"
            )
        except Exception as exc:
            return False, f"Flight mode change failed: {exc}"

    @staticmethod
    def _mav_result_name(mavutil: Any, result: int) -> str:
        try:
            return str(mavutil.mavlink.enums["MAV_RESULT"][int(result)].name)
        except Exception:
            return f"MAV_RESULT_{int(result)}"

    def ensure_guided_for_reposition(self) -> Tuple[bool, str]:
        guided = str(self.config["guided_mode_name"]).upper()
        current = str(self.get().get("flight_mode") or "").upper()
        if current == guided:
            return True, f"Already in {guided}"
        if not bool(self.config["auto_switch_to_guided"]):
            return False, f"Flight mode {current or 'UNKNOWN'} is not {guided}"
        allowed_from = {
            str(value).upper() for value in self.config["auto_switch_guided_from_modes"]
        }
        if current not in allowed_from:
            return False, f"Refusing auto-switch from mode {current or 'UNKNOWN'}"
        return self.set_flight_mode(guided)

    def send_hold_current_position(self) -> Tuple[bool, str]:
        state = self.get()
        if state.get("latitude") is None or state.get("longitude") is None:
            return False, "Cannot hold without current GPS"
        return self.send_reposition(
            float(state["latitude"]),
            float(state["longitude"]),
            max(0.0, float(state.get("relative_altitude") or 0.0)),
        )

    def stop(self) -> None:
        self._running.clear()
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3)


def detector_process_main(
    config: Dict[str, Any], input_queue: Any, output_queue: Any
) -> None:
    """TensorRT/PyTorch inference process, isolated from the ROS publish loop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # This Jetson's torchvision optional image extension was built against a
    # different torch ABI. Ultralytics/TensorRT does not use that extension for
    # our OpenCV numpy frames, so keep the known non-actionable warning out of
    # the flight log while retaining all inference/runtime errors.
    warnings.filterwarnings(
        "ignore", message="Failed to load image Python extension.*"
    )
    from ultralytics import YOLO

    engine_path = os.path.expanduser(str(config["engine_path"]))
    fallback_path = os.path.expanduser(str(config.get("fallback_model_path", "")))
    source = "TensorRT"
    try:
        model = YOLO(engine_path, task="detect")
    except Exception as exc:
        output_queue.put({"error": f"TensorRT load failed: {exc}"})
        model = YOLO(fallback_path, task="detect")
        source = "PyTorch fallback"
    fallback_id = 1
    target_streaks: Dict[int, int] = {}
    target_track_boxes: Dict[int, Tuple[int, int, int, int]] = {}
    target_last_seen: Dict[int, float] = {}
    target_classes = {
        str(value).strip().lower()
        for value in config.get("target_classes", ["car"])
        if str(value).strip()
    }
    if not target_classes:
        target_classes = {"car"}
    target_color_mode = str(config.get("target_color_mode", "white")).strip().lower()
    if target_color_mode not in ("white", "any"):
        target_color_mode = "white"
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        for attempt in range(2):
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(config["tracker"]),
                    conf=float(config["confidence_threshold"]),
                    iou=0.45,
                    verbose=False,
                )
                detections: List[Dict[str, Any]] = []
                detected_at = time.time()
                detected_monotonic = time.monotonic()
                for result in results:
                    names = result.names
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = str(
                            names.get(class_id, class_id)
                            if isinstance(names, dict)
                            else names[class_id]
                        ).lower()
                        if class_name not in target_classes:
                            continue
                        bbox = tuple(int(value) for value in box.xyxy[0].tolist())
                        if box.id is not None:
                            track_id = int(box.id[0].item())
                        else:
                            track_id, fallback_id = fallback_id, fallback_id + 1
                        x1, y1, x2, y2 = bbox
                        height, width = frame.shape[:2]
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)
                        (
                            white_ratio,
                            white_pixels,
                            mean_s,
                            mean_v,
                            red_ratio,
                            edge_density,
                        ) = white_vehicle_score(frame, (x1, y1, x2, y2), config)
                        aspect_ratio = (x2 - x1) / max(1.0, float(y2 - y1))
                        is_white_body = (
                            white_ratio >= float(config["white_ratio_threshold"])
                            and white_ratio <= float(config["white_ratio_max"])
                            and white_pixels >= int(config["white_min_pixels"])
                            and mean_s
                            <= float(config["white_roi_mean_saturation_max"])
                            and mean_v >= float(config["white_roi_mean_value_min"])
                            and red_ratio
                            <= float(config["red_veto_ratio_threshold"])
                            and edge_density >= float(config["car_min_edge_density"])
                            and aspect_ratio >= float(config["car_aspect_ratio_min"])
                            and aspect_ratio <= float(config["car_aspect_ratio_max"])
                        )
                        is_target = target_color_mode == "any" or is_white_body
                        if track_id not in target_streaks:
                            best_old_id = None
                            best_iou = 0.0
                            match_iou = float(
                                config.get("target_streak_match_iou", 0.35)
                            )
                            hold_s = float(
                                config.get("target_streak_hold_s", 2.5)
                            )
                            for old_id, old_bbox in target_track_boxes.items():
                                if (
                                    detected_monotonic
                                    - float(target_last_seen.get(old_id, 0.0))
                                    > hold_s
                                ):
                                    continue
                                ax1, ay1, ax2, ay2 = bbox
                                bx1, by1, bx2, by2 = old_bbox
                                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                                union = area_a + area_b - intersection
                                score = intersection / union if union else 0.0
                                if score >= match_iou and score > best_iou:
                                    best_old_id, best_iou = old_id, score
                            if best_old_id is not None:
                                target_streaks[track_id] = target_streaks.pop(
                                    best_old_id, 0
                                )
                                target_track_boxes.pop(best_old_id, None)
                                target_last_seen.pop(best_old_id, None)
                        if is_target:
                            target_streaks[track_id] = target_streaks.get(track_id, 0) + 1
                        else:
                            target_streaks[track_id] = 0
                        target_track_boxes[track_id] = bbox
                        target_last_seen[track_id] = detected_monotonic
                        streak = target_streaks[track_id]
                        confirm_frames = int(
                            config.get("target_confirm_frames", 0) or 0
                        )
                        if confirm_frames <= 0:
                            confirm_frames = int(config.get("white_confirm_frames", 3))
                        confirmed = streak >= max(1, confirm_frames)
                        snapshot = ""
                        if (
                            confirmed
                            and bool(config.get("include_candidate_image_base64", True))
                            and x2 > x1
                            and y2 > y1
                        ):
                            crop = frame[y1:y2, x1:x2]
                            max_width = max(
                                1, int(config.get("candidate_image_max_width", 320))
                            )
                            max_height = max(
                                1, int(config.get("candidate_image_max_height", 240))
                            )
                            crop_height, crop_width = crop.shape[:2]
                            resize_scale = min(
                                1.0,
                                max_width / max(1, crop_width),
                                max_height / max(1, crop_height),
                            )
                            if resize_scale < 1.0:
                                crop = cv2.resize(
                                    crop,
                                    (
                                        max(1, int(round(crop_width * resize_scale))),
                                        max(1, int(round(crop_height * resize_scale))),
                                    ),
                                    interpolation=cv2.INTER_AREA,
                                )
                            ok, encoded = cv2.imencode(
                                ".jpg",
                                crop,
                                [
                                    cv2.IMWRITE_JPEG_QUALITY,
                                    int(config["candidate_jpeg_quality"]),
                                ],
                            )
                            if ok:
                                snapshot = base64.b64encode(encoded.tobytes()).decode("ascii")
                        detections.append(
                            {
                                "bbox": bbox,
                                "track_id": track_id,
                                "confidence": float(box.conf[0].item()),
                                "white_ratio": white_ratio,
                                "white_pixels": white_pixels,
                                "white_mean_saturation": mean_s,
                                "white_mean_value": mean_v,
                                "red_ratio": red_ratio,
                                "edge_density": edge_density,
                                "aspect_ratio": aspect_ratio,
                                "white_streak": streak,
                                # Keep the legacy fields for the current Unity
                                # dashboard and experiment CSV schema.
                                "white_confirmed": confirmed,
                                "target_confirmed": confirmed,
                                "target_color_mode": target_color_mode,
                                "image_base64": snapshot,
                                "detected_at": detected_at,
                            }
                        )
                streak_hold_s = float(config.get("target_streak_hold_s", 2.5))
                retained_ids = {
                    track_id
                    for track_id, last_seen in target_last_seen.items()
                    if detected_monotonic - float(last_seen) <= streak_hold_s
                }
                target_streaks = {
                    track_id: streak
                    for track_id, streak in target_streaks.items()
                    if track_id in retained_ids
                }
                target_track_boxes = {
                    track_id: bbox
                    for track_id, bbox in target_track_boxes.items()
                    if track_id in retained_ids
                }
                target_last_seen = {
                    track_id: last_seen
                    for track_id, last_seen in target_last_seen.items()
                    if track_id in retained_ids
                }
                detections.sort(key=lambda item: item["confidence"], reverse=True)
                payload = {
                    "detections": detections[:50],
                    "source": source,
                }
                try:
                    while True:
                        output_queue.get_nowait()
                except queue.Empty:
                    pass
                output_queue.put_nowait(payload)
                break
            except Exception as exc:
                if source == "TensorRT" and attempt == 0 and os.path.isfile(fallback_path):
                    output_queue.put({"error": f"TensorRT inference failed: {exc}"})
                    model = YOLO(fallback_path, task="detect")
                    source = "PyTorch fallback"
                    continue
                output_queue.put({"error": f"Detector inference failed: {exc}"})
                source = "failed"
                break


class ProcessDetector:
    """Latest-frame IPC wrapper that keeps inference from blocking ROS timers."""

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.logger = logger
        context = mp.get_context("spawn")
        self.input_queue = context.Queue(maxsize=1)
        self.output_queue = context.Queue(maxsize=2)
        self.process = context.Process(
            target=detector_process_main,
            args=(config, self.input_queue, self.output_queue),
            name="car-detector",
            daemon=True,
        )
        self.process.start()
        self.latest: List[Dict[str, Any]] = []
        self.source = "loading"
        self.in_flight = False

    def submit_and_get(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
        try:
            while True:
                payload = self.output_queue.get_nowait()
                if payload.get("error"):
                    self.logger.error(payload["error"])
                if "detections" in payload:
                    self.latest = payload["detections"]
                    self.source = payload.get("source", self.source)
                    self.in_flight = False
        except queue.Empty:
            pass
        # Send one frame only when the prior inference is complete. This avoids
        # repeatedly pickling/replacing 900 KB frames while TensorRT is busy.
        if not self.in_flight:
            try:
                self.input_queue.put_nowait(frame)
                self.in_flight = True
            except queue.Full:
                pass
        if not self.process.is_alive() and self.source != "failed":
            self.source = "failed"
            self.logger.error("Detector process stopped unexpectedly")
        return copy.deepcopy(self.latest), self.source

    def stop(self) -> None:
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)


class UnityDashboardBridge(Node):
    """Continuously publishes detector and mock navigation data to Unity."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("unity_dashboard_bridge")
        self.config = {**DEFAULT_CONFIG, **config}
        self.source_session_id = uuid.uuid4().hex
        self.mode = str(self.config["mode"]).lower()
        if self.mode not in ("live", "mock"):
            raise ValueError("mode must be 'live' or 'mock'")
        if self.mode == "mock":
            self.config["mock_gps"] = True
            self.config["mock_imu"] = True

        self.detections_pub = self.create_publisher(
            String, "/unity/detections_json", 10
        )
        image_qos = QoSProfile(
            depth=max(1, int(self.config.get("image_qos_depth", 10)))
        )
        self.raw_image_pub = self.create_publisher(
            CompressedImage, "/d455i/color/image_raw/compressed", image_qos
        )
        self.image_pub = self.create_publisher(
            CompressedImage, "/d455i/color/image_annotated/compressed", image_qos
        )
        self.depth_pub = self.create_publisher(
            Image, "/d455i/depth/image_rect_raw", qos_profile_sensor_data
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/d455i/color/camera_info", 5
        )
        self.d455_imu_pub = self.create_publisher(Imu, "/d455i/imu", 20)
        self.gps_pub = self.create_publisher(NavSatFix, "/uav/gps", 10)
        self.imu_pub = self.create_publisher(Imu, "/uav/imu", 10)
        self.velocity_pub = self.create_publisher(
            TwistStamped, "/uav/velocity_ned", 10
        )
        self.battery_pub = self.create_publisher(BatteryState, "/uav/battery", 10)
        self.telemetry_pub = self.create_publisher(
            String, "/uav/telemetry_json", 10
        )
        self.mavlink_pub = self.create_publisher(String, "/uav/mavlink_json", 200)
        status_qos = QoSProfile(depth=20)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(
            String, "/unity/target_command_status", status_qos
        )
        self.command_sub = self.create_subscription(
            String, "/unity/target_command", self._command_callback, 10
        )

        self.model = None
        self.model_source = "none"
        self.capture = None
        self.realsense: Optional[RealSenseSource] = None
        self.rtsp_camera: Optional[RtspFfmpegSource] = None
        self.telemetry: Optional[MavlinkTelemetryReader] = None
        self.process_detector: Optional[ProcessDetector] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_intrinsics: Optional[Dict[str, float]] = None
        self._latest_d455_imu: Dict[str, Any] = {}
        self._telemetry_state: Dict[str, Any] = {}
        self._last_valid_gps: Optional[Dict[str, Any]] = None
        self._depth_running = threading.Event()
        self._depth_event = threading.Event()
        self._depth_lock = threading.Lock()
        self._depth_pending: Optional[Tuple[np.ndarray, Any]] = None
        self._last_depth_submit = 0.0
        self._fallback_id = 1
        self._last_fallback_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
        self._tick = 0
        self._active_command: Optional[Dict[str, Any]] = None
        self._candidate_cache: Dict[int, Dict[str, Any]] = {}
        self._candidate_archive: Dict[int, Dict[str, Any]] = {}
        self._candidate_preview_last_sent: Dict[str, float] = {}
        self._latest_confirmed_target_detections: List[Dict[str, Any]] = []
        self._latest_target_frame_shape: Tuple[int, ...] = ()
        self._geo_filters: Dict[int, Dict[str, Any]] = {}
        self._detector_running = threading.Event()
        self._detector_event = threading.Event()
        self._detector_lock = threading.Lock()
        self._detector_pending: Optional[np.ndarray] = None
        self._detector_frame: Optional[np.ndarray] = None
        self._detector_results: List[Dict[str, Any]] = []
        self._visual_detection_cache: Dict[int, Dict[str, Any]] = {}
        self._visual_detection_templates: Dict[int, np.ndarray] = {}
        self._visual_tracking_misses: Dict[int, int] = {}
        self._depth_running.set()
        self._depth_thread = threading.Thread(
            target=self._depth_publish_loop, name="depth-publisher", daemon=True
        )
        self._depth_thread.start()

        if self.mode == "live":
            self._open_camera()
            if bool(self.config["mavlink_enabled"]):
                self.telemetry = MavlinkTelemetryReader(
                    self.config, self.get_logger(), self._publish_mavlink_json
                )
                self.telemetry.start()
            self.process_detector = ProcessDetector(self.config, self.get_logger())
            self.model_source = "loading"
        else:
            self.get_logger().info("Mock mode: using generated camera and detections")

        rate = max(0.1, float(self.config["publish_rate_hz"]))
        self.publish_timer = self.create_timer(1.0 / rate, self._publish_callback)
        self.command_timer = self.create_timer(1.0, self._command_status_tick)
        self.get_logger().info(
            f"Unity Dashboard bridge ready: mode={self.mode}, rate={rate:g} Hz"
        )
        if bool(self.config["enable_flight_control"]):
            self.get_logger().warning(
                "OUTDOOR CONTROL ENABLED: validated CONFIRM may send AP6 REPOSITION"
            )
        else:
            self.get_logger().warning(
                "INDOOR/DISPLAY mode: target commands cannot control AP6"
            )

    def _open_camera(self) -> None:
        source = str(self.config.get("camera_source", "")).strip().lower()
        if not source:
            source = "realsense" if bool(self.config["use_realsense"]) else "opencv"
        if source in ("realsense", "d455", "d455i") or bool(self.config["use_realsense"]):
            self.realsense = RealSenseSource(self.config, self.get_logger())
            self.realsense.start()
            self.get_logger().info("D455i capture/reconnect thread started")
            return
        if source in ("rtsp", "g3p"):
            self.rtsp_camera = RtspFfmpegSource(self.config, self.get_logger())
            self.rtsp_camera.start()
            self.get_logger().info("G3P RTSP capture/reconnect thread started")
            return

        index = int(self.config["camera_index"])
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            self.capture = capture
            self.get_logger().info(f"OpenCV camera {index} opened")
        else:
            capture.release()
            self.get_logger().error(
                f"Could not open camera {index}; using generated debug image"
            )

    def _load_detector(self) -> None:
        engine_path = os.path.expanduser(str(self.config["engine_path"]))
        if str(self.config["detector_backend"]) != "ultralytics_engine":
            self.get_logger().error(
                f"Unsupported detector_backend={self.config['detector_backend']}; "
                "using mock detector"
            )
            return
        try:
            if not os.path.isfile(engine_path):
                raise FileNotFoundError(engine_path)
            from ultralytics import YOLO

            self.model = YOLO(engine_path)
            self.model_source = "TensorRT"
            self.get_logger().info(f"TensorRT engine loaded: {engine_path}")
        except Exception as exc:
            self.model = None
            self.model_source = "none"
            self.get_logger().error(
                f"Could not load Ultralytics TensorRT engine: {exc}; "
                "trying PyTorch fallback"
            )
            self._load_fallback_model()

    def _load_fallback_model(self) -> bool:
        model_path = os.path.expanduser(str(self.config.get("fallback_model_path", "")))
        try:
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(model_path or "fallback_model_path is empty")
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.model_source = "PyTorch fallback"
            self.get_logger().warning(f"Detector switched to portable model: {model_path}")
            return True
        except Exception as exc:
            self.model = None
            self.model_source = "mock"
            self.get_logger().error(f"Could not load fallback model: {exc}; using mock detector")
            return False

    def _read_frame(self) -> Tuple[np.ndarray, bool]:
        if self.realsense is not None:
            frame, depth, intrinsics, imu = self.realsense.get()
            self._latest_depth = depth
            self._latest_intrinsics = intrinsics
            self._latest_d455_imu = imu
            if frame is not None:
                return frame, True
        if self.rtsp_camera is not None:
            frame, depth, intrinsics, imu = self.rtsp_camera.get()
            self._latest_depth = depth
            self._latest_intrinsics = intrinsics
            self._latest_d455_imu = imu
            if frame is not None:
                return frame, True
        if self.capture is not None:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                return frame, True
        return self._debug_frame(), False

    def _debug_frame(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "UNITY UAV DASHBOARD - MOCK CAMERA",
            (38, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (190, 190, 190),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (190, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _fallback_track_id(self, bbox: Tuple[int, int, int, int]) -> int:
        best_id, best_iou = 0, 0.0
        for track_id, old_bbox in self._last_fallback_tracks:
            score = self._box_iou(bbox, old_bbox)
            if score > best_iou:
                best_id, best_iou = track_id, score
        if best_iou < 0.25:
            best_id = self._fallback_id
            self._fallback_id += 1
        return best_id

    def _mock_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        height, width = frame.shape[:2]
        travel = max(1, width - 260)
        x1 = 70 + int((self._tick * 7) % travel)
        y1 = int(height * 0.48 + 18 * math.sin(self._tick * 0.12))
        bbox = (x1, y1, min(width - 1, x1 + 150), min(height - 1, y1 + 82))
        return [
            {
                "bbox": bbox,
                "track_id": 1,
                "confidence": 0.91,
                "white_ratio": 0.85,
                "white_streak": int(
                    self.config.get("target_confirm_frames", 0)
                    or self.config.get("white_confirm_frames", 3)
                ),
                "white_confirmed": True,
                "target_confirmed": True,
                "target_color_mode": str(
                    self.config.get("target_color_mode", "white")
                ),
                "image_base64": "",
                "detected_at": time.time(),
            }
        ]

    def _model_detections(
        self, frame: np.ndarray, allow_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            return self._mock_detections(frame)
        detections: List[Dict[str, Any]] = []
        new_fallback_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=str(self.config["tracker"]),
                conf=float(self.config["confidence_threshold"]),
                iou=0.45,
                verbose=False,
            )
            for result in results:
                names = result.names
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(
                        names.get(class_id, class_id)
                        if isinstance(names, dict)
                        else names[class_id]
                    ).lower()
                    if class_name != "car":
                        continue
                    coords = box.xyxy[0].tolist()
                    bbox = tuple(int(value) for value in coords)
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                    else:
                        track_id = self._fallback_track_id(bbox)
                        new_fallback_tracks.append((track_id, bbox))
                    detections.append(
                        {
                            "bbox": bbox,
                            "track_id": track_id,
                            "confidence": float(box.conf[0].item()),
                        }
                    )
            self._last_fallback_tracks = new_fallback_tracks
            return detections
        except Exception as exc:
            failed_source = self.model_source
            self.get_logger().error(f"{failed_source} inference failed: {exc}")
            self.model = None
            if allow_fallback and failed_source == "TensorRT" and self._load_fallback_model():
                return self._model_detections(frame, allow_fallback=False)
            self.model_source = "mock"
            return self._mock_detections(frame)

    def _start_detector_worker(self) -> None:
        self._detector_running.set()
        self._detector_thread = threading.Thread(
            target=self._detector_loop, name="car-detector", daemon=True
        )
        self._detector_thread.start()

    def _detector_loop(self) -> None:
        while self._detector_running.is_set():
            self._detector_event.wait(timeout=0.5)
            self._detector_event.clear()
            if not self._detector_running.is_set():
                break
            with self._detector_lock:
                frame = self._detector_pending
                self._detector_pending = None
            if frame is None:
                continue
            detections = self._model_detections(frame)
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            detections = detections[: max(0, int(self.config["max_candidates"]))]
            annotated = frame.copy()
            display_limit = max(
                0, int(self.config.get("visual_display_max_detections", 3))
            )
            for item in detections[:display_limit]:
                self._draw_detection(annotated, item)
            with self._detector_lock:
                self._detector_frame = annotated
                self._detector_results = copy.deepcopy(detections)

    def _latest_detection(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        now_epoch = time.time()
        max_display_age = float(self.config["detection_display_max_age_s"])
        draw_non_white = bool(self.config["draw_non_white_detections"])
        if self.process_detector is not None:
            detections, source = self.process_detector.submit_and_get(frame)
            self.model_source = source
            detections = [
                item
                for item in detections
                if now_epoch - float(item.get("detected_at") or 0.0) <= max_display_age
            ]
            visual_detections = self._visual_detections(
                detections, now_epoch, frame
            )
            annotated = frame.copy()
            for item in self._select_visual_detections(
                visual_detections, draw_non_white
            ):
                self._draw_detection(annotated, item)
            self._draw_navigation_safe_region(annotated)
            return annotated, detections
        with self._detector_lock:
            # Latest-frame semantics: replace a frame not yet picked up by the
            # slow detector instead of building latency in a queue.
            self._detector_pending = frame.copy()
            detections = copy.deepcopy(self._detector_results)
        detections = [
            item
            for item in detections
            if now_epoch - float(item.get("detected_at") or 0.0) <= max_display_age
        ]
        visual_detections = self._visual_detections(detections, now_epoch, frame)
        annotated = frame.copy()
        for item in self._select_visual_detections(
            visual_detections, draw_non_white
        ):
            self._draw_detection(annotated, item)
        self._draw_navigation_safe_region(annotated)
        self._detector_event.set()
        return annotated, detections

    def _select_visual_detections(
        self, visual_detections: List[Dict[str, Any]], draw_all: bool
    ) -> List[Dict[str, Any]]:
        drawable = [
            item
            for item in visual_detections
            if draw_all
            or bool(item.get("target_confirmed", item.get("white_confirmed")))
        ]
        limit = max(0, int(self.config.get("visual_display_max_detections", 3)))
        return drawable[:limit]

    def _draw_navigation_safe_region(self, frame: np.ndarray) -> None:
        if not bool(self.config.get("draw_navigation_safe_region", False)):
            return
        height, width = frame.shape[:2]
        margin = min(
            0.45,
            max(
                0.0,
                float(
                    self.config.get(
                        "auto_verification_min_edge_margin_ratio", 0.18
                    )
                ),
            ),
        )
        x1, y1 = int(round(width * margin)), int(round(height * margin))
        x2, y2 = int(round(width * (1.0 - margin))), int(
            round(height * (1.0 - margin))
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 210, 80), 1)
        cv2.putText(
            frame,
            "NAV TARGET ZONE",
            (x1 + 5, min(height - 5, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 210, 80),
            1,
            cv2.LINE_AA,
        )

    def _visual_detections(
        self,
        detections: List[Dict[str, Any]],
        now_epoch: float,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Track visible cars across detector misses, without navigation use."""
        hold_s = max(0.0, float(self.config.get("visual_detection_hold_s", 15.0)))
        deduplicate_iou = float(
            self.config.get("visual_detection_deduplicate_iou", 0.35)
        )
        live_ids = set()

        for detection in detections:
            track_id = int(detection.get("track_id", -1))
            if track_id < 0:
                continue
            live_ids.add(track_id)
            bbox = tuple(int(value) for value in detection["bbox"])

            # ByteTrack can allocate a new ID after a miss. Remove an older,
            # strongly-overlapping held box so Unity never shows duplicates.
            for cached_id, cached in list(self._visual_detection_cache.items()):
                if cached_id == track_id:
                    continue
                cached_bbox = tuple(int(value) for value in cached["bbox"])
                if self._box_iou(bbox, cached_bbox) >= deduplicate_iou:
                    self._visual_detection_cache.pop(cached_id, None)
                    self._visual_detection_templates.pop(cached_id, None)
                    self._visual_tracking_misses.pop(cached_id, None)

            cached_detection = copy.deepcopy(detection)
            cached_detection["visual_held"] = False
            self._visual_detection_cache[track_id] = cached_detection
            template = self._visual_template(frame, bbox)
            if template is not None:
                self._visual_detection_templates[track_id] = template
            self._visual_tracking_misses[track_id] = 0

        miss_limit = max(
            1, int(self.config.get("visual_tracking_miss_frames", 2))
        )
        for track_id, detection in list(self._visual_detection_cache.items()):
            if track_id in live_ids:
                continue
            age = now_epoch - float(detection.get("detected_at") or 0.0)
            if age > hold_s or not self._track_visual_box(
                frame, track_id, detection
            ):
                misses = self._visual_tracking_misses.get(track_id, 0) + 1
                self._visual_tracking_misses[track_id] = misses
                if age > hold_s or misses >= miss_limit:
                    self._visual_detection_cache.pop(track_id, None)
                    self._visual_detection_templates.pop(track_id, None)
                    self._visual_tracking_misses.pop(track_id, None)
            else:
                self._visual_tracking_misses[track_id] = 0

        visual = []
        for track_id, detection in self._visual_detection_cache.items():
            item = copy.deepcopy(detection)
            item["visual_held"] = track_id not in live_ids
            visual.append(item)
        visual.sort(
            key=lambda item: (
                bool(item.get("visual_held")),
                -float(item.get("confidence") or 0.0),
            )
        )
        return visual

    @staticmethod
    def _visual_template(
        frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return gray.copy() if float(gray.std()) >= 4.0 else None

    def _track_visual_box(
        self,
        frame: np.ndarray,
        track_id: int,
        detection: Dict[str, Any],
    ) -> bool:
        template = self._visual_detection_templates.get(track_id)
        if template is None or template.size == 0:
            return False

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = tuple(int(value) for value in detection["bbox"])
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        expansion = max(
            0.25, float(self.config.get("visual_tracking_search_expansion", 1.0))
        )
        margin_x = max(12, int(box_width * expansion))
        margin_y = max(12, int(box_height * expansion))
        sx1, sy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        sx2, sy2 = min(frame_width, x2 + margin_x), min(frame_height, y2 + margin_y)
        search = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
        if search.size == 0:
            return False

        best_score = -1.0
        best_bbox = None
        for scale in (0.85, 1.0, 1.15):
            template_width = max(6, int(round(template.shape[1] * scale)))
            template_height = max(6, int(round(template.shape[0] * scale)))
            if template_width > search.shape[1] or template_height > search.shape[0]:
                continue
            scaled = (
                template
                if scale == 1.0
                else cv2.resize(
                    template,
                    (template_width, template_height),
                    interpolation=cv2.INTER_AREA,
                )
            )
            scores = cv2.matchTemplate(search, scaled, cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(scores)
            if float(maximum) > best_score:
                nx1, ny1 = sx1 + max_location[0], sy1 + max_location[1]
                best_score = float(maximum)
                best_bbox = (
                    nx1,
                    ny1,
                    min(frame_width - 1, nx1 + template_width),
                    min(frame_height - 1, ny1 + template_height),
                )

        minimum_score = float(self.config.get("visual_tracking_min_score", 0.55))
        if best_bbox is None or best_score < minimum_score:
            return False
        detection["bbox"] = best_bbox
        detection["visual_track_score"] = best_score
        return True

    def _candidate_from_detection(
        self, detection: Dict[str, Any], frame_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = detection["bbox"]
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gps_valid = self._gps_fix_valid(self._telemetry_state)
        geo_available = gps_valid or bool(self.config["mock_gps"])
        base_lat = float(
            self._telemetry_state["latitude"]
            if gps_valid
            else self.config["default_latitude"]
        )
        base_lon = float(
            self._telemetry_state["longitude"]
            if gps_valid
            else self.config["default_longitude"]
        )
        north_m, east_m, range_m, measurement_error = self._target_offset_ned(
            detection, center_x, center_y, width, height
        )
        latitude = base_lat + math.degrees(north_m / 6_378_137.0)
        longitude = base_lon + math.degrees(
            east_m
            / (6_378_137.0 * max(0.01, math.cos(math.radians(base_lat))))
        )
        gps_error = float(self._telemetry_state.get("eph_m") or 0.0)
        attitude_error = range_m * math.radians(float(self.config["attitude_error_deg"]))
        latency = max(0.0, time.time() - float(detection.get("detected_at") or time.time()))
        speed = math.hypot(
            float(self._telemetry_state.get("vx") or 0.0),
            float(self._telemetry_state.get("vy") or 0.0),
        )
        position_error = (
            math.sqrt(
                measurement_error**2
                + gps_error**2
                + attitude_error**2
                + (latency * speed) ** 2
            )
            if geo_available
            else 999.0
        )
        latitude, longitude, position_error = self._filter_geolocation(
            int(detection["track_id"]),
            float(detection.get("detected_at") or time.time()),
            latitude,
            longitude,
            position_error,
        )
        target_e, target_n = wgs84_to_twd97(latitude, longitude)
        candidate_prefix = str(
            self.config.get("candidate_id_prefix", "white-car")
        ).strip() or "car"
        return {
            "candidate_id": f"{candidate_prefix}-{int(detection['track_id'])}",
            "track_id": int(detection["track_id"]),
            "class_name": str(self.config["class_name_override"] or "car"),
            "target_color_mode": str(
                self.config.get("target_color_mode", "white")
            ),
            "latitude": latitude,
            "longitude": longitude,
            "twd97_e": target_e,
            "twd97_n": target_n,
            "position_error_m": position_error,
            "confidence": float(detection["confidence"]),
            "image_base64": str(detection.get("image_base64") or ""),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "frame_width": int(width),
            "frame_height": int(height),
            "bbox_edge_margin_ratio": max(
                0.0,
                min(
                    float(x1) / max(1.0, float(width)),
                    float(y1) / max(1.0, float(height)),
                    float(width - x2) / max(1.0, float(width)),
                    float(height - y2) / max(1.0, float(height)),
                ),
            ),
        }

    def _filter_geolocation(
        self,
        track_id: int,
        detected_at: float,
        latitude: float,
        longitude: float,
        error_m: float,
    ) -> Tuple[float, float, float]:
        record = self._geo_filters.setdefault(
            track_id, {"last_detected_at": 0.0, "samples": []}
        )
        if detected_at > float(record["last_detected_at"]):
            record["last_detected_at"] = detected_at
            record["samples"].append((latitude, longitude, error_m))
            window = max(1, int(self.config["geolocation_filter_window"]))
            record["samples"] = record["samples"][-window:]
        samples = record["samples"] or [(latitude, longitude, error_m)]
        latitudes = np.asarray([item[0] for item in samples], dtype=np.float64)
        longitudes = np.asarray([item[1] for item in samples], dtype=np.float64)
        filtered_lat = float(np.median(latitudes))
        filtered_lon = float(np.median(longitudes))
        dispersion = 0.0
        for sample_lat, sample_lon, _ in samples:
            dispersion = max(
                dispersion,
                wgs84_distance_m(filtered_lat, filtered_lon, sample_lat, sample_lon),
            )
        conservative_error = max(float(np.median([item[2] for item in samples])), dispersion)
        return filtered_lat, filtered_lon, conservative_error

    def _target_offset_ned(
        self,
        detection: Dict[str, Any],
        center_x: float,
        center_y: float,
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        intr = self._latest_intrinsics or {
            "fx": width,
            "fy": width,
            "ppx": width / 2.0,
            "ppy": height / 2.0,
        }
        depth_result: Optional[Tuple[float, float]] = None
        if bool(self.config["publish_depth"]) and self._latest_depth is not None:
            x1, y1, x2, y2 = detection["bbox"]
            depth_height, depth_width = self._latest_depth.shape[:2]
            xa = max(0, min(depth_width, int(x1 + 0.25 * (x2 - x1))))
            xb = max(0, min(depth_width, int(x2 - 0.25 * (x2 - x1))))
            ya = max(0, min(depth_height, int(y1 + 0.20 * (y2 - y1))))
            yb = max(0, min(depth_height, int(y2 - 0.20 * (y2 - y1))))
            if xb > xa and yb > ya:
                values = self._latest_depth[ya:yb, xa:xb].reshape(-1)
                values = values[np.isfinite(values)]
                values = values[
                    (values >= float(self.config["target_min_depth_m"]))
                    & (values <= float(self.config["target_max_depth_m"]))
                ]
                if values.size >= 20:
                    depth = float(np.median(values))
                    mad = float(1.4826 * np.median(np.abs(values - depth)))
                    depth_result = depth, max(0.10, mad)
        fx, fy = float(intr["fx"]), float(intr["fy"])
        ppx, ppy = float(intr["ppx"]), float(intr["ppy"])
        optical_to_frd = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        mount_rpy = [math.radians(float(v)) for v in self.config["camera_mount_rpy_deg"]]
        mount = euler_matrix(*mount_rpy)
        flight = self._telemetry_state
        body_to_ned = euler_matrix(
            float(flight.get("roll") or 0.0),
            float(flight.get("pitch") or 0.0),
            float(flight.get("yaw") or 0.0),
        )
        offset = np.asarray(self.config["camera_offset_frd_m"], dtype=np.float64)
        ray_optical = np.array(
            [(center_x - ppx) / fx, (center_y - ppy) / fy, 1.0], dtype=np.float64
        )
        if depth_result is not None:
            depth, error = depth_result
            body = offset + mount @ (optical_to_frd @ (ray_optical * depth))
            ned = body_to_ned @ body
            return float(ned[0]), float(ned[1]), float(np.linalg.norm(ned)), error
        body_ray = mount @ (optical_to_frd @ ray_optical)
        ned_ray = body_to_ned @ body_ray
        agl = float(flight.get("relative_altitude") or 0.0)
        if agl > 0.5 and float(ned_ray[2]) > 0.05:
            scale = max(0.0, (agl - float((body_to_ned @ offset)[2])) / float(ned_ray[2]))
            ned = body_to_ned @ offset + ned_ray * scale
            distance = float(np.linalg.norm(ned))
            return float(ned[0]), float(ned[1]), distance, max(2.5, distance * 0.12)
        dx, dy = center_x - width / 2.0, center_y - height / 2.0
        scale = float(self.config["meters_per_pixel_at_center"])
        return -dy * scale, dx * scale, math.hypot(dx, dy) * scale, 8.0

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection: Dict[str, Any]) -> None:
        x1, y1, x2, y2 = detection["bbox"]
        track_id = int(detection["track_id"])
        confidence = float(detection["confidence"])
        confirmed = bool(
            detection.get("target_confirmed", detection.get("white_confirmed"))
        )
        visual_held = bool(detection.get("visual_held"))
        color = (
            (0, 220, 255)
            if visual_held
            else ((255, 255, 255) if confirmed else (0, 165, 255))
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ratio = float(detection.get("white_ratio") or 0.0)
        streak = int(detection.get("white_streak") or 0)
        if str(detection.get("target_color_mode", "white")) == "any":
            label = f"CAR ID:{track_id} {confidence:.2f} [{streak}]"
        else:
            label = f"ID:{track_id} {confidence:.2f} W:{ratio:.2f} [{streak}]"
        if visual_held:
            label += " TRACK"
        cv2.rectangle(frame, (x1, max(0, y1 - 27)), (x1 + 260, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def _safe_publish(self, publisher: Any, msg: Any, label: str) -> None:
        try:
            publisher.publish(msg)
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().error(f"{label} publish failed: {exc}")

    def _publish_mavlink_json(self, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self._safe_publish(self.mavlink_pub, msg, "mavlink json")

    def _publish_callback(self) -> None:
        callback_start = time.perf_counter()
        self._tick += 1
        self._telemetry_state = self.telemetry.get() if self.telemetry else {}
        frame, camera_ok = self._read_frame()
        publish_raw = bool(self.config.get("publish_raw_compressed", True))
        if (
            not publish_raw
            and bool(
                self.config.get(
                    "publish_raw_compressed_when_subscribed", False
                )
            )
        ):
            publish_raw = self.raw_image_pub.get_subscription_count() > 0
        publish_annotated = bool(
            self.config.get("publish_annotated_compressed", True)
        )
        raw_frame = frame.copy() if publish_raw else None
        after_read = time.perf_counter()
        if camera_ok:
            frame, detections = self._latest_detection(frame)
        elif self.mode == "mock":
            detections = self._mock_detections(frame)
            detections = detections[: max(0, int(self.config["max_candidates"]))]
            for item in detections:
                self._draw_detection(frame, item)
        else:
            # A live camera outage must fail closed. Mock detections are only
            # valid in explicit mock mode and must never become flight targets.
            detections = []
        now_epoch = time.time()
        target_detections = [
            item
            for item in detections
            if bool(item.get("target_confirmed", item.get("white_confirmed")))
            and now_epoch - float(item.get("detected_at") or 0.0)
            <= float(self.config["target_max_age_s"])
        ]
        if str(self.config.get("target_color_mode", "white")).lower() == "any":
            target_detections.sort(
                key=lambda item: float(item["confidence"]), reverse=True
            )
        else:
            target_detections.sort(
                key=lambda item: (
                    float(item.get("white_ratio") or 0.0),
                    item["confidence"],
                ),
                reverse=True,
            )
        target_detections = target_detections[
            : max(0, int(self.config["max_candidates"]))
        ]
        # Preserve the current confirmed boxes for mission-specific visual
        # servoing. Unlike dashboard candidates, these do not need to keep the
        # same ByteTrack ID for target_confirm_seconds.
        self._latest_confirmed_target_detections = copy.deepcopy(target_detections)
        self._latest_target_frame_shape = tuple(frame.shape)
        gps_ready_for_target = self._gps_fix_valid(self._telemetry_state) or bool(
            self.config["mock_gps"]
        )
        target_position_allowed = gps_ready_for_target or not bool(
            self.config.get("target_position_require_gps", True)
        )
        target_confirm_seconds = max(
            0.0, float(self.config.get("target_confirm_seconds", 3.0))
        )
        for item in target_detections:
            track_id = int(item["track_id"])
            existing = self._candidate_cache.get(track_id, {})
            first_seen = float(existing.get("first_seen_epoch") or item["detected_at"])
            stable_for = max(0.0, now_epoch - first_seen)
            candidate = copy.deepcopy(existing.get("candidate"))
            if (
                candidate is None
                and stable_for >= target_confirm_seconds
                and target_position_allowed
            ):
                candidate = self._candidate_from_detection(item, frame.shape)
                candidate["locked_at_epoch"] = now_epoch
                candidate["stable_for_s"] = stable_for
            elif (
                candidate is not None
                and bool(self.config.get("continuous_candidate_geolocation", False))
                and float(item.get("detected_at") or 0.0)
                > float(existing.get("geolocated_at_epoch") or 0.0)
                and target_position_allowed
            ):
                locked_at = float(candidate.get("locked_at_epoch") or now_epoch)
                # Keep the first confirmed crop as a stable Unity preview.
                # Position/confidence continue updating, but repeatedly sending
                # a changing Base64 JPEG wastes bandwidth and decode work.
                locked_preview = str(candidate.get("image_base64") or "")
                updated_candidate = self._candidate_from_detection(item, frame.shape)
                if locked_preview:
                    updated_candidate["image_base64"] = locked_preview
                candidate = updated_candidate
                candidate["locked_at_epoch"] = locked_at
                candidate["stable_for_s"] = stable_for
            elif candidate is not None:
                candidate["stable_for_s"] = stable_for
            self._candidate_cache[track_id] = {
                "candidate": copy.deepcopy(candidate),
                "last_seen_epoch": float(item["detected_at"]),
                "first_seen_epoch": first_seen,
                "white_ratio": float(item.get("white_ratio") or 0.0),
                "pending": candidate is None,
                "stable_for_s": stable_for,
                "geolocated_at_epoch": (
                    float(item.get("detected_at") or now_epoch)
                    if candidate is not None
                    else float(existing.get("geolocated_at_epoch") or 0.0)
                ),
            }
            if candidate is not None:
                self._candidate_archive[track_id] = copy.deepcopy(
                    self._candidate_cache[track_id]
                )
        age_limit = float(self.config["candidate_display_hold_s"])
        self._candidate_cache = {
            track_id: record
            for track_id, record in self._candidate_cache.items()
            if now_epoch - float(record["last_seen_epoch"]) <= age_limit
        }
        archive_age_limit = max(age_limit, float(self.config["candidate_archive_age_s"]))
        self._candidate_archive = {
            track_id: record
            for track_id, record in self._candidate_archive.items()
            if now_epoch - float(record["last_seen_epoch"]) <= archive_age_limit
        }
        candidates = [
            record["candidate"]
            for record in sorted(
                self._candidate_cache.values(),
                key=lambda r: float(r.get("first_seen_epoch") or 0.0),
            )
            if record.get("candidate") is not None
        ][: max(0, int(self.config["max_candidates"]))]
        for candidate in candidates:
            track_id = int(candidate.get("track_id", -1))
            record = self._candidate_cache.get(track_id) or self._candidate_archive.get(track_id) or {}
            last_seen = float(record.get("last_seen_epoch") or now_epoch)
            age_s = max(0.0, now_epoch - last_seen)
            candidate["age_s"] = age_s
            candidate["display_hold_remaining_s"] = max(0.0, age_limit - age_s)
        self._after_candidates_updated(candidates)
        control_blockers = self._control_blockers(self._telemetry_state)
        confirm_blockers = list(control_blockers)
        if not candidates:
            confirm_blockers.append(
                "no_confirmed_car_target"
                if str(self.config.get("target_color_mode", "white")).lower() == "any"
                else "no_confirmed_white_car_target"
            )
        confirm_blockers.extend(self._candidate_confirm_blockers(candidates))
        mode_text = (
            f"LIVE {self.model_source}"
            if camera_ok and self.model_source not in ("mock", "failed", "none")
            else "MOCK/FALLBACK"
        )
        cv2.putText(
            frame,
            mode_text,
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        # Send a candidate crop immediately, then only occasionally for late
        # Unity connections. The client caches it by candidate ID between the
        # lightweight position updates.
        preview_now = time.monotonic()
        preview_interval = max(
            0.2, float(self.config.get("candidate_image_resend_interval_s", 5.0))
        )
        published_candidates = copy.deepcopy(candidates)
        active_preview_keys = set()
        for candidate in published_candidates:
            preview_key = str(candidate.get("candidate_id") or "")
            if not preview_key:
                continue
            active_preview_keys.add(preview_key)
            snapshot = str(candidate.get("image_base64") or "")
            last_sent = float(
                self._candidate_preview_last_sent.get(preview_key, 0.0)
            )
            if snapshot and preview_now - last_sent >= preview_interval:
                self._candidate_preview_last_sent[preview_key] = preview_now
            else:
                candidate["image_base64"] = ""
        self._candidate_preview_last_sent = {
            key: sent_at
            for key, sent_at in self._candidate_preview_last_sent.items()
            if key in active_preview_keys
        }

        detection_msg = String()
        detection_payload = {
                "source_session_id": self.source_session_id,
                "timestamp": time.time(),
                "candidates": published_candidates,
                "candidate_count": len(candidates),
                "ready_for_confirm": bool(candidates) and not confirm_blockers,
                "confirm_blockers": confirm_blockers,
                "target_confirm_seconds": float(
                    self.config.get("target_confirm_seconds", 3.0)
                ),
                "candidate_display_hold_s": float(
                    self.config.get("candidate_display_hold_s", 10.0)
                ),
                "debug_detections": [
                    {
                        "track_id": int(item["track_id"]),
                        "confidence": float(item.get("confidence") or 0.0),
                        "white_ratio": float(item.get("white_ratio") or 0.0),
                        "white_pixels": int(item.get("white_pixels") or 0),
                        "red_ratio": float(item.get("red_ratio") or 0.0),
                        "edge_density": float(item.get("edge_density") or 0.0),
                        "aspect_ratio": float(item.get("aspect_ratio") or 0.0),
                        "white_streak": int(item.get("white_streak") or 0),
                        "white_confirmed": bool(item.get("white_confirmed")),
                        "target_confirmed": bool(
                            item.get("target_confirmed", item.get("white_confirmed"))
                        ),
                        "bbox": list(item.get("bbox") or []),
                    }
                    for item in detections[: max(0, int(self.config["max_candidates"]))]
                ],
                "detector_source": self.model_source,
                "camera_ok": bool(camera_ok),
            }
        detection_payload.update(self._detection_payload_extras(candidates))
        detection_msg.data = json.dumps(
            detection_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._safe_publish(self.detections_pub, detection_msg, "detections")
        if raw_frame is not None:
            self._publish_image(
                raw_frame,
                self.raw_image_pub,
                quality=int(self.config.get("raw_jpeg_quality", 70)),
                max_width=int(
                    self.config.get("raw_publish_image_max_width", 0) or 0
                ),
                max_height=int(
                    self.config.get("raw_publish_image_max_height", 0) or 0
                ),
            )
        if publish_annotated:
            self._publish_image(frame, self.image_pub)
        after_image = time.perf_counter()
        self._publish_d455_data()
        after_d455 = time.perf_counter()
        self._publish_gps()
        self._publish_imu()
        self._publish_flight_telemetry()
        callback_end = time.perf_counter()
        if self._tick % 20 == 0 and callback_end - callback_start > 0.25:
            self.get_logger().warning(
                "Slow publish callback %.0f ms (read %.0f, image %.0f, d455 %.0f, telemetry %.0f)"
                % (
                    1000 * (callback_end - callback_start),
                    1000 * (after_read - callback_start),
                    1000 * (after_image - after_read),
                    1000 * (after_d455 - after_image),
                    1000 * (callback_end - after_d455),
                )
            )

    def _publish_image(
        self,
        frame: np.ndarray,
        publisher: Any,
        quality: Optional[int] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> None:
        if quality is None:
            quality = int(self.config["jpeg_quality"])
        quality = max(1, min(100, int(quality)))
        output = frame
        if max_width is None:
            max_width = int(self.config.get("publish_image_max_width") or 0)
        if max_height is None:
            max_height = int(self.config.get("publish_image_max_height") or 0)
        max_width = max(0, int(max_width))
        max_height = max(0, int(max_height))
        if max_width > 0 or max_height > 0:
            height, width = frame.shape[:2]
            scale = 1.0
            if max_width > 0 and width > max_width:
                scale = min(scale, max_width / float(width))
            if max_height > 0 and height > max_height:
                scale = min(scale, max_height / float(height))
            if scale < 1.0:
                new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                output = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            self.get_logger().error("JPEG encoding failed")
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        try:
            self._safe_publish(publisher, msg, "compressed image")
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().error(f"Compressed image publish failed: {exc}")

    def _gps_fix_valid(self, data: Dict[str, Any]) -> bool:
        if not data:
            return False
        age = time.monotonic() - float(data.get("gps_update_monotonic") or 0.0)
        return (
            bool(data.get("connected"))
            and int(data.get("fix_type") or 0) >= int(self.config["gps_min_fix_type"])
            and data.get("latitude") is not None
            and data.get("longitude") is not None
            and age <= float(self.config["gps_stale_timeout_s"])
        )

    def _camera_connected(self) -> bool:
        if self.realsense is not None:
            return bool(self.realsense.connected)
        if self.rtsp_camera is not None:
            return bool(self.rtsp_camera.connected)
        if self.capture is not None:
            return bool(self.capture.isOpened())
        return False

    def _publish_d455_data(self) -> None:
        stamp = self.get_clock().now().to_msg()
        if self._latest_depth is not None:
            now = time.monotonic()
            interval = 1.0 / max(0.01, float(self.config["depth_publish_rate_hz"]))
            if now - self._last_depth_submit >= interval:
                self._last_depth_submit = now
                with self._depth_lock:
                    self._depth_pending = (self._latest_depth.copy(), stamp)
                self._depth_event.set()
        intr = self._latest_intrinsics
        publish_camera_info = bool(self.config["publish_camera_info"])
        if (
            not publish_camera_info
            and bool(
                self.config.get(
                    "publish_camera_info_when_subscribed", False
                )
            )
        ):
            publish_camera_info = self.camera_info_pub.get_subscription_count() > 0
        if publish_camera_info and intr:
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = "d455i_color_optical_frame"
            info.width, info.height = int(intr["width"]), int(intr["height"])
            fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
            info.distortion_model = "plumb_bob"
            info.k = [fx, 0.0, ppx, 0.0, fy, ppy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, ppx, 0.0, 0.0, fy, ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.camera_info_pub.publish(info)
        if bool(self.config["publish_d455_imu"]):
            d455_msg = Imu()
            d455_msg.header.stamp = stamp
            d455_msg.header.frame_id = "d455i_imu"
            d455_msg.orientation_covariance[0] = -1.0
            av = self._latest_d455_imu.get("angular_velocity", (0.0, 0.0, 0.0))
            la = self._latest_d455_imu.get("linear_acceleration", (0.0, 0.0, 0.0))
            d455_msg.angular_velocity.x, d455_msg.angular_velocity.y, d455_msg.angular_velocity.z = map(float, av)
            d455_msg.linear_acceleration.x, d455_msg.linear_acceleration.y, d455_msg.linear_acceleration.z = map(float, la)
            self.d455_imu_pub.publish(d455_msg)

    def _depth_publish_loop(self) -> None:
        while self._depth_running.is_set():
            self._depth_event.wait(timeout=0.5)
            self._depth_event.clear()
            if not self._depth_running.is_set():
                break
            with self._depth_lock:
                pending = self._depth_pending
                self._depth_pending = None
            if pending is None:
                continue
            depth, stamp = pending
            depth = np.ascontiguousarray(depth.astype(np.float32, copy=False))
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = "d455i_color_optical_frame"
            msg.height, msg.width = depth.shape[:2]
            msg.encoding = "32FC1"
            msg.is_bigendian = False
            msg.step = int(depth.strides[0])
            msg.data = depth.tobytes()
            try:
                self.depth_pub.publish(msg)
            except Exception as exc:
                if self._depth_running.is_set() and rclpy.ok():
                    self.get_logger().error(f"Depth publish failed: {exc}")

    def _publish_gps(self) -> None:
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "flight_controller_gps"
        msg.status.service = NavSatStatus.SERVICE_GPS
        if bool(self.config["mock_gps"]):
            msg.status.status = NavSatStatus.STATUS_FIX
            msg.latitude = float(self.config["default_latitude"])
            msg.longitude = float(self.config["default_longitude"])
            msg.altitude = float(self.config["default_altitude"])
            horizontal_error, vertical_error = 2.5, 4.0
        elif self._gps_fix_valid(self._telemetry_state):
            msg.status.status = NavSatStatus.STATUS_FIX
            msg.latitude = float(self._telemetry_state["latitude"])
            msg.longitude = float(self._telemetry_state["longitude"])
            msg.altitude = float(self._telemetry_state.get("altitude") or float("nan"))
            horizontal_error = float(self._telemetry_state.get("eph_m") or 5.0)
            vertical_error = float(self._telemetry_state.get("epv_m") or 8.0)
            self._last_valid_gps = {
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "altitude": msg.altitude,
                "horizontal_error": horizontal_error,
                "vertical_error": vertical_error,
                "monotonic": time.monotonic(),
            }
        elif self._last_valid_gps is not None and (
            time.monotonic() - float(self._last_valid_gps.get("monotonic") or 0.0)
        ) <= float(self.config["gps_display_hold_s"]):
            msg.status.status = NavSatStatus.STATUS_FIX
            msg.latitude = float(self._last_valid_gps["latitude"])
            msg.longitude = float(self._last_valid_gps["longitude"])
            msg.altitude = float(self._last_valid_gps.get("altitude") or float("nan"))
            horizontal_error = float(self._last_valid_gps.get("horizontal_error") or 5.0)
            vertical_error = float(self._last_valid_gps.get("vertical_error") or 8.0)
        else:
            msg.status.status = NavSatStatus.STATUS_NO_FIX
            msg.latitude = msg.longitude = msg.altitude = float("nan")
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            self._safe_publish(self.gps_pub, msg, "gps")
            return
        msg.position_covariance = [
            horizontal_error**2, 0.0, 0.0,
            0.0, horizontal_error**2, 0.0,
            0.0, 0.0, vertical_error**2,
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._safe_publish(self.gps_pub, msg, "gps")

    def _publish_imu(self) -> None:
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        state = self._telemetry_state
        if state.get("connected") and not bool(self.config["mock_imu"]):
            msg.header.frame_id = "aircraft_frd"
            qx, qy, qz, qw = euler_to_quaternion(
                float(state.get("roll") or 0.0),
                float(state.get("pitch") or 0.0),
                float(state.get("yaw") or 0.0),
            )
            msg.orientation.x, msg.orientation.y = qx, qy
            msg.orientation.z, msg.orientation.w = qz, qw
            msg.angular_velocity.x = float(state.get("rollspeed") or 0.0)
            msg.angular_velocity.y = float(state.get("pitchspeed") or 0.0)
            msg.angular_velocity.z = float(state.get("yawspeed") or 0.0)
            msg.linear_acceleration.x = float(state.get("xacc") or 0.0)
            msg.linear_acceleration.y = float(state.get("yacc") or 0.0)
            msg.linear_acceleration.z = float(state.get("zacc") or 0.0)
        else:
            msg.header.frame_id = "d455i_imu_fallback"
            msg.orientation.w = 1.0
            msg.orientation_covariance[0] = -1.0
            av = self._latest_d455_imu.get("angular_velocity", (0.0, 0.0, 0.0))
            la = self._latest_d455_imu.get("linear_acceleration", (0.0, 0.0, 0.0))
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, av)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, la)
        self._safe_publish(self.imu_pub, msg, "imu")

    def _publish_flight_telemetry(self) -> None:
        state = self._telemetry_state
        stamp = self.get_clock().now().to_msg()
        velocity = TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = "ned"
        velocity.twist.linear.x = float(state.get("vx") or 0.0)
        velocity.twist.linear.y = float(state.get("vy") or 0.0)
        velocity.twist.linear.z = float(state.get("vz") or 0.0)
        self._safe_publish(self.velocity_pub, velocity, "velocity")
        battery = BatteryState()
        battery.header.stamp = stamp
        battery.header.frame_id = "aircraft"
        battery.voltage = float(state.get("voltage_v") or float("nan"))
        battery.current = float(state.get("current_a") or float("nan"))
        remaining = state.get("battery_remaining_pct")
        battery.percentage = float(remaining) / 100.0 if remaining is not None else float("nan")
        self._safe_publish(self.battery_pub, battery, "battery")
        safe_state = {key: value for key, value in state.items() if key != "gps_update_monotonic"}
        control_blockers = self._control_blockers(state)
        gps_age_s = time.monotonic() - float(state.get("gps_update_monotonic") or 0.0)
        display_gps_age_s = None
        display_gps = None
        if self._last_valid_gps is not None:
            display_gps_age_s = time.monotonic() - float(
                self._last_valid_gps.get("monotonic") or 0.0
            )
            if display_gps_age_s <= float(self.config["gps_display_hold_s"]):
                display_gps = {
                    key: self._last_valid_gps.get(key)
                    for key in ("latitude", "longitude", "altitude", "horizontal_error")
                }
        if bool(self.config.get("unity_map_preview", False)):
            display_gps_age_s = 0.0
            display_gps = {
                "latitude": float(self.config["default_latitude"]),
                "longitude": float(self.config["default_longitude"]),
                "altitude": float(self.config["default_altitude"]),
                "horizontal_error": 0.0,
            }
        safe_state.update(
            {
                "source_session_id": self.source_session_id,
                "timestamp": time.time(),
                "gps_valid": self._gps_fix_valid(state),
                "gps_age_s": gps_age_s,
                "gps_display_valid": display_gps is not None,
                "gps_display_age_s": display_gps_age_s,
                "gps_display": display_gps,
                "map_position_valid": display_gps is not None,
                "display_latitude": (
                    display_gps.get("latitude") if display_gps is not None else None
                ),
                "display_longitude": (
                    display_gps.get("longitude") if display_gps is not None else None
                ),
                "display_altitude": (
                    display_gps.get("altitude") if display_gps is not None else None
                ),
                "display_horizontal_error_m": (
                    display_gps.get("horizontal_error") if display_gps is not None else None
                ),
                "camera_source": str(self.config.get("camera_source", "")),
                "camera_connected": self._camera_connected(),
                "d455i_connected": bool(self.realsense and self.realsense.connected),
                "g3p_connected": bool(self.rtsp_camera and self.rtsp_camera.connected),
                "d455i_imu_available": bool(self._latest_d455_imu),
                "detector_source": self.model_source,
                "flight_control_enabled": bool(self.config["enable_flight_control"]),
                "flight_control_ready": not control_blockers,
                "flight_control_blockers": control_blockers,
            }
        )
        safe_state.update(self._telemetry_payload_extras())
        telemetry_msg = String()
        telemetry_msg.data = json.dumps(safe_state, separators=(",", ":"), allow_nan=False)
        self._safe_publish(self.telemetry_pub, telemetry_msg, "telemetry")

    def _after_candidates_updated(self, candidates: List[Dict[str, Any]]) -> None:
        """Extension hook used by mission-specific bridge programs."""

    def _candidate_confirm_blockers(
        self, candidates: List[Dict[str, Any]]
    ) -> List[str]:
        return []

    def _detection_payload_extras(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {}

    def _telemetry_payload_extras(self) -> Dict[str, Any]:
        return {}

    def _control_blockers(self, state: Dict[str, Any]) -> List[str]:
        if not bool(self.config["enable_flight_control"]):
            return ["flight_control_disabled_by_config"]
        blockers = []
        if not self._camera_connected():
            blockers.append("camera_unavailable")
        if self.model_source not in ("TensorRT", "PyTorch fallback"):
            blockers.append("detector_not_ready")
        if not state.get("connected"):
            blockers.append("mavlink_disconnected")
        if not self._gps_fix_valid(state):
            blockers.append("gps_invalid_or_stale")
        if int(state.get("satellites") or 0) < int(self.config["minimum_gps_satellites"]):
            blockers.append("insufficient_satellites")
        eph = state.get("eph_m")
        if eph is None or float(eph) > float(self.config["maximum_gps_eph_m"]):
            blockers.append("gps_eph_too_high")
        if not bool(state.get("armed")):
            blockers.append("aircraft_not_armed")
        mode = str(state.get("flight_mode") or "").upper()
        allowed = {str(value).upper() for value in self.config["allowed_flight_modes"]}
        auto_switch_allowed = (
            bool(self.config["auto_switch_to_guided"])
            and mode
            in {
                str(value).upper()
                for value in self.config["auto_switch_guided_from_modes"]
            }
        )
        if mode not in allowed and not auto_switch_allowed:
            blockers.append("flight_mode_not_allowed")
        if float(state.get("relative_altitude") or 0.0) < float(
            self.config["minimum_flight_altitude_m"]
        ):
            blockers.append("below_minimum_altitude")
        return blockers

    def _publish_status(
        self,
        command_id: str,
        track_id: int,
        status: str,
        message: str,
        distance_remaining_m: float,
    ) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "command_id": command_id,
                "track_id": int(track_id),
                "status": status,
                "message": message,
                "distance_remaining_m": float(distance_remaining_m),
            },
            separators=(",", ":"),
        )
        self._safe_publish(self.status_pub, msg, "target command status")
        self.get_logger().info(f"Command {command_id}: {status}")

    def _command_callback(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
            if not isinstance(command, dict):
                raise ValueError("command must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f"Invalid target command: {exc}")
            return

        action = str(command.get("action", "")).upper()
        command_id = str(command.get("command_id", ""))
        if action in ("GOTO_TARGET", "CONFIRM"):
            try:
                track_id = int(command.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1
            self._publish_status(
                command_id,
                track_id,
                "RECEIVED",
                "Orin NX received target command",
                0.0,
            )
            if self.mode == "mock":
                self._active_command = {
                    "command_id": command_id,
                    "track_id": track_id,
                    "distance": 30.0,
                    "simulated": True,
                    "last_status_time": time.monotonic(),
                }
                self._publish_status(
                    command_id, track_id, "ACCEPTED", "Mock target accepted", 30.0
                )
                return
            if bool(self.config["simulate_flight_commands"]):
                record, resolved_track_id, reason = self._candidate_record_for_command(
                    command, track_id
                )
                if record is None:
                    self._publish_status(
                        command_id,
                        track_id,
                        "REJECTED",
                        reason,
                        0.0,
                    )
                    return
                age = time.time() - float(record["last_seen_epoch"])
                candidate = record["candidate"]
                if age > float(self.config["target_max_age_s"]):
                    self._publish_status(
                        command_id, track_id, "REJECTED", "Target expired", 0.0
                    )
                    return
                self._active_command = {
                    "command_id": command_id,
                    "track_id": resolved_track_id,
                    "distance": 30.0,
                    "simulated": True,
                    "last_status_time": time.monotonic(),
                }
                self._publish_status(
                    command_id,
                    resolved_track_id,
                    "ACCEPTED",
                    f"Indoor simulation accepted; AP6 not contacted ({reason})",
                    30.0,
                )
                return
            candidate, distance, reason = self._validate_goto(command, track_id)
            if candidate is None:
                self._publish_status(
                    command_id, track_id, "REJECTED", reason, max(0.0, distance)
                )
                return
            assert self.telemetry is not None
            mode_ok, mode_message = self.telemetry.ensure_guided_for_reposition()
            if not mode_ok:
                self._publish_status(
                    command_id, track_id, "REJECTED", mode_message, distance
                )
                return
            self._telemetry_state = self.telemetry.get()
            target_relative_altitude = float(
                command.get("target_relative_altitude")
                or self.config.get("goto_relative_altitude_m", 20.0)
            )
            sent, send_message = self.telemetry.send_reposition(
                float(candidate["latitude"]),
                float(candidate["longitude"]),
                target_relative_altitude,
            )
            if not sent:
                self._publish_status(
                    command_id, track_id, "REJECTED", send_message, distance
                )
                return
            self._active_command = {
                "command_id": command_id,
                "track_id": int(candidate["track_id"]),
                "candidate_id": candidate["candidate_id"],
                "target_latitude": float(candidate["latitude"]),
                "target_longitude": float(candidate["longitude"]),
                "target_relative_altitude": target_relative_altitude,
                "distance": distance,
                "started_monotonic": time.monotonic(),
                "last_status_time": time.monotonic(),
                "arrival_hits": 0,
                "simulated": False,
            }
            self._publish_status(
                command_id,
                int(candidate["track_id"]),
                "ACCEPTED",
                f"Validated car target; {mode_message}; {send_message}",
                distance,
            )
        elif action == "CANCEL":
            active = self._active_command or {}
            track_id = int(command.get("track_id", active.get("track_id", -1)))
            response_id = command_id or str(active.get("command_id", ""))
            message = "Target command cancelled"
            if active and not active.get("simulated") and self.telemetry is not None:
                held, hold_message = self.telemetry.send_hold_current_position()
                message = (
                    f"Cancelled; {hold_message}"
                    if held
                    else f"Cancelled locally; hold failed: {hold_message}"
                )
            self._active_command = None
            self._publish_status(
                response_id, track_id, "CANCELLED", message, 0.0
            )
        else:
            self.get_logger().warning(f"Unknown command action: {action!r}")

    def _validate_goto(
        self,
        command: Dict[str, Any],
        track_id: int,
        require_target_position: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        if not bool(self.config["enable_flight_control"]):
            return None, 0.0, "Flight control disabled by current config"
        if self.telemetry is None or not self._telemetry_state.get("connected"):
            return None, 0.0, "Flight controller telemetry is not connected"
        record, resolved_track_id, lookup_reason = self._candidate_record_for_command(
            command, track_id
        )
        if record is None:
            return None, 0.0, lookup_reason
        age = time.time() - float(record["last_seen_epoch"])
        if age > float(self.config["target_max_age_s"]):
            return None, 0.0, f"Target expired ({age:.1f} s old)"
        candidate = copy.deepcopy(record["candidate"])
        track_id = resolved_track_id
        if not self._gps_fix_valid(self._telemetry_state):
            return None, 0.0, "GPS fix is invalid or stale"
        if int(self._telemetry_state.get("satellites") or 0) < int(
            self.config["minimum_gps_satellites"]
        ):
            return None, 0.0, "Not enough GPS satellites"
        eph = self._telemetry_state.get("eph_m")
        if eph is None or float(eph) > float(self.config["maximum_gps_eph_m"]):
            return None, 0.0, "GPS horizontal error is too high"
        if not bool(self._telemetry_state.get("armed")):
            return None, 0.0, "Aircraft is not armed"
        mode = str(self._telemetry_state.get("flight_mode") or "").upper()
        allowed = {str(value).upper() for value in self.config["allowed_flight_modes"]}
        auto_switch_allowed = (
            bool(self.config["auto_switch_to_guided"])
            and mode
            in {
                str(value).upper()
                for value in self.config["auto_switch_guided_from_modes"]
            }
        )
        if mode not in allowed and not auto_switch_allowed:
            return None, 0.0, f"Flight mode {mode or 'UNKNOWN'} is not allowed"
        relative_altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        if relative_altitude < float(self.config["minimum_flight_altitude_m"]):
            return None, 0.0, "Aircraft is below minimum control altitude"
        if self._telemetry_state.get("altitude") is None:
            return None, 0.0, "MSL altitude is unavailable"
        distance = 0.0
        if require_target_position:
            if float(candidate["position_error_m"]) > float(
                self.config["maximum_target_error_m"]
            ):
                return None, 0.0, "Target position error is too high"
            distance = wgs84_distance_m(
                float(self._telemetry_state["latitude"]),
                float(self._telemetry_state["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            if distance > float(self.config["maximum_goto_distance_m"]):
                return None, distance, "Target exceeds maximum goto distance"
        return candidate, distance, "validated"

    def _candidate_record_for_command(
        self, command: Dict[str, Any], track_id: int
    ) -> Tuple[Optional[Dict[str, Any]], int, str]:
        requested_id = str(command.get("candidate_id") or "")
        records = {
            **self._candidate_archive,
            **self._candidate_cache,
        }
        if track_id >= 0 and track_id in records:
            record = records[track_id]
            candidate = record["candidate"]
            if candidate is None:
                stable_for = float(record.get("stable_for_s") or 0.0)
                needed = float(self.config.get("target_confirm_seconds", 3.0))
                return (
                    None,
                    track_id,
                    f"target is still confirming ({stable_for:.1f}/{needed:.1f} s)",
                )
            if requested_id and requested_id != str(candidate.get("candidate_id", "")):
                return None, track_id, "candidate_id and track_id do not match"
            return record, track_id, "matched_by_track_id"

        if requested_id:
            for archived_track_id, record in records.items():
                candidate = record["candidate"]
                if candidate is None:
                    continue
                if requested_id == str(candidate.get("candidate_id", "")):
                    return record, int(archived_track_id), "matched_by_candidate_id"
            return None, track_id, "candidate_id is not a current confirmed car"

        command_lat = command.get("target_latitude", command.get("latitude"))
        command_lon = command.get("target_longitude", command.get("longitude"))
        if command_lat is not None and command_lon is not None:
            try:
                target_lat = float(command_lat)
                target_lon = float(command_lon)
            except (TypeError, ValueError):
                return None, track_id, "target latitude/longitude are invalid"
            best_track_id = -1
            best_record: Optional[Dict[str, Any]] = None
            best_distance = float("inf")
            now = time.time()
            max_age = float(self.config["target_max_age_s"])
            for archived_track_id, record in records.items():
                if now - float(record["last_seen_epoch"]) > max_age:
                    continue
                candidate = record["candidate"]
                if candidate is None:
                    continue
                distance = wgs84_distance_m(
                    target_lat,
                    target_lon,
                    float(candidate["latitude"]),
                    float(candidate["longitude"]),
                )
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = int(archived_track_id)
                    best_record = record
            if (
                best_record is not None
                and best_distance <= float(self.config["command_target_match_distance_m"])
            ):
                return (
                    best_record,
                    best_track_id,
                    f"matched_by_target_position_{best_distance:.1f}m",
                )
            return None, track_id, "target position does not match a current car"

        return None, track_id, "Track ID is not a current confirmed car"

    def _command_status_tick(self) -> None:
        if self._active_command is None:
            return
        command = self._active_command
        now = time.monotonic()
        if now - float(command["last_status_time"]) < 1.0:
            return
        command["last_status_time"] = now
        if command.get("simulated"):
            command["distance"] = max(0.0, float(command["distance"]) - 5.0)
            if command["distance"] <= 0.0:
                self._publish_status(
                    command["command_id"], command["track_id"], "ARRIVED", "Mock arrived", 0.0
                )
                self._active_command = None
            else:
                self._publish_status(
                    command["command_id"],
                    command["track_id"],
                    "EXECUTING",
                    "Mock flight to target",
                    command["distance"],
                )
            return
        state = self.telemetry.get() if self.telemetry is not None else {}
        if not state.get("connected") or state.get("latitude") is None:
            self._publish_status(
                command["command_id"],
                command["track_id"],
                "FAILED",
                "Flight telemetry or GPS was lost",
                float(command["distance"]),
            )
            self._active_command = None
            return
        distance = wgs84_distance_m(
            float(state["latitude"]),
            float(state["longitude"]),
            float(command["target_latitude"]),
            float(command["target_longitude"]),
        )
        command["distance"] = distance
        if time.monotonic() - float(command["started_monotonic"]) > float(
            self.config["goto_timeout_s"]
        ):
            if self.telemetry is not None:
                self.telemetry.send_hold_current_position()
            self._publish_status(
                command["command_id"], command["track_id"], "FAILED", "Goto timed out", distance
            )
            self._active_command = None
            return
        target_relative_altitude = float(
            command.get("target_relative_altitude")
            or self.config.get("goto_relative_altitude_m", 20.0)
        )
        altitude_error = abs(
            float(state.get("relative_altitude") or 0.0) - target_relative_altitude
        )
        if (
            distance <= float(self.config["arrival_radius_m"])
            and altitude_error
            <= float(self.config.get("arrival_altitude_tolerance_m", 3.0))
        ):
            command["arrival_hits"] = int(command["arrival_hits"]) + 1
        else:
            command["arrival_hits"] = 0
        if int(command["arrival_hits"]) >= int(self.config["arrival_confirm_seconds"]):
            self._publish_status(
                command["command_id"],
                command["track_id"],
                "ARRIVED",
                "Aircraft reached the selected car",
                distance,
            )
            self._active_command = None
        else:
            self._publish_status(
                command["command_id"],
                command["track_id"],
                "EXECUTING",
                "AP6 flying to selected car",
                distance,
            )

    def destroy_node(self) -> bool:
        # Stop hardware readers first so they cannot log/publish after ROS's
        # signal handler invalidates the context.
        if self.telemetry is not None:
            self.telemetry.stop()
        if self.realsense is not None:
            self.realsense.stop()
        if self.rtsp_camera is not None:
            self.rtsp_camera.stop()
        self._depth_running.clear()
        self._depth_event.set()
        self._depth_thread.join(timeout=3)
        if self.process_detector is not None:
            self.process_detector.stop()
        self._detector_running.clear()
        self._detector_event.set()
        if hasattr(self, "_detector_thread"):
            self._detector_thread.join(timeout=3)
        if self.capture is not None:
            self.capture.release()
        return super().destroy_node()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a YAML mapping")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--mode", choices=("live", "mock"), help="override config mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    rclpy.init()
    node: Optional[UnityDashboardBridge] = None
    try:
        node = UnityDashboardBridge(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
