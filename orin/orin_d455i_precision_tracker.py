#!/usr/bin/env python3
"""All-in-one Orin D455i car tracking, geolocation, transmission, and logging.

The HTTP/Socket.IO API remains compatible with ``pi/gui.py``:

* ``/video`` contains only new-car event stills zooming out from the car to the
  complete camera frame. Continuous camera video is never sent over the network.
* ``/status`` reports flight-controller state and the latest detection.
* Socket.IO namespace ``/events`` emits ``detection`` and ``gps`` events.

Target coordinates are not copied from the aircraft GPS.  The program aligns
D455 depth to color, robustly estimates range inside the car box, deprojects the
pixel with the measured camera intrinsics, applies camera mounting and MAVLink
attitude rotations, and finally converts the NED offset to WGS84.

Typical use on this Orin:

    python3 orin_d455i_precision_tracker.py

Hardware-free verification:

    python3 orin_d455i_precision_tracker.py --fake-camera --mock-flight \
        --no-server --run-seconds 12
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO


MODEL_DEFAULT = "/home/hrc/Downloads/VisDrone_train1_5090best.pt"
CAR_CLASS_NAME = "car"
EARTH_RADIUS_M = 6_378_137.0

log = logging.getLogger("orin_precision_tracker")
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
    """Explicitly accept the namespace used by the ground station."""
    return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class JsonlWriter:
    """Thread-safe, line-buffered JSONL audit log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(record, default=json_safe, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._file.close()


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
    """FRD body to NED, or a local intrinsic XYZ mounting rotation."""
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass
class CameraSample:
    color: np.ndarray
    depth_m: Optional[np.ndarray]
    intrinsics: Intrinsics
    camera_timestamp_ms: float
    imu: Dict[str, Any] = field(default_factory=dict)


class ROS2Publisher:
    """Publish standard ROS 2 messages for recording with ros2 bag record."""

    def __init__(
        self,
        compressed_only: bool = False,
        image_fps: float = 0.5,
        jpeg_quality: int = 50,
    ):
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import BatteryState, CameraInfo, CompressedImage, Image, Imu, NavSatFix
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self.node = rclpy.create_node("orin_d455i_precision_tracker")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(depth=20)
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        image_record_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.compressed_only = compressed_only
        self.image_interval_s = 1.0 / max(float(image_fps), 0.01)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self._last_image_publish = 0.0

        self.compressed_annotated_pub = self.node.create_publisher(
            CompressedImage, "/d455/color/image_annotated/compressed", reliable_qos
        )
        self.compressed_tx_pub = self.node.create_publisher(
            CompressedImage, "/uav/transmission/image/compressed", event_qos
        )

        self.color_pub = self.node.create_publisher(
            Image, "/d455/color/image_raw", image_record_qos
        )
        self.annotated_pub = self.node.create_publisher(
            Image, "/d455/color/image_annotated", image_record_qos
        )
        self.depth_pub = self.node.create_publisher(
            Image, "/d455/depth/image_rect_raw", image_record_qos
        )
        self.camera_info_pub = self.node.create_publisher(
            CameraInfo, "/d455/color/camera_info", image_record_qos
        )
        self.d455_imu_pub = self.node.create_publisher(Imu, "/d455/imu", sensor_qos)
        self.gps_pub = self.node.create_publisher(NavSatFix, "/uav/gps/fix", sensor_qos)
        self.uav_imu_pub = self.node.create_publisher(Imu, "/uav/imu", sensor_qos)
        self.velocity_pub = self.node.create_publisher(
            TwistStamped, "/uav/velocity_ned", sensor_qos
        )
        self.battery_pub = self.node.create_publisher(BatteryState, "/uav/battery", sensor_qos)
        self.flight_pub = self.node.create_publisher(String, "/uav/flight_state", reliable_qos)
        self.detection_pub = self.node.create_publisher(
            String, "/uav/car_detections", reliable_qos
        )
        self.target_fix_pub = self.node.create_publisher(
            NavSatFix, "/uav/target/fix", event_qos
        )
        self.target_detail_pub = self.node.create_publisher(
            String, "/uav/target/geolocation", event_qos
        )
        self.tx_image_pub = self.node.create_publisher(
            Image, "/uav/transmission/image", reliable_qos
        )
        self.tx_progress_pub = self.node.create_publisher(
            String, "/uav/transmission/progress", reliable_qos
        )

        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._spin, name="ros2-publisher", daemon=True)
        self._thread.start()
        log.info("ROS 2 publishers ready on node %s", self.node.get_name())
        if self.compressed_only:
            log.info(
                "ROS 2 compressed-only mode: annotated JPEG %.2f fps, quality=%d",
                1.0 / self.image_interval_s,
                self.jpeg_quality,
            )

    def _spin(self) -> None:
        while self._running.is_set() and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.1)

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    @staticmethod
    def _image_message(frame: np.ndarray, encoding: str, frame_id: str, stamp):
        from sensor_msgs.msg import Image

        array = np.ascontiguousarray(frame)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(array.shape[0])
        msg.width = int(array.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(array.strides[0])
        msg.data = array.tobytes()
        return msg

    def _compressed_message(self, frame: np.ndarray, frame_id: str, stamp):
        from sensor_msgs.msg import CompressedImage

        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return None
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        return msg

    @staticmethod
    def _quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    @staticmethod
    def _json_message(payload: Dict[str, Any]):
        from std_msgs.msg import String

        msg = String()
        msg.data = json.dumps(payload, default=json_safe, separators=(",", ":"))
        return msg

    @staticmethod
    def _navsat_message(position: Dict[str, Any], stamp, frame_id: str):
        from sensor_msgs.msg import NavSatFix, NavSatStatus

        msg = NavSatFix()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        fix_type = int(position.get("fix_type") or position.get("aircraft_fix_type") or 0)
        msg.status.status = NavSatStatus.STATUS_FIX if fix_type >= 2 else NavSatStatus.STATUS_NO_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        msg.latitude = float(position.get("latitude") or float("nan"))
        msg.longitude = float(position.get("longitude") or float("nan"))
        msg.altitude = float(position.get("altitude") or float("nan"))
        horizontal_sigma = position.get("estimated_horizontal_sigma_m") or position.get("gps_eph_m")
        vertical_sigma = position.get("gps_epv_m")
        if horizontal_sigma is not None:
            hs2 = float(horizontal_sigma) ** 2
            vs2 = float(vertical_sigma if vertical_sigma is not None else horizontal_sigma) ** 2
            msg.position_covariance = [hs2, 0.0, 0.0, 0.0, hs2, 0.0, 0.0, 0.0, vs2]
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        return msg

    def _publish_flight_and_detections(
        self,
        flight: Dict[str, Any],
        detections: List[Dict[str, Any]],
        frame_index: int,
        stamp,
    ) -> None:
        from geometry_msgs.msg import TwistStamped
        from sensor_msgs.msg import BatteryState, Imu

        if flight.get("latitude") is not None:
            self.gps_pub.publish(self._navsat_message(flight, stamp, "pixhawk_gps"))

        uav_imu = Imu()
        uav_imu.header.stamp = stamp
        uav_imu.header.frame_id = "aircraft_frd"
        qx, qy, qz, qw = self._quaternion(
            float(flight.get("roll") or 0.0),
            float(flight.get("pitch") or 0.0),
            float(flight.get("yaw") or 0.0),
        )
        uav_imu.orientation.x, uav_imu.orientation.y = qx, qy
        uav_imu.orientation.z, uav_imu.orientation.w = qz, qw
        uav_imu.angular_velocity.x = float(flight.get("rollspeed") or 0.0)
        uav_imu.angular_velocity.y = float(flight.get("pitchspeed") or 0.0)
        uav_imu.angular_velocity.z = float(flight.get("yawspeed") or 0.0)
        self.uav_imu_pub.publish(uav_imu)

        velocity = TwistStamped()
        velocity.header.stamp = stamp
        velocity.header.frame_id = "ned"
        velocity.twist.linear.x = float(flight.get("vx") or 0.0)
        velocity.twist.linear.y = float(flight.get("vy") or 0.0)
        velocity.twist.linear.z = float(flight.get("vz") or 0.0)
        self.velocity_pub.publish(velocity)

        battery = BatteryState()
        battery.header.stamp = stamp
        battery.header.frame_id = "aircraft"
        battery.voltage = float(flight.get("voltage_v") or float("nan"))
        battery.current = float(flight.get("current_a") or float("nan"))
        remaining = flight.get("battery_remaining_pct")
        battery.percentage = float(remaining) / 100.0 if remaining is not None else float("nan")
        self.battery_pub.publish(battery)

        self.flight_pub.publish(self._json_message(flight))
        self.detection_pub.publish(self._json_message({
            "timestamp": utc_now(),
            "frame_index": frame_index,
            "detections": detections,
        }))

    def publish_frame(
        self,
        sample: CameraSample,
        annotated: np.ndarray,
        detections: List[Dict[str, Any]],
        flight: Dict[str, Any],
        frame_index: int,
    ) -> None:
        from sensor_msgs.msg import CameraInfo, Imu

        now = time.monotonic()
        if now - self._last_image_publish >= self.image_interval_s:
            self._last_image_publish = now
            compressed = self._compressed_message(
                annotated, "d455_color", self._stamp()
            )
            if compressed is not None:
                self.compressed_annotated_pub.publish(compressed)
        stamp = self._stamp()
        if self.compressed_only:
            self._publish_flight_and_detections(flight, detections, frame_index, stamp)
            return

        self.color_pub.publish(self._image_message(sample.color, "bgr8", "d455_color", stamp))
        self.annotated_pub.publish(
            self._image_message(annotated, "bgr8", "d455_color", stamp)
        )
        if sample.depth_m is not None:
            self.depth_pub.publish(
                self._image_message(sample.depth_m.astype(np.float32), "32FC1", "d455_color", stamp)
            )

        intr = sample.intrinsics
        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = "d455_color"
        camera_info.width = intr.width
        camera_info.height = intr.height
        camera_info.distortion_model = "plumb_bob"
        camera_info.k = [intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0]
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [intr.fx, 0.0, intr.ppx, 0.0, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_pub.publish(camera_info)

        if sample.imu:
            d455_imu = Imu()
            d455_imu.header.stamp = stamp
            d455_imu.header.frame_id = "d455_imu"
            d455_imu.orientation_covariance[0] = -1.0
            av = sample.imu.get("angular_velocity", (0.0, 0.0, 0.0))
            la = sample.imu.get("linear_acceleration", (0.0, 0.0, 0.0))
            d455_imu.angular_velocity.x, d455_imu.angular_velocity.y, d455_imu.angular_velocity.z = map(float, av)
            d455_imu.linear_acceleration.x, d455_imu.linear_acceleration.y, d455_imu.linear_acceleration.z = map(float, la)
            self.d455_imu_pub.publish(d455_imu)

        self._publish_flight_and_detections(flight, detections, frame_index, stamp)

    def publish_event(self, event: Dict[str, Any]) -> None:
        target = event.get("target_location") or {}
        if target.get("valid"):
            stamp = self._stamp()
            self.target_fix_pub.publish(self._navsat_message(target, stamp, "target_wgs84"))
        self.target_detail_pub.publish(self._json_message(event))

    def publish_transmission(self, view: np.ndarray, record: Dict[str, Any]) -> None:
        stamp = self._stamp()
        compressed = self._compressed_message(view, "progressive_transmission", stamp)
        if compressed is not None:
            self.compressed_tx_pub.publish(compressed)
        if self.compressed_only:
            return
        self.tx_image_pub.publish(
            self._image_message(view, "bgr8", "progressive_transmission", stamp)
        )
        self.tx_progress_pub.publish(self._json_message(record))

    def stop(self) -> None:
        # Let reliable one-shot event and progressive-image samples reach rosbag2.
        time.sleep(0.5)
        self._running.clear()
        self._thread.join(timeout=2)
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        log.info("ROS 2 publisher stopped")


class FlightTelemetry:
    """Continuously reads the useful navigation, attitude, and flight messages."""

    MESSAGE_RATES = {
        "GLOBAL_POSITION_INT": 10.0,
        "GPS_RAW_INT": 2.0,
        "ATTITUDE": 20.0,
        "VFR_HUD": 5.0,
        "SYS_STATUS": 2.0,
        "BATTERY_STATUS": 2.0,
        "HIGHRES_IMU": 10.0,
        "ALTITUDE": 5.0,
        "DISTANCE_SENSOR": 10.0,
        "WIND_COV": 2.0,
    }

    def __init__(self, port: str, baud: int, mock: bool = False):
        self.port = port
        self.baud = baud
        self.mock = mock
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._conn = None
        self._data = self._empty_state()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "timestamp": utc_now(),
            "connected": False,
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "relative_altitude": None,
            "terrain_altitude": None,
            "bottom_clearance": None,
            "rangefinder_m": None,
            "heading": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "rollspeed": 0.0,
            "pitchspeed": 0.0,
            "yawspeed": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "groundspeed": 0.0,
            "airspeed": 0.0,
            "climb": 0.0,
            "throttle": 0.0,
            "fix_type": 0,
            "satellites": 0,
            "hdop": None,
            "vdop": None,
            "gps_eph_m": None,
            "gps_epv_m": None,
            "voltage_v": None,
            "current_a": None,
            "battery_remaining_pct": None,
            "flight_mode": None,
            "armed": None,
            "last_message": None,
            "last_command_ack_command": None,
            "last_command_ack_result": None,
            "last_command_ack_progress": None,
            "last_command_ack_result_param2": None,
        }

    def start(self) -> None:
        self._running.set()
        target = self._mock_loop if self.mock else self._reader_loop
        threading.Thread(target=target, name="mavlink", daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def _mock_loop(self) -> None:
        start = time.monotonic()
        while self._running.is_set():
            t = time.monotonic() - start
            with self._lock:
                self._data.update({
                    "timestamp": utc_now(),
                    "connected": True,
                    "latitude": 25.033964 + 0.00001 * math.sin(t / 20),
                    "longitude": 121.564468 + 0.00001 * math.cos(t / 20),
                    "altitude": 82.3,
                    "relative_altitude": 30.0,
                    "rangefinder_m": 30.0,
                    "heading": 25.0,
                    "yaw": math.radians(25.0),
                    "fix_type": 6,
                    "satellites": 18,
                    "hdop": 0.7,
                    "gps_eph_m": 0.15,
                    "gps_epv_m": 0.25,
                    "groundspeed": 2.0,
                    "armed": True,
                    "flight_mode": "MOCK",
                    "last_message": "MOCK",
                })
            time.sleep(0.05)

    def _request_rates(self, conn, mavutil) -> None:
        for name, hz in self.MESSAGE_RATES.items():
            msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None:
                continue
            try:
                conn.mav.command_long_send(
                    conn.target_system,
                    conn.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,
                    int(1_000_000 / hz),
                    0, 0, 0, 0, 0,
                )
            except Exception:
                pass

    def _reader_loop(self) -> None:
        from pymavlink import mavutil

        while self._running.is_set():
            try:
                log.info("MAVLink connecting to %s @ %d", self.port, self.baud)
                conn = mavutil.mavlink_connection(self.port, baud=self.baud, autoreconnect=False)
                heartbeat = conn.wait_heartbeat(timeout=8)
                if heartbeat is None:
                    raise TimeoutError("heartbeat timeout")
                self._conn = conn
                self._request_rates(conn, mavutil)
                with self._lock:
                    self._data["connected"] = True
                log.info("MAVLink heartbeat received (system %d)", conn.target_system)

                while self._running.is_set():
                    msg = conn.recv_match(blocking=True, timeout=1.0)
                    if msg is not None and msg.get_type() != "BAD_DATA":
                        self._consume(msg, mavutil)
            except Exception as exc:
                with self._lock:
                    self._data["connected"] = False
                log.warning("MAVLink disconnected: %s; retrying", exc)
                time.sleep(3)

    def _consume(self, msg, mavutil) -> None:
        mt = msg.get_type()
        with self._lock:
            d = self._data
            d["timestamp"] = utc_now()
            d["last_message"] = mt
            if mt == "GLOBAL_POSITION_INT":
                if abs(msg.lat) > 1 and abs(msg.lon) > 1:
                    d["latitude"] = msg.lat / 1e7
                    d["longitude"] = msg.lon / 1e7
                    d["altitude"] = msg.alt / 1000.0
                d["relative_altitude"] = msg.relative_alt / 1000.0
                d["vx"], d["vy"], d["vz"] = msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0
                if msg.hdg != 65535:
                    d["heading"] = msg.hdg / 100.0
            elif mt == "GPS_RAW_INT":
                d["fix_type"] = int(msg.fix_type)
                d["satellites"] = int(getattr(msg, "satellites_visible", 0))
                eph, epv = getattr(msg, "eph", 65535), getattr(msg, "epv", 65535)
                d["gps_eph_m"] = None if eph == 65535 else eph / 100.0
                d["gps_epv_m"] = None if epv == 65535 else epv / 100.0
                d["hdop"] = None if eph == 65535 else eph / 100.0
                d["vdop"] = None if epv == 65535 else epv / 100.0
            elif mt == "ATTITUDE":
                for key in ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"):
                    d[key] = float(getattr(msg, key))
                d["heading"] = math.degrees(d["yaw"]) % 360.0
            elif mt == "VFR_HUD":
                for key in ("groundspeed", "airspeed", "climb", "throttle"):
                    d[key] = float(getattr(msg, key, 0.0))
            elif mt == "ALTITUDE":
                terrain = float(getattr(msg, "altitude_terrain", float("nan")))
                clearance = float(getattr(msg, "bottom_clearance", float("nan")))
                d["terrain_altitude"] = terrain if math.isfinite(terrain) else None
                d["bottom_clearance"] = clearance if math.isfinite(clearance) and clearance > 0 else None
            elif mt == "DISTANCE_SENSOR":
                distance_cm = float(getattr(msg, "current_distance", 0.0))
                min_cm = float(getattr(msg, "min_distance", 0.0))
                max_cm = float(getattr(msg, "max_distance", float("inf")))
                orientation = int(getattr(msg, "orientation", -1))
                if min_cm <= distance_cm <= max_cm and distance_cm > 0 and orientation in (-1, 25):
                    d["rangefinder_m"] = distance_cm / 100.0
            elif mt == "SYS_STATUS":
                voltage = float(getattr(msg, "voltage_battery", -1))
                current = float(getattr(msg, "current_battery", -1))
                remaining = int(getattr(msg, "battery_remaining", -1))
                d["voltage_v"] = None if voltage < 0 else voltage / 1000.0
                d["current_a"] = None if current < 0 else current / 100.0
                d["battery_remaining_pct"] = None if remaining < 0 else remaining
            elif mt == "HEARTBEAT":
                d["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    d["flight_mode"] = mavutil.mode_string_v10(msg)
                except Exception:
                    pass
            elif mt == "COMMAND_ACK":
                d["last_command_ack_command"] = int(getattr(msg, "command", -1))
                d["last_command_ack_result"] = int(getattr(msg, "result", -1))
                d["last_command_ack_progress"] = int(getattr(msg, "progress", 0))
                d["last_command_ack_result_param2"] = int(getattr(msg, "result_param2", 0))


class D455Camera:
    """Aligned D455 RGB/depth capture with latest-frame semantics."""

    def __init__(
        self,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        fake: bool,
        enable_depth: bool = True,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.fake = fake
        self.enable_depth = enable_depth
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._sample: Optional[CameraSample] = None
        self._sample_pending = False
        self._pipeline = None

    def start(self) -> None:
        self._running.set()
        target = self._fake_loop if self.fake else self._realsense_loop
        self._thread = threading.Thread(target=target, name="d455", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                log.warning("D455 capture thread did not stop within 8 seconds")

    def get(self) -> Optional[CameraSample]:
        with self._lock:
            if self._sample is None:
                return None
            s = self._sample
            result = CameraSample(
                color=s.color.copy(),
                depth_m=None if s.depth_m is None else s.depth_m.copy(),
                intrinsics=s.intrinsics,
                camera_timestamp_ms=s.camera_timestamp_ms,
                imu=copy.deepcopy(s.imu),
            )
            self._sample_pending = False
            return result

    def _fake_loop(self) -> None:
        intr = Intrinsics(self.width, self.height, 600.0, 600.0, self.width / 2, self.height / 2)
        start = time.monotonic()
        while self._running.is_set():
            t = time.monotonic() - start
            img = np.full((self.height, self.width, 3), (45, 55, 50), np.uint8)
            cx = int(self.width * 0.5 + math.sin(t * 0.35) * self.width * 0.2)
            cy = int(self.height * 0.48)
            x1, y1, x2, y2 = cx - 65, cy - 32, cx + 65, cy + 32
            cv2.rectangle(img, (x1, y1), (x2, y2), (35, 80, 210), -1)
            cv2.circle(img, (x1 + 25, y2), 10, (10, 10, 10), -1)
            cv2.circle(img, (x2 - 25, y2), 10, (10, 10, 10), -1)
            depth = None
            if self.enable_depth:
                depth = np.full((self.height, self.width), 30.0, np.float32)
                depth[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)] = 28.0
            with self._lock:
                if not self._sample_pending:
                    self._sample = CameraSample(img, depth, intr, t * 1000.0, {"source": "fake"})
                    self._sample_pending = True
            time.sleep(1.0 / self.fps)

    @staticmethod
    def _motion_tuple(frame) -> Tuple[float, float, float]:
        v = frame.as_motion_frame().get_motion_data()
        return float(v.x), float(v.y), float(v.z)

    def _realsense_loop(self) -> None:
        import pyrealsense2 as rs

        while self._running.is_set():
            pipe = rs.pipeline()
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            imu_enabled = True
            try:
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                profile = pipe.start(cfg)
            except Exception as exc:
                log.warning("D455 IMU unavailable (%s); opening camera without IMU", exc)
                try:
                    pipe.stop()
                except Exception:
                    pass
                pipe, cfg = rs.pipeline(), rs.config()
                if self.serial:
                    cfg.enable_device(self.serial)
                cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                if self.enable_depth:
                    cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                profile = pipe.start(cfg)
                imu_enabled = False

            self._pipeline = pipe
            try:
                color_sensor = profile.get_device().first_color_sensor()
                ae = os.environ.get("D455_COLOR_AUTO_EXPOSURE")
                exposure = os.environ.get("D455_COLOR_EXPOSURE")
                gain = os.environ.get("D455_COLOR_GAIN")
                awb = os.environ.get("D455_COLOR_AUTO_WHITE_BALANCE")
                wb = os.environ.get("D455_COLOR_WHITE_BALANCE")
                brightness = os.environ.get('D455_COLOR_BRIGHTNESS')
                contrast = os.environ.get('D455_COLOR_CONTRAST')
                gamma = os.environ.get('D455_COLOR_GAMMA')
                saturation = os.environ.get('D455_COLOR_SATURATION')
                sharpness = os.environ.get('D455_COLOR_SHARPNESS')
                backlight = os.environ.get('D455_COLOR_BACKLIGHT_COMPENSATION')
                if ae is not None:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if ae != "0" else 0.0)
                if exposure:
                    color_sensor.set_option(rs.option.exposure, float(exposure))
                if gain:
                    color_sensor.set_option(rs.option.gain, float(gain))
                if awb is not None:
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if awb != "0" else 0.0)
                if wb:
                    color_sensor.set_option(rs.option.white_balance, float(wb))
                if brightness:
                    color_sensor.set_option(rs.option.brightness, float(brightness))
                if contrast:
                    color_sensor.set_option(rs.option.contrast, float(contrast))
                if gamma:
                    color_sensor.set_option(rs.option.gamma, float(gamma))
                if saturation:
                    color_sensor.set_option(rs.option.saturation, float(saturation))
                if sharpness:
                    color_sensor.set_option(rs.option.sharpness, float(sharpness))
                if backlight:
                    color_sensor.set_option(rs.option.backlight_compensation, float(backlight))
                if any(v is not None for v in (ae, exposure, gain, awb, wb)):
                    log.info(
                        "D455 color controls: auto_exposure=%s exposure=%s gain=%s auto_wb=%s wb=%s",
                        ae, exposure, gain, awb, wb,
                    )
            except Exception as exc:
                log.warning("D455 color control setup failed: %s", exc)
            align = rs.align(rs.stream.color) if self.enable_depth else None
            depth_scale = (
                profile.get_device().first_depth_sensor().get_depth_scale()
                if self.enable_depth else None
            )
            log.info(
                "D455 started serial=%s mode=%s%s",
                self.serial or "auto",
                "RGB+depth" if self.enable_depth else "RGB-only",
                f" depth_scale={depth_scale:.6f}" if depth_scale is not None else "",
            )
            imu: Dict[str, Any] = {}
            try:
                while self._running.is_set():
                    frames = pipe.wait_for_frames(timeout_ms=5000)
                    with self._lock:
                        if self._sample_pending:
                            continue
                    processed = align.process(frames) if align is not None else frames
                    color_frame = processed.get_color_frame()
                    depth_frame = processed.get_depth_frame() if self.enable_depth else None
                    if not color_frame or (self.enable_depth and not depth_frame):
                        continue
                    color = np.asanyarray(color_frame.get_data())
                    depth_m = (
                        np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                        if depth_frame is not None and depth_scale is not None else None
                    )
                    i = color_frame.profile.as_video_stream_profile().intrinsics
                    intr = Intrinsics(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy)
                    if imu_enabled:
                        try:
                            accel = frames.first_or_default(rs.stream.accel)
                            gyro = frames.first_or_default(rs.stream.gyro)
                            if accel:
                                imu["linear_acceleration"] = self._motion_tuple(accel)
                            if gyro:
                                imu["angular_velocity"] = self._motion_tuple(gyro)
                        except Exception:
                            pass
                    with self._lock:
                        self._sample = CameraSample(
                            color, depth_m, intr, color_frame.get_timestamp(), copy.deepcopy(imu)
                        )
                        self._sample_pending = True
            except Exception as exc:
                if self._running.is_set():
                    log.warning("D455 capture failed: %s; retrying", exc)
                    time.sleep(2)
            finally:
                try:
                    pipe.stop()
                except Exception:
                    pass


class TargetGeolocator:
    """D455 pixel/depth plus calibrated transforms to WGS84 target position."""

    def __init__(
        self,
        mount_rpy_deg: Sequence[float],
        camera_offset_frd_m: Sequence[float],
        min_depth_m: float,
        max_depth_m: float,
        attitude_sigma_deg: float,
    ):
        roll, pitch, yaw = [math.radians(float(v)) for v in mount_rpy_deg]
        self.mount_rotation = euler_matrix(roll, pitch, yaw)
        self.camera_offset = np.asarray(camera_offset_frd_m, dtype=np.float64)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.attitude_sigma_rad = math.radians(attitude_sigma_deg)
        self.mount_rpy_deg = list(mount_rpy_deg)

        # RealSense optical axes (right, down, forward) to aircraft FRD
        # (forward, right, down), before applying the camera mount rotation.
        self.optical_to_frd = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )

    def _robust_depth(self, depth: np.ndarray, bbox: Sequence[float]) -> Optional[Tuple[float, float, int]]:
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Use the central 50% to reject background around a loose detector box.
        xa = max(0, int(x1 + 0.25 * (x2 - x1)))
        xb = min(w, int(x2 - 0.25 * (x2 - x1)))
        ya = max(0, int(y1 + 0.20 * (y2 - y1)))
        yb = min(h, int(y2 - 0.20 * (y2 - y1)))
        if xb <= xa or yb <= ya:
            return None
        values = depth[ya:yb, xa:xb].reshape(-1)
        values = values[np.isfinite(values)]
        values = values[(values >= self.min_depth_m) & (values <= self.max_depth_m)]
        if values.size < 12:
            return None

        # Select the dominant 0.5 m depth cluster, then use median/MAD.
        bins = np.floor(values / 0.5).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        dominant = unique[int(np.argmax(counts))]
        cluster = values[np.abs(bins - dominant) <= 1]
        if cluster.size < 12:
            cluster = values
        distance = float(np.median(cluster))
        mad = float(1.4826 * np.median(np.abs(cluster - distance)))
        return distance, max(mad, 0.02), int(cluster.size)

    def locate(
        self,
        depth_m: Optional[np.ndarray],
        intr: Intrinsics,
        bbox: Sequence[float],
        flight: Dict[str, Any],
    ) -> Dict[str, Any]:
        base = {
            "valid": False,
            "method": "d455_aligned_depth_attitude_wgs84",
            "mount_rpy_deg": self.mount_rpy_deg,
            "camera_offset_frd_m": self.camera_offset.tolist(),
        }
        if flight.get("latitude") is None or flight.get("longitude") is None:
            return {**base, "reason": "no_aircraft_gps"}
        x1, y1, x2, y2 = [float(v) for v in bbox]
        u = max(0.0, min(intr.width - 1.0, (x1 + x2) * 0.5))
        v = max(0.0, min(intr.height - 1.0, (y1 + y2) * 0.5))
        body_to_ned = euler_matrix(
            float(flight.get("roll") or 0.0),
            float(flight.get("pitch") or 0.0),
            float(flight.get("yaw") or math.radians(float(flight.get("heading") or 0.0))),
        )
        depth_result = None if depth_m is None else self._robust_depth(depth_m, bbox)
        if depth_result is None:
            return self._ray_ground_fallback(base, u, v, intr, body_to_ned, flight)

        depth, depth_sigma, sample_count = depth_result
        optical = np.array(
            [(u - intr.ppx) * depth / intr.fx, (v - intr.ppy) * depth / intr.fy, depth],
            dtype=np.float64,
        )
        body = self.camera_offset + self.mount_rotation @ (self.optical_to_frd @ optical)
        ned = body_to_ned @ body
        north, east, down = [float(vv) for vv in ned]
        aircraft_lat = float(flight["latitude"])
        aircraft_lon = float(flight["longitude"])
        lat = aircraft_lat + math.degrees(north / EARTH_RADIUS_M)
        lon = aircraft_lon + math.degrees(east / (EARTH_RADIUS_M * max(math.cos(math.radians(aircraft_lat)), 1e-6)))
        alt = None if flight.get("altitude") is None else float(flight["altitude"]) - down
        gps_sigma = float(flight.get("gps_eph_m") or 0.0)
        attitude_sigma = depth * self.attitude_sigma_rad
        horizontal_sigma = math.sqrt(gps_sigma ** 2 + depth_sigma ** 2 + attitude_sigma ** 2)
        return {
            **base,
            "valid": True,
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "north_m": north,
            "east_m": east,
            "down_m": down,
            "slant_range_m": depth,
            "depth_sigma_m": depth_sigma,
            "estimated_horizontal_sigma_m": horizontal_sigma,
            "position_quality": "depth_measured",
            "depth_sample_count": sample_count,
            "pixel": [u, v],
            "camera_point_optical_m": optical.tolist(),
            "body_point_frd_m": body.tolist(),
            "ned_offset_m": ned.tolist(),
            "aircraft_fix_type": flight.get("fix_type"),
            "calibration_required_for_survey_grade": True,
        }

    def _ray_ground_fallback(
        self,
        base: Dict[str, Any],
        u: float,
        v: float,
        intr: Intrinsics,
        body_to_ned: np.ndarray,
        flight: Dict[str, Any],
    ) -> Dict[str, Any]:
        # At competition altitude (~60 m), RealSense/rangefinder readings can be
        # invalid or clipped. Prefer the flight controller relative altitude for
        # ground-plane ray projection; only fall back to terrain/rangefinder if REL
        # is unavailable.
        height = (
            flight.get("relative_altitude")
            or flight.get("bottom_clearance")
            or flight.get("rangefinder_m")
        )
        if height is None or float(height) <= 0:
            return {**base, "reason": "invalid_depth_and_no_height_agl"}
        height = float(height)
        ray_optical = np.array(
            [(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, 1.0], dtype=np.float64
        )
        ray_body = self.mount_rotation @ (self.optical_to_frd @ ray_optical)
        ray_ned = body_to_ned @ ray_body
        camera_origin_ned = body_to_ned @ self.camera_offset
        if ray_ned[2] <= 1e-4:
            return {**base, "reason": "camera_ray_does_not_intersect_ground"}
        scale = (height - float(camera_origin_ned[2])) / float(ray_ned[2])
        if scale <= 0:
            return {**base, "reason": "ground_intersection_behind_camera"}
        ned = camera_origin_ned + scale * ray_ned
        north, east, down = [float(value) for value in ned]
        aircraft_lat = float(flight["latitude"])
        aircraft_lon = float(flight["longitude"])
        lat = aircraft_lat + math.degrees(north / EARTH_RADIUS_M)
        lon = aircraft_lon + math.degrees(
            east / (EARTH_RADIUS_M * max(math.cos(math.radians(aircraft_lat)), 1e-6))
        )
        alt = None if flight.get("altitude") is None else float(flight["altitude"]) - down
        gps_sigma = float(flight.get("gps_eph_m") or 0.0)
        attitude_sigma = float(np.linalg.norm(ned[:2])) * self.attitude_sigma_rad
        height_sigma = max(0.5, height * 0.05)
        horizontal_sigma = math.sqrt(gps_sigma ** 2 + attitude_sigma ** 2 + height_sigma ** 2)
        return {
            **base,
            "valid": True,
            "method": "attitude_compensated_ground_ray_wgs84",
            "position_quality": "estimated_no_valid_depth",
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "north_m": north,
            "east_m": east,
            "down_m": down,
            "slant_range_m": float(np.linalg.norm(ned - camera_origin_ned)),
            "estimated_horizontal_sigma_m": horizontal_sigma,
            "pixel": [u, v],
            "ned_offset_m": ned.tolist(),
            "height_agl_m": height,
            "height_source": (
                "relative_altitude"
                if flight.get("relative_altitude")
                else "bottom_clearance"
                if flight.get("bottom_clearance")
                else "rangefinder"
            ),
            "aircraft_fix_type": flight.get("fix_type"),
            "calibration_required_for_survey_grade": True,
        }


class FFmpegRecorder:
    """Non-blocking H.264/MP4 writer with hardware encoder preference."""

    def __init__(self, path: Path, width: int, height: int, fps: float, encoder: str):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = self._select_encoder(encoder)
        self._q: queue.Queue = queue.Queue(maxsize=max(30, int(fps * 3)))
        self._running = threading.Event()
        self._proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _encoder_works(name: str) -> bool:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=size=64x64:rate=1", "-frames:v", "1", "-c:v", name,
            "-f", "null", "-",
        ]
        try:
            return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8).returncode == 0
        except Exception:
            return False

    def _select_encoder(self, requested: str) -> str:
        if requested != "auto":
            return requested
        for candidate in ("h264_nvenc", "h264_v4l2m2m", "libx264"):
            if self._encoder_works(candidate):
                log.info("video encoder selected: %s", candidate)
                return candidate
        return "mpeg4"

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.3f}", "-i", "-", "-an", "-c:v", self.encoder,
        ]
        if self.encoder == "libx264":
            cmd += ["-preset", "veryfast", "-crf", "24"]
        elif self.encoder in ("h264_nvenc", "h264_v4l2m2m"):
            cmd += ["-b:v", "4M"]
        else:
            cmd += ["-q:v", "5"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._running.set()
        threading.Thread(target=self._loop, name=f"record-{self.path.stem}", daemon=True).start()

    def write(self, frame: np.ndarray) -> None:
        if not self._running.is_set():
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.task_done()
                self._q.put_nowait(frame.copy())
            except queue.Empty:
                pass

    def _loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        while self._running.is_set() or not self._q.empty():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                self._running.clear()
            finally:
                self._q.task_done()

    def stop(self) -> None:
        self._running.clear()
        deadline = time.monotonic() + 8
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.terminate()


class SharedOutput:
    def __init__(self):
        self._lock = threading.Lock()
        self.jpeg = b""
        self.revision = 0
        self.latest_event: Dict[str, Any] = {}
        self.latest_transmission: Dict[str, Any] = {}
        self.transmitting = False

    def publish_jpeg(self, data: bytes) -> int:
        with self._lock:
            self.jpeg = data
            self.revision += 1
            return self.revision

    def get_jpeg(self) -> Tuple[bytes, int]:
        with self._lock:
            return self.jpeg, self.revision

    def set_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_event = copy.deepcopy(event)

    def get_event(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.latest_event)

    def set_transmission(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_transmission = copy.deepcopy(record)

    def get_transmission(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.latest_transmission)


@dataclass
class TransmissionJob:
    event_id: str
    track_id: str
    frame: np.ndarray
    bbox: List[float]
    event: Dict[str, Any]


class ProgressiveTransmitter:
    """Transmit only new-car stills, expanding from target ROI to panorama."""

    def __init__(
        self,
        shared: SharedOutput,
        recorder: FFmpegRecorder,
        tx_log: JsonlWriter,
        output_size: Tuple[int, int],
        steps: int,
        interval_s: float,
        jpeg_quality: int,
        bps: int,
        ros2: Optional[ROS2Publisher] = None,
    ):
        self.shared = shared
        self.recorder = recorder
        self.tx_log = tx_log
        self.width, self.height = output_size
        self.steps = max(2, steps)
        self.interval_s = interval_s
        self.jpeg_quality = jpeg_quality
        self.bps = bps
        self.ros2 = ros2
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="progressive-tx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=max(10.0, self.steps * self.interval_s + 2.0))

    def submit(self, job: TransmissionJob) -> bool:
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            log.warning("transmission queue full; dropping %s", job.track_id)
            return False

    @staticmethod
    def _crop_for_progress(frame: np.ndarray, bbox: Sequence[float], progress: float) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = max(x2 - x1, 8.0), max(y2 - y1, 8.0)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        start_w, start_h = min(w, bw * 1.35), min(h, bh * 1.35)
        # Smoothstep makes the beginning readable and the final expansion gentle.
        eased = progress * progress * (3.0 - 2.0 * progress)
        crop_w = start_w + (w - start_w) * eased
        crop_h = start_h + (h - start_h) * eased
        left = cx - crop_w * 0.5
        top = cy - crop_h * 0.5
        left = min(max(left, 0.0), w - crop_w)
        top = min(max(top, 0.0), h - crop_h)
        xa, ya = int(round(left)), int(round(top))
        xb, yb = int(round(left + crop_w)), int(round(top + crop_h))
        return frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]

    def _render(self, job: TransmissionJob, stage: int) -> np.ndarray:
        progress = stage / (self.steps - 1)
        crop = self._crop_for_progress(job.frame, job.bbox, progress)
        view = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
        label = f"{job.track_id}  ZOOM OUT {stage + 1}/{self.steps}  {progress * 100:.0f}%"
        cv2.rectangle(view, (0, self.height - 34), (self.width, self.height), (0, 0, 0), -1)
        cv2.putText(view, label, (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (60, 240, 255), 2, cv2.LINE_AA)
        return view

    @staticmethod
    def _important_metadata(job: TransmissionJob) -> Dict[str, Any]:
        detection = (job.event.get("detections") or [{}])[0]
        target = job.event.get("target_location") or {}
        flight = job.event.get("gps") or {}

        def present(source: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
            return {
                key: source[key]
                for key in keys
                if source.get(key) is not None and source.get(key) != ""
            }

        return {
            "detection": present(
                detection, ("track_id", "class", "confidence", "bbox", "source")
            ),
            "target": present(
                target,
                (
                    "valid", "latitude", "longitude", "altitude",
                    "estimated_horizontal_sigma_m", "slant_range_m",
                    "position_quality", "reason",
                ),
            ),
            "flight": present(
                flight,
                (
                    "latitude", "longitude", "altitude", "relative_altitude",
                    "rangefinder_m", "bottom_clearance", "fix_type", "satellites",
                    "hdop", "heading", "groundspeed", "battery_remaining_pct",
                    "voltage_v", "flight_mode", "armed",
                ),
            ),
        }

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                job: TransmissionJob = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.shared.transmitting = True
            started = time.time()
            total_bytes = 0
            stages_sent = 0
            try:
                for stage in range(self.steps):
                    if not self._running.is_set():
                        break
                    view = self._render(job, stage)
                    ok, encoded = cv2.imencode(
                        ".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                    if not ok:
                        continue
                    payload = encoded.tobytes()
                    total_bytes += len(payload)
                    stages_sent += 1
                    image_revision = self.shared.publish_jpeg(payload)
                    self.recorder.write(view)
                    record = {
                        "type": "progressive_frame",
                        "timestamp": utc_now(),
                        "event_id": job.event_id,
                        "track_id": job.track_id,
                        "stage": stage,
                        "stages": self.steps,
                        "progress": stage / (self.steps - 1),
                        "jpeg_bytes": len(payload),
                        "wire_bytes": len(payload),
                        "cumulative_bytes": total_bytes,
                        "image_revision": image_revision,
                        "image_url": "/target-image",
                        "target": job.event.get("target_location"),
                        "metadata": self._important_metadata(job),
                    }
                    self.tx_log.write(record)
                    self.shared.set_transmission(record)
                    if self.ros2 is not None:
                        self.ros2.publish_transmission(view, record)
                    # Send only metadata here. The GUI fetches this versioned
                    # JPEG over HTTP, which is retryable and avoids base64 cost.
                    socketio.emit("transmission_progress", record, namespace="/events")
                    delay = self.interval_s
                    if self.bps > 0:
                        delay = max(delay, len(payload) / self.bps)
                    deadline = time.monotonic() + delay
                    while self._running.is_set() and time.monotonic() < deadline:
                        time.sleep(min(0.1, deadline - time.monotonic()))
                completed = stages_sent == self.steps
                final_record = {
                    "type": "progressive_complete" if completed else "progressive_aborted",
                    "timestamp": utc_now(),
                    "event_id": job.event_id,
                    "track_id": job.track_id,
                    "elapsed_s": time.time() - started,
                    "total_bytes": total_bytes,
                    "stages_sent": stages_sent,
                    "stages": self.steps,
                }
                self.tx_log.write(final_record)
                self.shared.set_transmission(final_record)
                socketio.emit("transmission_complete", final_record, namespace="/events")
            except Exception as exc:
                log.exception("progressive transmission failed for %s", job.track_id)
                error_record = {
                    "type": "progressive_error",
                    "timestamp": utc_now(),
                    "event_id": job.event_id,
                    "track_id": job.track_id,
                    "elapsed_s": time.time() - started,
                    "total_bytes": total_bytes,
                    "stages_sent": stages_sent,
                    "stages": self.steps,
                    "error": str(exc),
                }
                self.tx_log.write(error_record)
                self.shared.set_transmission(error_record)
                socketio.emit("transmission_complete", error_record, namespace="/events")
            finally:
                self.shared.transmitting = False
                self._q.task_done()


class PrecisionTracker:
    def __init__(
        self,
        args,
        session_dir: Path,
        flight: FlightTelemetry,
        camera: D455Camera,
        ros2: Optional[ROS2Publisher] = None,
    ):
        self.args = args
        self.session_dir = session_dir
        self.flight = flight
        self.camera = camera
        self.ros2 = ros2
        self.shared = SHARED
        self._running = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Optional[Exception] = None
        self._model = None
        self._botsort = None
        self._car_class = 3
        self._inference_device = "cpu"
        self._use_half = False
        self._hit_count: Dict[int, int] = {}
        self._sent_tracks: set[int] = set()
        self._gate_notified_tracks: set[int] = set()
        self._track_features: Dict[int, np.ndarray] = {}
        self._sent_target_signatures: List[Dict[str, Any]] = []
        self._frame_index = 0

        self.frame_log = JsonlWriter(session_dir / "flight_and_detections.jsonl")
        self.tx_log = JsonlWriter(session_dir / "transmission.jsonl")
        self.geolocator = TargetGeolocator(
            (args.camera_roll_deg, args.camera_pitch_deg, args.camera_yaw_deg),
            (args.camera_x_m, args.camera_y_m, args.camera_z_m),
            args.min_depth,
            args.max_depth,
            args.attitude_sigma_deg,
        )
        self.main_recorder = FFmpegRecorder(
            session_dir / "annotated_flight.mp4", args.width, args.height, args.fps, args.video_encoder
        )
        self.tx_recorder = FFmpegRecorder(
            session_dir / "progressive_transmission.mp4", args.tx_width, args.tx_height,
            1.0 / max(args.tx_interval, 0.04), args.video_encoder
        )
        self.transmitter = ProgressiveTransmitter(
            self.shared, self.tx_recorder, self.tx_log,
            (args.tx_width, args.tx_height), args.tx_steps, args.tx_interval,
            args.tx_jpeg_quality, args.tx_bps,
            ros2,
        )

    def _write_tracker_yaml(self) -> Path:
        path = self.session_dir / "botsort_reid.yaml"
        path.write_text(
            "\n".join([
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.10",
                "new_track_thresh: 0.25",
                "track_buffer: 90",
                "match_thresh: 0.80",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                "proximity_thresh: 0.50",
                "appearance_thresh: 0.80",
                f"with_reid: {'True' if self.args.with_reid else 'False'}",
                f"model: {self.args.reid_model}",
                "",
            ]), encoding="ascii"
        )
        return path

    def start(self) -> None:
        # Spawn encoders before CUDA initialization. Fork/exec after a CUDA
        # context exists can invalidate lazy caches on Jetson unified memory.
        self.main_recorder.start()
        self.tx_recorder.start()
        self.transmitter.start()
        if not self.args.fake_camera:
            import torch
            from ultralytics import YOLO

            if not Path(self.args.model).is_file():
                raise FileNotFoundError(f"model not found: {self.args.model}")
            cuda_ready = torch.cuda.is_available()
            if self.args.device == "auto":
                self._inference_device = "0" if cuda_ready else "cpu"
            else:
                self._inference_device = self.args.device
            if self._inference_device != "cpu" and not cuda_ready:
                raise RuntimeError(
                    f"CUDA device {self._inference_device} was requested, but this PyTorch build has no CUDA"
                )
            if self.args.require_gpu and not cuda_ready:
                raise RuntimeError("GPU preflight failed: CUDA-enabled PyTorch is required")
            self._use_half = bool(self.args.half and self._inference_device != "cpu")
            if not cuda_ready:
                log.warning(
                    "CUDA is unavailable in the installed PyTorch; using CPU/FP32. "
                    "Install the Jetson CUDA PyTorch build for flight performance."
                )
            log.info("loading YOLO model %s", self.args.model)
            self._model = YOLO(self.args.model)
            names = self._model.names
            matches = [int(i) for i, name in names.items() if str(name).lower() == CAR_CLASS_NAME]
            if not matches:
                raise RuntimeError(f"class 'car' not present in model names: {names}")
            self._car_class = matches[0]
            log.info(
                "car class=%d; tracker=BoT-SORT, ReID=%s, device=%s, half=%s",
                self._car_class, self.args.with_reid, self._inference_device, self._use_half,
            )
        self._tracker_yaml = self._write_tracker_yaml()
        if not self.args.fake_camera:
            import torch
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.utils import IterableSimpleNamespace, YAML

            tracker_cfg = IterableSimpleNamespace(**YAML.load(self._tracker_yaml))
            self._botsort = BOTSORT(tracker_cfg, frame_rate=self.args.fps)
            warm_size = self.args.tile_imgsz if self.args.sliced_inference else self.args.imgsz
            warm_images = [np.zeros((warm_size, warm_size, 3), dtype=np.uint8)]
            if self.args.sliced_inference and self.args.full_frame_pass:
                warm_images.append(
                    np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
                )
            started = time.perf_counter()
            for warm in warm_images:
                self._model.predict(
                    warm,
                    imgsz=warm_size,
                    device=self._inference_device,
                    half=self._use_half,
                    classes=[self._car_class],
                    verbose=False,
                )
            if cuda_ready:
                torch.cuda.synchronize()
            log.info("GPU model warm-up completed in %.2fs", time.perf_counter() - started)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="detect", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError("detection worker did not become ready within 30 seconds")
        if self._startup_error is not None:
            raise RuntimeError(f"detection worker warm-up failed: {self._startup_error}")

    def stop(self) -> None:
        self._running.clear()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                raise RuntimeError("inference thread did not stop within 60 seconds")
        self.transmitter.stop()
        self.main_recorder.stop()
        self.tx_recorder.stop()
        self.frame_log.close()
        self.tx_log.close()

    def _fake_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([20, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [{"track_id": 1, "confidence": 0.95, "bbox": [x, y, x + w, y + h]}]

    @staticmethod
    def _axis_starts(length: int, tile: int, overlap: float) -> List[int]:
        if length <= tile:
            return [0]
        stride = max(1, int(round(tile * (1.0 - overlap))))
        starts = list(range(0, max(1, length - tile + 1), stride))
        final = length - tile
        if not starts or starts[-1] != final:
            starts.append(final)
        return sorted(set(starts))

    @staticmethod
    def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    def _global_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        order = sorted(range(len(detections)), key=lambda i: detections[i]["confidence"], reverse=True)
        keep: List[Dict[str, Any]] = []
        while order:
            current = order.pop(0)
            chosen = detections[current]
            keep.append(chosen)
            order = [
                idx for idx in order
                if self._box_iou(chosen["bbox"], detections[idx]["bbox"]) < self.args.tile_nms_iou
            ]
        return keep

    @staticmethod
    def _appearance_feature(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(48, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        features = []
        for channel, limit in ((0, 180), (1, 256), (2, 256)):
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, limit]).reshape(-1)
            features.append(hist)
        feature = np.concatenate(features).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 1e-6 else feature

    def _detect_sliced(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        assert self._model is not None
        h, w = frame.shape[:2]
        candidates: List[Dict[str, Any]] = []
        tiles: List[Tuple[np.ndarray, int, int, str]] = []
        if self.args.full_frame_pass:
            tiles.append((frame, 0, 0, "full"))
        tile = min(self.args.tile_size, max(h, w))
        for y in self._axis_starts(h, min(tile, h), self.args.tile_overlap):
            for x in self._axis_starts(w, min(tile, w), self.args.tile_overlap):
                tiles.append((frame[y:min(y + tile, h), x:min(x + tile, w)], x, y, "tile"))

        for image, offset_x, offset_y, source in tiles:
            tile_started = time.monotonic()
            result = self._model.predict(
                image,
                conf=self.args.conf,
                iou=self.args.iou,
                classes=[self._car_class],
                imgsz=self.args.tile_imgsz,
                device=self._inference_device,
                half=self._use_half,
                verbose=False,
            )[0]
            if self._frame_index == 1:
                log.info(
                    "tile profile source=%s offset=(%d,%d) shape=%s time=%.3fs",
                    source,
                    offset_x,
                    offset_y,
                    image.shape,
                    time.monotonic() - tile_started,
                )
            if result.boxes is None:
                continue
            for idx in range(len(result.boxes)):
                x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[idx].cpu().tolist()]
                candidates.append({
                    "confidence": float(result.boxes.conf[idx].item()),
                    "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                    "detection_source": source,
                })
        return self._global_nms(candidates)

    def _infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.args.fake_camera:
            return self._fake_detections(frame)
        assert self._model is not None and self._botsort is not None
        detect_started = time.monotonic()
        if self.args.sliced_inference:
            detected = self._detect_sliced(frame)
        else:
            result = self._model.predict(
                frame, conf=self.args.conf, iou=self.args.iou, classes=[self._car_class],
                imgsz=self.args.imgsz, device=self._inference_device, half=self._use_half,
                verbose=False,
            )[0]
            detected = [] if result.boxes is None else [
                {
                    "confidence": float(result.boxes.conf[i].item()),
                    "bbox": [float(v) for v in result.boxes.xyxy[i].cpu().tolist()],
                    "detection_source": "full",
                }
                for i in range(len(result.boxes))
            ]
        detect_elapsed = time.monotonic() - detect_started

        from ultralytics.engine.results import Boxes
        import torch

        box_array = np.asarray(
            [[*d["bbox"], d["confidence"], float(self._car_class)] for d in detected],
            dtype=np.float32,
        ).reshape(-1, 6)
        boxes = Boxes(box_array, frame.shape[:2]).cpu().numpy()
        features_np = np.asarray(
            [self._appearance_feature(frame, d["bbox"]) for d in detected], dtype=np.float32
        ).reshape(-1, 48)
        features = torch.from_numpy(features_np) if self.args.with_reid else None
        track_started = time.monotonic()
        tracks = self._botsort.update(boxes, frame, features)
        track_elapsed = time.monotonic() - track_started
        if self._frame_index == 1:
            log.info(
                "inference profile: sliced_detection=%.3fs botsort=%.3fs candidates=%d",
                detect_elapsed,
                track_elapsed,
                len(detected),
            )
        results: List[Dict[str, Any]] = []
        for row in tracks:
            source_index = int(row[-1])
            if source_index < 0 or source_index >= len(detected):
                continue
            track_id = int(row[4])
            if len(features_np):
                self._track_features[track_id] = features_np[source_index]
            results.append({
                "track_id": track_id,
                "confidence": float(row[5]),
                "bbox": [float(v) for v in row[:4]],
                "detection_source": detected[source_index]["detection_source"],
            })
        return results

    @staticmethod
    def _wgs84_distance_m(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        mean_lat = math.radians((float(a["latitude"]) + float(b["latitude"])) * 0.5)
        north = math.radians(float(a["latitude"]) - float(b["latitude"])) * EARTH_RADIUS_M
        east = (
            math.radians(float(a["longitude"]) - float(b["longitude"]))
            * EARTH_RADIUS_M * math.cos(mean_lat)
        )
        return math.hypot(north, east)

    def _is_duplicate_target(self, target: Dict[str, Any], feature: Optional[np.ndarray]) -> bool:
        for signature in self._sent_target_signatures:
            previous_target = signature["target"]
            previous = signature.get("feature")
            similarity = float(np.dot(feature, previous)) if feature is not None and previous is not None else 0.0
            if not target.get("valid"):
                if (
                    not previous_target.get("valid")
                    and similarity >= max(self.args.dedupe_appearance, 0.92)
                ):
                    return True
                continue
            if not previous_target.get("valid"):
                continue
            distance = self._wgs84_distance_m(target, previous_target)
            if distance <= self.args.dedupe_near_m:
                return True
            if distance <= self.args.dedupe_radius_m and similarity >= self.args.dedupe_appearance:
                return True
        return False

    @staticmethod
    def _annotate(frame: np.ndarray, detections: List[Dict[str, Any]], flight: Dict[str, Any]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            target = det.get("target_location", {})
            color = (30, 230, 80) if target.get("valid") else (40, 190, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"CAR T{det['track_id']} {det['confidence']:.2f}"
            if target.get("valid"):
                text += f" {target['slant_range_m']:.1f}m"
            cv2.putText(out, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)
        gps_text = "GPS NO FIX"
        if flight.get("latitude") is not None:
            gps_text = (
                f"UAV {flight['latitude']:.7f},{flight['longitude']:.7f} "
                f"ALT {float(flight.get('altitude') or 0):.1f}m HDG {float(flight.get('heading') or 0):.1f} "
                f"SAT {flight.get('satellites', 0)}"
            )
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, gps_text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return out

    def _build_event(
        self, det: Dict[str, Any], target: Dict[str, Any], flight: Dict[str, Any], sample: CameraSample
    ) -> Dict[str, Any]:
        event_id = f"{int(time.time() * 1000)}_T{det['track_id']}"
        detection = {
            "id": int(det["track_id"]),
            "track_id": f"T{det['track_id']}",
            "class": CAR_CLASS_NAME,
            "confidence": round(float(det["confidence"]), 6),
            "bbox": [round(float(v), 2) for v in det["bbox"]],
            "source": (
                "yolo-sliced-botsort-reid"
                if self.args.sliced_inference and self.args.with_reid
                else "yolo-sliced-botsort"
                if self.args.sliced_inference
                else "yolo-botsort-reid"
                if self.args.with_reid
                else "yolo-botsort"
            ),
            "detection_source": det.get("detection_source", "full"),
            "target_location": target,
        }
        return {
            "type": "car_detection",
            "event_id": event_id,
            "timestamp": utc_now(),
            "cars": 1,
            "camera": 0,
            "camera_timestamp_ms": sample.camera_timestamp_ms,
            "gps": flight,
            "target_location": target,
            "detections": [detection],
        }

    def _loop(self) -> None:
        last_flight_emit = 0.0
        try:
            if not self.args.fake_camera and self.args.sliced_inference:
                warm_sample = None
                deadline = time.monotonic() + 5.0
                while warm_sample is None and time.monotonic() < deadline:
                    warm_sample = self.camera.get()
                    if warm_sample is None:
                        time.sleep(0.05)
                if warm_sample is None:
                    raise RuntimeError("D455 did not provide a frame for worker GPU warm-up")
                started = time.monotonic()
                self._detect_sliced(warm_sample.color)
                import torch
                torch.cuda.synchronize()
                log.info("detection worker real-image warm-up completed in %.2fs", time.monotonic() - started)
            self._ready.set()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            log.exception("detection worker warm-up failed")
            return
        while self._running.is_set():
            started = time.monotonic()
            sample = self.camera.get()
            if sample is None:
                time.sleep(0.02)
                continue
            self._frame_index += 1
            flight = self.flight.get()
            try:
                raw_detections = self._infer(sample.color)
            except Exception:
                log.exception("inference error")
                time.sleep(0.1)
                continue
            inference_s = time.monotonic() - started
            if self._frame_index == 1 or self._frame_index % 10 == 0:
                log.info(
                    "inference frame=%d time=%.3fs effective=%.2f fps detections=%d",
                    self._frame_index,
                    inference_s,
                    1.0 / max(inference_s, 1e-6),
                    len(raw_detections),
                )

            detections: List[Dict[str, Any]] = []
            for det in raw_detections:
                tid = int(det["track_id"])
                self._hit_count[tid] = self._hit_count.get(tid, 0) + 1
                target = self.geolocator.locate(
                    sample.depth_m, sample.intrinsics, det["bbox"], flight
                )
                det["target_location"] = target
                detections.append(det)

                if self._hit_count[tid] >= self.args.min_hits and tid not in self._sent_tracks:
                    sigma = target.get("estimated_horizontal_sigma_m")
                    target_ready = (
                        target.get("valid")
                        and int(flight.get("fix_type") or 0) >= self.args.min_gps_fix
                        and sigma is not None
                        and float(sigma) <= self.args.max_target_sigma_m
                    )
                    # On the ground, still return the detected-car image so the
                    # complete radio/GUI path can be tested indoors.  Once armed,
                    # never send a target event without approved positioning.
                    ground_test_image = flight.get("armed") is False
                    if not target_ready and not ground_test_image:
                        if tid not in self._gate_notified_tracks:
                            self._gate_notified_tracks.add(tid)
                            blocked = {
                                "type": "target_gate_blocked",
                                "timestamp": utc_now(),
                                "track_id": f"T{tid}",
                                "reason": target.get("reason") or "gps_accuracy_gate",
                                "fix_type": flight.get("fix_type"),
                                "estimated_horizontal_sigma_m": sigma,
                            }
                            socketio.emit("target_status", blocked, namespace="/events")
                            self.frame_log.write(blocked)
                        continue
                    feature = self._track_features.get(tid)
                    if self._is_duplicate_target(target, feature):
                        self._sent_tracks.add(tid)
                        self.frame_log.write({
                            "type": "duplicate_target_suppressed",
                            "timestamp": utc_now(),
                            "track_id": tid,
                            "target_location": target,
                        })
                        log.info("duplicate target suppressed: T%d", tid)
                        continue
                    self._sent_tracks.add(tid)
                    self._sent_target_signatures.append({
                        "target": copy.deepcopy(target),
                        "feature": None if feature is None else feature.copy(),
                    })
                    event = self._build_event(det, target, flight, sample)
                    self.shared.set_event(event)
                    if self.ros2 is not None:
                        self.ros2.publish_event(event)
                    socketio.emit("detection", event, namespace="/events")
                    self.frame_log.write({**event, "log_record_type": "new_car_event"})
                    job = TransmissionJob(
                        event["event_id"], f"T{tid}", sample.color.copy(),
                        list(det["bbox"]), event,
                    )
                    self.transmitter.submit(job)
                    log.info(
                        "new car T%d confidence=%.3f target=%s",
                        tid, det["confidence"],
                        (f"{target.get('latitude'):.7f},{target.get('longitude'):.7f}"
                         if target.get("valid") else target.get("reason")),
                    )

            annotated = self._annotate(sample.color, detections, flight)
            if self.ros2 is not None:
                self.ros2.publish_frame(
                    sample, annotated, detections, flight, self._frame_index
                )
            self.main_recorder.write(annotated)
            self.frame_log.write({
                "type": "frame",
                "timestamp": utc_now(),
                "frame_index": self._frame_index,
                "camera_timestamp_ms": sample.camera_timestamp_ms,
                "flight": flight,
                "d455_imu": sample.imu,
                "intrinsics": asdict(sample.intrinsics),
                "detections": detections,
            })

            now = time.monotonic()
            if now - last_flight_emit >= 1.0:
                last_flight_emit = now
                socketio.emit("gps", flight, namespace="/events")

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, 1.0 / self.args.fps - elapsed))


SHARED = SharedOutput()
FLIGHT: Optional[FlightTelemetry] = None


def mjpeg_stream() -> Iterable[bytes]:
    seen = -1
    while True:
        data, revision = SHARED.get_jpeg()
        if data and revision != seen:
            seen = revision
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        else:
            time.sleep(0.02)


@app.route("/video")
@app.route("/video/cam0")
def video():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/target-image")
def target_image():
    data, revision = SHARED.get_jpeg()
    if not data:
        return Response(status=404)
    response = Response(data, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Image-Revision"] = str(revision)
    return response


@app.route("/status")
def status():
    return jsonify({
        "device": "jetson-orin-d455i",
        "gps": FLIGHT.get() if FLIGHT else None,
        "last_detection": SHARED.get_event(),
        "last_transmission": SHARED.get_transmission(),
        "transmitting": SHARED.transmitting,
    })


@app.route("/health")
def health():
    return jsonify({"ok": True, "timestamp": utc_now()})


def parse_args():
    p = argparse.ArgumentParser(description="Orin D455i precision car tracker")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--device", default="auto", help="auto, 0, or cpu")
    p.add_argument("--require-gpu", action="store_true",
                   help="abort startup unless CUDA PyTorch and a GPU are available")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--sliced-inference", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tile-size", type=int, default=720)
    p.add_argument("--tile-imgsz", type=int, default=640)
    p.add_argument("--tile-overlap", type=float, default=0.22)
    p.add_argument("--tile-nms-iou", type=float, default=0.50)
    p.add_argument("--full-frame-pass", action=argparse.BooleanOptionalAction, default=False,
                   help="extra context pass; disabled on 8 GB Orin to avoid unified-memory thrashing")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with-reid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reid-model", default="auto")
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--dedupe-near-m", type=float, default=1.0)
    p.add_argument("--dedupe-radius-m", type=float, default=4.0)
    p.add_argument("--dedupe-appearance", type=float, default=0.80)
    p.add_argument("--min-gps-fix", type=int, default=3)
    p.add_argument("--max-target-sigma-m", type=float, default=5.0)

    p.add_argument("--camera-serial", default="234222301359")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--fake-camera", action="store_true")
    p.add_argument("--color-only", action="store_true",
                   help="disable D455 depth alignment; recommended at 60 m with a rangefinder")
    p.add_argument("--min-depth", type=float, default=0.4)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--attitude-sigma-deg", type=float, default=1.0,
                   help="combined FC attitude and camera-mount angular uncertainty")

    p.add_argument("--camera-roll-deg", type=float, default=0.0)
    p.add_argument("--camera-pitch-deg", type=float, default=-90.0,
                   help="camera forward-axis pitch in aircraft FRD; -90 means looking down")
    p.add_argument("--camera-yaw-deg", type=float, default=0.0)
    p.add_argument("--camera-x-m", type=float, default=0.0, help="camera forward offset from FC")
    p.add_argument("--camera-y-m", type=float, default=0.0, help="camera right offset from FC")
    p.add_argument("--camera-z-m", type=float, default=0.0, help="camera down offset from FC")

    p.add_argument("--mavlink", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mock-flight", action="store_true")
    p.add_argument("--ros2", action="store_true",
                   help="publish standard ROS 2 topics for ros2 bag record")
    p.add_argument("--ros2-compressed-only", action="store_true",
                   help="publish only low-rate JPEG topics for bandwidth/storage saving")
    p.add_argument("--ros-image-fps", type=float, default=0.5,
                   help="annotated JPEG frequency in compressed-only mode")
    p.add_argument("--ros-jpeg-quality", type=int, default=50)
    p.add_argument("--ros2-warmup", type=float, default=0.0,
                   help="wait for rosbag topic discovery before camera/inference starts")

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--stream-fps", type=float, default=10.0)
    p.add_argument("--tx-width", type=int, default=480)
    p.add_argument("--tx-height", type=int, default=270)
    p.add_argument("--tx-steps", type=int, default=12)
    p.add_argument("--tx-interval", type=float, default=0.35)
    p.add_argument("--tx-jpeg-quality", type=int, default=65)
    p.add_argument("--tx-bps", type=int, default=0,
                   help="optional application throttle in bytes/s; 0 disables throttling")

    p.add_argument("--record-dir", default="/home/hrc/recordings/orin_car_project")
    p.add_argument("--video-encoder", default="auto",
                   help="auto, h264_nvenc, h264_v4l2m2m, libx264, or another ffmpeg encoder")
    p.add_argument("--run-seconds", type=float, default=0.0,
                   help="stop after N seconds; useful with --no-server tests")
    return p.parse_args()


def main() -> int:
    global FLIGHT
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("camera width, height, and fps must be positive")
    if args.tx_width <= 0 or args.tx_height <= 0 or not (1 <= args.tx_jpeg_quality <= 100):
        raise SystemExit("invalid transmission image settings")

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.record_dir).expanduser() / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=True), encoding="ascii"
    )
    log.info("session output: %s", session_dir)

    FLIGHT = FlightTelemetry(args.mavlink, args.baud, mock=args.mock_flight)
    camera = D455Camera(
        args.camera_serial,
        args.width,
        args.height,
        args.fps,
        args.fake_camera,
        enable_depth=not args.color_only,
    )
    ros2_publisher = (
        ROS2Publisher(args.ros2_compressed_only, args.ros_image_fps, args.ros_jpeg_quality)
        if args.ros2 else None
    )
    tracker = PrecisionTracker(args, session_dir, FLIGHT, camera, ros2_publisher)
    stopped = threading.Event()

    def request_stop(_sig=None, _frame=None):
        stopped.set()
        if not args.no_server:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if ros2_publisher is not None and args.ros2_warmup > 0:
        log.info("waiting %.1f seconds for ROS 2 recorder discovery", args.ros2_warmup)
        time.sleep(args.ros2_warmup)
    FLIGHT.start()
    camera.start()
    tracker.start()

    log.info("model=%s", args.model)
    log.info("camera=%s mount RPY=(%.2f, %.2f, %.2f) deg", args.camera_serial,
             args.camera_roll_deg, args.camera_pitch_deg, args.camera_yaw_deg)
    log.info("recording annotated video, progressive transmission video, and JSONL telemetry")

    try:
        if args.no_server:
            deadline = None if args.run_seconds <= 0 else time.monotonic() + args.run_seconds
            while not stopped.wait(0.2):
                if deadline is not None and time.monotonic() >= deadline:
                    break
        else:
            if args.run_seconds > 0:
                import _thread
                threading.Timer(args.run_seconds, _thread.interrupt_main).start()
            log.info("ground display: http://<orin-ip>:%d/video", args.port)
            try:
                socketio.run(
                    app, host=args.host, port=args.port, debug=False, use_reloader=False,
                    allow_unsafe_werkzeug=True,
                )
            except KeyboardInterrupt:
                pass
    finally:
        log.info("stopping and finalizing MP4 files")
        tracker.stop()
        camera.stop()
        FLIGHT.stop()
        if ros2_publisher is not None:
            ros2_publisher.stop()
        log.info("all outputs finalized in %s", session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
