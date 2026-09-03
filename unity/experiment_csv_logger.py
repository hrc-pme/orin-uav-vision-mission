#!/usr/bin/env python3
"""CSV experiment logger for the Unity UAV dashboard ROS 2 topics.

This node is intentionally read-only.  It records enough timestamps and fields
to verify, after a run, whether Unity showed a candidate, whether CONFIRM was
pressed, whether Orin NX received it, and whether the command status reached
ACCEPTED / EXECUTING / ARRIVED.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import time
from typing import Any, Dict
from zoneinfo import ZoneInfo

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import BatteryState, CompressedImage, Imu, NavSatFix
from std_msgs.msg import String


TAIWAN_TZ = ZoneInfo("Asia/Taipei")


def tw_time_text(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(float(timestamp), TAIWAN_TZ).isoformat(timespec="milliseconds")


def ros_stamp_to_float(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class CsvWriter:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()
        self.last_flush_monotonic = time.monotonic()

    def row(self, **kwargs: Any) -> None:
        self.writer.writerow(kwargs)
        now = time.monotonic()
        if now - self.last_flush_monotonic >= 1.0:
            self.file.flush()
            self.last_flush_monotonic = now

    def close(self) -> None:
        self.file.flush()
        self.file.close()


class ExperimentCsvLogger(Node):
    def __init__(self, out_dir: str):
        super().__init__("uav_experiment_csv_logger")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.start_wall = time.time()
        self.latest_gps: Dict[str, Any] = {}
        self.latest_imu: Dict[str, Any] = {}
        self.latest_velocity: Dict[str, Any] = {}
        self.latest_battery: Dict[str, Any] = {}
        self.latest_telemetry: Dict[str, Any] = {}
        self.image_dir = os.path.join(out_dir, "compressed_images")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "annotated"), exist_ok=True)
        self.image_index = 0
        self.candidate_image_dir = os.path.join(out_dir, "candidate_images")
        os.makedirs(self.candidate_image_dir, exist_ok=True)
        self.candidate_image_index = 0

        self.detections = CsvWriter(
            os.path.join(out_dir, "detections.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "message_timestamp",
                "message_time_tw",
                "candidate_index",
                "candidate_id",
                "track_id",
                "class_name",
                "confidence",
                "latitude",
                "longitude",
                "twd97_e",
                "twd97_n",
                "position_error_m",
                "locked_at_epoch",
                "stable_for_s",
                "age_s",
                "display_hold_remaining_s",
                "candidate_count",
                "candidate_image_filename",
            ],
        )
        self.debug_detections = CsvWriter(
            os.path.join(out_dir, "debug_detections.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "message_timestamp",
                "message_time_tw",
                "detection_index",
                "track_id",
                "confidence",
                "white_ratio",
                "white_pixels",
                "red_ratio",
                "edge_density",
                "aspect_ratio",
                "white_streak",
                "white_confirmed",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "detector_source",
                "camera_ok",
            ],
        )
        self.commands = CsvWriter(
            os.path.join(out_dir, "commands.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "action",
                "command_id",
                "candidate_id",
                "track_id",
                "raw_json",
            ],
        )
        self.status = CsvWriter(
            os.path.join(out_dir, "status.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "command_id",
                "track_id",
                "status",
                "message",
                "distance_remaining_m",
                "raw_json",
            ],
        )
        self.telemetry = CsvWriter(
            os.path.join(out_dir, "telemetry.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "connected",
                "port",
                "baud",
                "gps_valid",
                "latitude",
                "longitude",
                "altitude",
                "relative_altitude",
                "satellites",
                "eph_m",
                "epv_m",
                "gps1_fix_type",
                "gps1_satellites",
                "gps1_eph_m",
                "gps1_epv_m",
                "gps1_latitude",
                "gps1_longitude",
                "gps1_altitude",
                "gps2_fix_type",
                "gps2_satellites",
                "gps2_eph_m",
                "gps2_epv_m",
                "gps2_latitude",
                "gps2_longitude",
                "gps2_altitude",
                "flight_mode",
                "armed",
                "vx",
                "vy",
                "vz",
                "roll",
                "pitch",
                "yaw",
                "battery_voltage_v",
                "battery_remaining_pct",
                "d455i_connected",
                "detector_source",
                "flight_control_ready",
                "flight_control_blockers",
            ],
        )
        self.mavlink = CsvWriter(
            os.path.join(out_dir, "mavlink.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "message_timestamp",
                "message_time_tw",
                "message_type",
                "raw_json",
            ],
        )
        self.gps = CsvWriter(
            os.path.join(out_dir, "gps.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "ros_stamp",
                "ros_time_tw",
                "status",
                "latitude",
                "longitude",
                "altitude",
                "covariance_type",
            ],
        )
        self.imu = CsvWriter(
            os.path.join(out_dir, "imu.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "ros_stamp",
                "ros_time_tw",
                "frame_id",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ],
        )
        self.images = CsvWriter(
            os.path.join(out_dir, "images.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "ros_stamp",
                "ros_time_tw",
                "image_kind",
                "frame_id",
                "format",
                "bytes",
                "filename",
                "topic",
            ],
        )
        self.summary = CsvWriter(
            os.path.join(out_dir, "summary_events.csv"),
            [
                "wall_time",
                "local_time_tw",
                "elapsed_s",
                "event",
                "command_id",
                "candidate_id",
                "track_id",
                "status",
                "latitude",
                "longitude",
                "distance_remaining_m",
                "message",
            ],
        )

        self.create_subscription(String, "/unity/detections_json", self.on_detections, 10)
        self.create_subscription(String, "/unity/target_command", self.on_command, 10)
        self.create_subscription(String, "/unity/target_command_status", self.on_status, 10)
        self.create_subscription(String, "/uav/telemetry_json", self.on_telemetry, 10)
        self.create_subscription(String, "/uav/mavlink_json", self.on_mavlink, 200)
        self.create_subscription(NavSatFix, "/uav/gps", self.on_gps, 10)
        self.create_subscription(Imu, "/uav/imu", self.on_imu, 10)
        self.create_subscription(TwistStamped, "/uav/velocity_ned", self.on_velocity, 10)
        self.create_subscription(BatteryState, "/uav/battery", self.on_battery, 10)
        self.create_subscription(
            CompressedImage,
            "/d455i/color/image_raw/compressed",
            lambda msg: self.on_image(msg, "raw", "/d455i/color/image_raw/compressed"),
            10,
        )
        self.create_subscription(
            CompressedImage,
            "/d455i/color/image_annotated/compressed",
            lambda msg: self.on_image(
                msg, "annotated", "/d455i/color/image_annotated/compressed"
            ),
            10,
        )
        self.get_logger().info(f"CSV experiment logging to {out_dir}")

    def now_fields(self) -> Dict[str, float]:
        wall = time.time()
        return {
            "wall_time": wall,
            "local_time_tw": tw_time_text(wall),
            "elapsed_s": wall - self.start_wall,
        }

    def on_detections(self, msg: String) -> None:
        data = safe_json_loads(msg.data)
        candidates = data.get("candidates") or []
        if not isinstance(candidates, list):
            return
        base = self.now_fields()
        timestamp = data.get("timestamp", "")
        try:
            message_time_tw = tw_time_text(float(timestamp))
        except (TypeError, ValueError):
            message_time_tw = ""
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_image_filename = ""
            encoded_image = str(candidate.get("image_base64") or "")
            if encoded_image:
                try:
                    self.candidate_image_index += 1
                    candidate_image_filename = (
                        "candidate_images/"
                        f"candidate_{self.candidate_image_index:06d}_"
                        f"track_{candidate.get('track_id', 'unknown')}_"
                        f"{float(timestamp or time.time()):.9f}.jpg"
                    )
                    with open(os.path.join(self.out_dir, candidate_image_filename), "wb") as image_file:
                        image_file.write(base64.b64decode(encoded_image))
                except Exception as exc:
                    self.get_logger().warning(f"Failed to save candidate image: {exc}")
                    candidate_image_filename = ""
            row = {
                **base,
                "message_timestamp": timestamp,
                "message_time_tw": message_time_tw,
                "candidate_index": index,
                "candidate_count": len(candidates),
                "candidate_id": candidate.get("candidate_id", ""),
                "track_id": candidate.get("track_id", ""),
                "class_name": candidate.get("class_name", ""),
                "confidence": candidate.get("confidence", ""),
                "latitude": candidate.get("latitude", ""),
                "longitude": candidate.get("longitude", ""),
                "twd97_e": candidate.get("twd97_e", ""),
                "twd97_n": candidate.get("twd97_n", ""),
                "position_error_m": candidate.get("position_error_m", ""),
                "locked_at_epoch": candidate.get("locked_at_epoch", ""),
                "stable_for_s": candidate.get("stable_for_s", ""),
                "age_s": candidate.get("age_s", ""),
                "display_hold_remaining_s": candidate.get("display_hold_remaining_s", ""),
                "candidate_image_filename": candidate_image_filename,
            }
            self.detections.row(**row)
        debug_detections = data.get("debug_detections") or []
        if isinstance(debug_detections, list):
            for index, detection in enumerate(debug_detections):
                if not isinstance(detection, dict):
                    continue
                bbox = detection.get("bbox") or []
                bbox = bbox if isinstance(bbox, list) else []
                self.debug_detections.row(
                    **base,
                    message_timestamp=timestamp,
                    message_time_tw=message_time_tw,
                    detection_index=index,
                    track_id=detection.get("track_id", ""),
                    confidence=detection.get("confidence", ""),
                    white_ratio=detection.get("white_ratio", ""),
                    white_pixels=detection.get("white_pixels", ""),
                    red_ratio=detection.get("red_ratio", ""),
                    edge_density=detection.get("edge_density", ""),
                    aspect_ratio=detection.get("aspect_ratio", ""),
                    white_streak=detection.get("white_streak", ""),
                    white_confirmed=detection.get("white_confirmed", ""),
                    bbox_x1=bbox[0] if len(bbox) > 0 else "",
                    bbox_y1=bbox[1] if len(bbox) > 1 else "",
                    bbox_x2=bbox[2] if len(bbox) > 2 else "",
                    bbox_y2=bbox[3] if len(bbox) > 3 else "",
                    detector_source=data.get("detector_source", ""),
                    camera_ok=data.get("camera_ok", ""),
                )

    def on_command(self, msg: String) -> None:
        data = safe_json_loads(msg.data)
        row = {
            **self.now_fields(),
            "action": str(data.get("action", "")).upper(),
            "command_id": data.get("command_id", ""),
            "candidate_id": data.get("candidate_id", ""),
            "track_id": data.get("track_id", ""),
            "raw_json": msg.data,
        }
        self.commands.row(**row)
        self.summary.row(
            **self.now_fields(),
            event="UNITY_CONFIRM_RECEIVED_BY_ROS",
            command_id=row["command_id"],
            candidate_id=row["candidate_id"],
            track_id=row["track_id"],
            status="",
            latitude="",
            longitude="",
            distance_remaining_m="",
            message=row["action"],
        )

    def on_status(self, msg: String) -> None:
        data = safe_json_loads(msg.data)
        row = {
            **self.now_fields(),
            "command_id": data.get("command_id", ""),
            "track_id": data.get("track_id", ""),
            "status": data.get("status", ""),
            "message": data.get("message", ""),
            "distance_remaining_m": data.get("distance_remaining_m", ""),
            "raw_json": msg.data,
        }
        self.status.row(**row)
        self.summary.row(
            **self.now_fields(),
            event="ORIN_COMMAND_STATUS",
            command_id=row["command_id"],
            candidate_id="",
            track_id=row["track_id"],
            status=row["status"],
            latitude=self.latest_gps.get("latitude", ""),
            longitude=self.latest_gps.get("longitude", ""),
            distance_remaining_m=row["distance_remaining_m"],
            message=row["message"],
        )

    def on_telemetry(self, msg: String) -> None:
        data = safe_json_loads(msg.data)
        self.latest_telemetry = data
        gps1 = data.get("gps1") if isinstance(data.get("gps1"), dict) else {}
        gps2 = data.get("gps2") if isinstance(data.get("gps2"), dict) else {}
        self.telemetry.row(
            **self.now_fields(),
            connected=data.get("connected", ""),
            port=data.get("port", ""),
            baud=data.get("baud", ""),
            gps_valid=data.get("gps_valid", ""),
            latitude=data.get("latitude", ""),
            longitude=data.get("longitude", ""),
            altitude=data.get("altitude", ""),
            relative_altitude=data.get("relative_altitude", ""),
            satellites=data.get("satellites", ""),
            eph_m=data.get("eph_m", ""),
            epv_m=data.get("epv_m", ""),
            gps1_fix_type=gps1.get("fix_type", ""),
            gps1_satellites=gps1.get("satellites", ""),
            gps1_eph_m=gps1.get("eph_m", ""),
            gps1_epv_m=gps1.get("epv_m", ""),
            gps1_latitude=gps1.get("latitude", ""),
            gps1_longitude=gps1.get("longitude", ""),
            gps1_altitude=gps1.get("altitude", ""),
            gps2_fix_type=gps2.get("fix_type", ""),
            gps2_satellites=gps2.get("satellites", ""),
            gps2_eph_m=gps2.get("eph_m", ""),
            gps2_epv_m=gps2.get("epv_m", ""),
            gps2_latitude=gps2.get("latitude", ""),
            gps2_longitude=gps2.get("longitude", ""),
            gps2_altitude=gps2.get("altitude", ""),
            flight_mode=data.get("flight_mode", ""),
            armed=data.get("armed", ""),
            vx=data.get("vx", ""),
            vy=data.get("vy", ""),
            vz=data.get("vz", ""),
            roll=data.get("roll", ""),
            pitch=data.get("pitch", ""),
            yaw=data.get("yaw", ""),
            battery_voltage_v=data.get("voltage_v", ""),
            battery_remaining_pct=data.get("battery_remaining_pct", ""),
            d455i_connected=data.get("d455i_connected", ""),
            detector_source=data.get("detector_source", ""),
            flight_control_ready=data.get("flight_control_ready", ""),
            flight_control_blockers="|".join(data.get("flight_control_blockers") or []),
        )

    def on_mavlink(self, msg: String) -> None:
        data = safe_json_loads(msg.data)
        timestamp = data.get("timestamp", "")
        try:
            message_time_tw = tw_time_text(float(timestamp))
        except (TypeError, ValueError):
            message_time_tw = ""
        self.mavlink.row(
            **self.now_fields(),
            message_timestamp=timestamp,
            message_time_tw=message_time_tw,
            message_type=data.get("message_type", ""),
            raw_json=msg.data,
        )

    def on_gps(self, msg: NavSatFix) -> None:
        self.latest_gps = {
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude,
        }
        self.gps.row(
            **self.now_fields(),
            ros_stamp=ros_stamp_to_float(msg),
            ros_time_tw=tw_time_text(ros_stamp_to_float(msg)),
            status=msg.status.status,
            latitude=msg.latitude,
            longitude=msg.longitude,
            altitude=msg.altitude,
            covariance_type=msg.position_covariance_type,
        )

    def on_imu(self, msg: Imu) -> None:
        self.latest_imu = {
            "orientation_x": msg.orientation.x,
            "orientation_y": msg.orientation.y,
            "orientation_z": msg.orientation.z,
            "orientation_w": msg.orientation.w,
        }
        self.imu.row(
            **self.now_fields(),
            ros_stamp=ros_stamp_to_float(msg),
            ros_time_tw=tw_time_text(ros_stamp_to_float(msg)),
            frame_id=msg.header.frame_id,
            orientation_x=msg.orientation.x,
            orientation_y=msg.orientation.y,
            orientation_z=msg.orientation.z,
            orientation_w=msg.orientation.w,
            angular_velocity_x=msg.angular_velocity.x,
            angular_velocity_y=msg.angular_velocity.y,
            angular_velocity_z=msg.angular_velocity.z,
            linear_acceleration_x=msg.linear_acceleration.x,
            linear_acceleration_y=msg.linear_acceleration.y,
            linear_acceleration_z=msg.linear_acceleration.z,
        )

    def on_velocity(self, msg: TwistStamped) -> None:
        self.latest_velocity = {
            "vx": msg.twist.linear.x,
            "vy": msg.twist.linear.y,
            "vz": msg.twist.linear.z,
        }

    def on_battery(self, msg: BatteryState) -> None:
        self.latest_battery = {"voltage": msg.voltage, "percentage": msg.percentage}

    def on_image(self, msg: CompressedImage, image_kind: str, topic: str) -> None:
        ros_stamp = ros_stamp_to_float(msg)
        self.image_index += 1
        stamp_source = ros_stamp if ros_stamp > 0.0 else time.time()
        stamp_text = tw_time_text(stamp_source).replace(":", "").replace("-", "")
        filename = f"{image_kind}/frame_{self.image_index:06d}_{stamp_text}.jpg"
        path = os.path.join(self.image_dir, filename)
        with open(path, "wb") as image_file:
            image_file.write(bytes(msg.data))
        self.images.row(
            **self.now_fields(),
            ros_stamp=ros_stamp,
            ros_time_tw=tw_time_text(stamp_source),
            image_kind=image_kind,
            frame_id=msg.header.frame_id,
            format=msg.format,
            bytes=len(msg.data),
            filename=os.path.join("compressed_images", filename),
            topic=topic,
        )

    def close(self) -> None:
        for writer in (
            self.detections,
            self.debug_detections,
            self.commands,
            self.status,
            self.telemetry,
            self.mavlink,
            self.gps,
            self.imu,
            self.images,
            self.summary,
        ):
            writer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rclpy.init()
    node = ExperimentCsvLogger(args.out_dir)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
