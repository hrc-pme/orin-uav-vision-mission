#!/usr/bin/env python3
"""Read-only local hardware/config preflight for the D455 car approach mission."""

import os
import sys
import time
from typing import List, Tuple

import yaml


CONFIG_PATH = "/home/hrc/xin_uav/unity/config_d455_car_approach.yaml"
UNITY_STATE_RECEIVER = (
    "/home/hrc/unity_windows_review/UAVDashboard/UavRosBridgeStateReceiver.cs"
)
RECORDER_PATH = "/home/hrc/xin_uav/unity/record_experiment.sh"
RUNNER_PATH = "/home/hrc/xin_uav/unity/run_d455_car_approach.sh"
UNITY_TARGET_UI = (
    "/home/hrc/unity_windows_review/UAVDashboard/TargetSelectionUIController.cs"
)
UNITY_IMAGE_RECEIVER = (
    "/home/hrc/unity_windows_review/UAVDashboard/CompressedImageUIReceiver.cs"
)
UNITY_COMMAND_PUBLISHER = (
    "/home/hrc/unity_windows_review/UAVDashboard/TargetCommandPublisher.cs"
)
UNITY_CANDIDATE_SUBSCRIBER = (
    "/home/hrc/unity_windows_review/UAVDashboard/TargetCandidateSubscriber.cs"
)
MISSION_BRIDGE = "/home/hrc/xin_uav/unity/d455_car_approach_bridge.py"
TELEMETRY_BRIDGE = "/home/hrc/xin_uav/unity/unity_dashboard_bridge.py"


def append_live_flight_controller_checks(
    checks: List[Tuple[str, bool, str]], config: dict
) -> None:
    """Passively sample MAVLink; never arm, change mode, or send flight commands."""
    mavlink_port = str(config.get("mavlink_port", ""))
    baud = int(config.get("mavlink_baud", 115200))
    sample_seconds = float(config.get("preflight_mavlink_sample_seconds", 8.0))
    minimum_satellites = int(config.get("minimum_gps_satellites", 8))
    maximum_eph_m = float(config.get("maximum_gps_eph_m", 5.0))
    minimum_voltage_v = float(config.get("minimum_battery_voltage_v", 21.0))
    minimum_remaining_pct = int(config.get("minimum_battery_remaining_pct", 25))

    connection = None
    try:
        from pymavlink import mavutil

        connection = mavutil.mavlink_connection(
            mavlink_port,
            baud=baud,
            autoreconnect=False,
            source_system=254,
        )
        heartbeat = None
        heartbeat_deadline = time.monotonic() + 4.0
        while time.monotonic() < heartbeat_deadline:
            candidate = connection.recv_match(
                type="HEARTBEAT",
                blocking=True,
                timeout=0.5,
            )
            if candidate is None:
                continue
            if (
                int(getattr(candidate, "type", mavutil.mavlink.MAV_TYPE_GCS))
                == mavutil.mavlink.MAV_TYPE_GCS
                or int(
                    getattr(
                        candidate,
                        "autopilot",
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    )
                )
                == mavutil.mavlink.MAV_AUTOPILOT_INVALID
            ):
                continue
            # Match the main bridge's controller selection.  AP6 also emits
            # an ArduPilot-looking heartbeat from system=255/component=235
            # with base_mode=128 and custom_mode=0; it is not the controlling
            # autopilot and must not become the telemetry request target.
            if not (
                int(getattr(candidate, "base_mode", 0))
                & int(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
            ):
                continue
            heartbeat = candidate
            connection.target_system = int(candidate.get_srcSystem())
            connection.target_component = int(candidate.get_srcComponent())
            break
        checks.append(
            (
                "flight controller heartbeat is live",
                heartbeat is not None,
                (
                    f"system={connection.target_system}, "
                    f"component={connection.target_component}"
                    if heartbeat is not None
                    else f"no heartbeat on {mavlink_port}"
                ),
            )
        )
        if heartbeat is None:
            return

        # A freshly rebooted ArduPilot USB link may emit HEARTBEAT/TIMESYNC
        # before it starts the GPS and battery streams.  Request these two
        # telemetry messages explicitly so preflight does not report a false
        # hardware failure before the main bridge has configured message rates.
        for message_id, interval_us in (
            (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 500_000),
            (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 200_000),
        ):
            connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                interval_us,
                0,
                0,
                0,
                0,
                0,
            )

        latest_gps = None
        latest_system_status = None
        unhealthy_messages = []
        deadline = time.monotonic() + sample_seconds
        while time.monotonic() < deadline:
            message = connection.recv_match(blocking=True, timeout=0.5)
            if message is None:
                continue
            message_type = message.get_type()
            if message_type == "GPS_RAW_INT":
                latest_gps = message
            elif message_type == "SYS_STATUS":
                latest_system_status = message
            elif message_type == "STATUSTEXT":
                text = str(getattr(message, "text", "")).strip()
                lowered = text.lower()
                if "battery" in lowered and (
                    "unhealthy" in lowered
                    or "failsafe" in lowered
                    or "critical" in lowered
                ):
                    unhealthy_messages.append(text)

        if latest_gps is None:
            checks.append(("outdoor GPS fix is flight-ready", False, "no GPS_RAW_INT"))
            checks.append(("outdoor GPS accuracy is acceptable", False, "no GPS_RAW_INT"))
        else:
            fix_type = int(getattr(latest_gps, "fix_type", 0))
            satellites = int(getattr(latest_gps, "satellites_visible", 0))
            raw_eph = int(getattr(latest_gps, "eph", 65535))
            eph_m = raw_eph / 100.0 if 0 <= raw_eph < 65535 else float("inf")
            checks.append(
                (
                    "outdoor GPS fix is flight-ready",
                    fix_type >= 3 and satellites >= minimum_satellites,
                    (
                        f"fix_type={fix_type}, satellites={satellites}; "
                        f"required fix>=3 and satellites>={minimum_satellites}"
                    ),
                )
            )
            checks.append(
                (
                    "outdoor GPS accuracy is acceptable",
                    fix_type >= 3
                    and 0 < raw_eph < 65535
                    and eph_m <= maximum_eph_m,
                    (
                        f"fix_type={fix_type}, EPH={eph_m:.2f} m; "
                        f"required <= {maximum_eph_m:.2f} m"
                    ),
                )
            )

        if latest_system_status is None:
            checks.append(
                ("main flight battery is flight-ready", False, "no SYS_STATUS")
            )
        else:
            voltage_mv = int(getattr(latest_system_status, "voltage_battery", 0))
            remaining_pct = int(
                getattr(latest_system_status, "battery_remaining", -1)
            )
            voltage_v = voltage_mv / 1000.0
            checks.append(
                (
                    "main flight battery is flight-ready",
                    voltage_v >= minimum_voltage_v
                    and remaining_pct >= minimum_remaining_pct
                    and not unhealthy_messages,
                    (
                        f"{voltage_v:.2f} V, {remaining_pct}%; "
                        f"required >= {minimum_voltage_v:.2f} V and "
                        f">= {minimum_remaining_pct}%"
                        + (
                            "; " + " | ".join(unhealthy_messages[-2:])
                            if unhealthy_messages
                            else ""
                        )
                    ),
                )
            )
    except Exception as exc:
        checks.append(("live flight-controller preflight", False, str(exc)))
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    checks: List[Tuple[str, bool, str]] = []
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        checks.append(("mission config parses", isinstance(config, dict), CONFIG_PATH))
    except Exception as exc:
        print(f"[FAIL] mission config parses: {exc}")
        return 1

    for key in ("engine_path", "fallback_model_path", "tracker"):
        path = os.path.expanduser(str(config.get(key, "")))
        checks.append((f"{key} exists", os.path.isfile(path), path))

    mavlink_port = str(config.get("mavlink_port", ""))
    checks.append(
        (
            "MAVLink serial device exists",
            os.path.exists(mavlink_port),
            mavlink_port,
        )
    )
    checks.append(
        (
            "MAVLink serial read/write permission",
            os.access(mavlink_port, os.R_OK | os.W_OK),
            mavlink_port,
        )
    )

    requested_profile = (
        int(config["camera_width"]),
        int(config["camera_height"]),
        int(config["camera_fps"]),
    )
    requested_serial = str(config.get("realsense_serial", ""))
    try:
        import pyrealsense2 as rs

        context = rs.context()
        matching_device = None
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            if not requested_serial or serial == requested_serial:
                matching_device = device
                requested_serial = serial
                break
        checks.append(
            ("D455 serial is connected", matching_device is not None, requested_serial)
        )
        profiles = set()
        if matching_device is not None:
            for sensor in matching_device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() != rs.stream.color:
                        continue
                    video = profile.as_video_stream_profile()
                    if profile.format() == rs.format.bgr8:
                        profiles.add((video.width(), video.height(), profile.fps()))
        checks.append(
            (
                "D455 BGR profile is supported",
                requested_profile in profiles,
                f"{requested_profile[0]}x{requested_profile[1]}@{requested_profile[2]}",
            )
        )
    except Exception as exc:
        checks.append(("pyrealsense2/D455 query", False, str(exc)))

    checks += [
        (
            "target accepts all car colors",
            str(config.get("target_color_mode", "")).lower() == "any",
            str(config.get("target_color_mode")),
        ),
        (
            "automatic verification enabled",
            bool(config.get("enable_auto_verification")),
            str(config.get("enable_auto_verification")),
        ),
        (
            "verification altitude is above final altitude",
            float(config.get("verification_altitude_m", 0.0))
            > float(config.get("goto_relative_altitude_m", 0.0)),
            (
                f"{config.get('verification_altitude_m')} m -> "
                f"{config.get('goto_relative_altitude_m')} m"
            ),
        ),
        (
            "CONFIRM starts detector-independent fixed-position descent to 20 m",
            bool(config.get("direct_visual_servo_enabled"))
            and not bool(config.get("staircase_visual_servo_enabled"))
            and bool(
                config.get("direct_fixed_position_descent_after_confirm")
            )
            and float(config.get("direct_visual_confirm_min_altitude_m", 0.0))
            >= 80.0
            and float(config.get("direct_visual_confirm_max_altitude_m", 999.0))
            <= 130.0
            and float(config.get("goto_relative_altitude_m", 0.0)) == 20.0
            and float(
                config.get(
                    "direct_fixed_position_confirm_center_ratio", 99.0
                )
            )
            <= 0.60
            and float(
                config.get(
                    "direct_fixed_position_confirm_max_detection_age_s",
                    99.0,
                )
            )
            <= 1.0
            and float(
                config.get("direct_fixed_position_max_drift_m", 99.0)
            )
            <= 10.0
            and int(config.get("minimum_battery_remaining_pct", 0)) >= 50
            and float(config.get("final_tracking_timeout_s", 0.0)) >= 300.0
            and not bool(config.get("draw_navigation_safe_region", True)),
            (
                f"confirm {config.get('direct_visual_confirm_min_altitude_m')}-"
                f"{config.get('direct_visual_confirm_max_altitude_m')} m -> "
                "fixed current GPS/heading -> 20 m, "
                f"confirm center<={config.get('direct_fixed_position_confirm_center_ratio')}, "
                "post-CONFIRM detector ignored, "
                f"battery>={config.get('minimum_battery_remaining_pct')}%"
            ),
        ),
        (
            "low-altitude verification survives brief tracker gaps",
            float(config.get("verification_miss_grace_s", 0.0)) >= 5.0
            and int(config.get("verification_min_samples", 0)) >= 5
            and float(config.get("verification_continuity_radius_m", 999.0))
            <= 15.0
            and float(config.get("verification_max_position_error_m", 999.0))
            <= 12.0
            and float(config.get("verification_ready_hold_s", 0.0)) >= 5.0
            and float(config.get("target_max_age_s", 0.0))
            > float(config.get("verification_ready_hold_s", 999.0)),
            (
                f"{config.get('verification_min_samples')} fixes, "
                f"grace={config.get('verification_miss_grace_s')} s, "
                f"ready hold={config.get('verification_ready_hold_s')} s, "
                f"continuity={config.get('verification_continuity_radius_m')} m, "
                f"error<={config.get('verification_max_position_error_m')} m"
            ),
        ),
        (
            "automatic descent rejects clipped edge cars",
            float(config.get("auto_verification_min_edge_margin_ratio", 0.0))
            >= 0.15
            and float(config.get("auto_verification_min_stable_seconds", 0.0))
            >= 3.0
            and float(config.get("auto_verification_detection_max_age_s", 99.0))
            <= 1.0
            and float(config.get("verification_min_edge_margin_ratio", 0.0))
            >= 0.05,
            (
                f"high margin={config.get('auto_verification_min_edge_margin_ratio')}, "
                f"stable={config.get('auto_verification_min_stable_seconds')} s, "
                f"low margin={config.get('verification_min_edge_margin_ratio')}"
            ),
        ),
        (
            "Unity draws one confirmed car box only",
            not bool(config.get("draw_non_white_detections", True))
            and int(config.get("visual_display_max_detections", 99)) == 1,
            (
                f"confirmed-only={not bool(config.get('draw_non_white_detections', True))}, "
                f"max={config.get('visual_display_max_detections')}"
            ),
        ),
        (
            "D455 depth disabled above its useful range",
            not bool(config.get("enable_depth"))
            and not bool(config.get("publish_depth")),
            "color + aircraft attitude/altitude projection",
        ),
        (
            "Unity receives annotated JPEG only",
            not bool(config.get("publish_raw_compressed", True))
            and bool(config.get("publish_annotated_compressed", True)),
            "/d455i/color/image_annotated/compressed",
        ),
        (
            "rosbag can request full-resolution raw RGB",
            bool(config.get("publish_raw_compressed_when_subscribed", False))
            and int(config.get("raw_publish_image_max_width", 0)) >= 1280
            and int(config.get("raw_publish_image_max_height", 0)) >= 720,
            (
                f"{config.get('raw_publish_image_max_width')}x"
                f"{config.get('raw_publish_image_max_height')} "
                f"JPEG quality={config.get('raw_jpeg_quality')}"
            ),
        ),
        (
            "rosbag can request D455 camera calibration",
            bool(config.get("publish_camera_info_when_subscribed", False)),
            "/d455i/color/camera_info",
        ),
        (
            "Unity image queue keeps latest frame only",
            int(config.get("image_qos_depth", 10)) == 1,
            f"QoS depth={config.get('image_qos_depth')}",
        ),
        (
            "Unity image bandwidth is limited",
            int(config.get("publish_image_max_width", 0)) == 400
            and int(config.get("publish_image_max_height", 0)) == 225
            and int(config.get("jpeg_quality", 100)) <= 50,
            (
                f"{config.get('publish_image_max_width')}x"
                f"{config.get('publish_image_max_height')} "
                f"JPEG quality={config.get('jpeg_quality')}"
            ),
        ),
        (
            "Unity target preview is a small compressed car crop",
            bool(config.get("include_candidate_image_base64", False))
            and int(config.get("candidate_image_max_width", 9999)) <= 256
            and int(config.get("candidate_image_max_height", 9999)) <= 160
            and int(config.get("candidate_jpeg_quality", 100)) <= 55
            and float(config.get("candidate_image_resend_interval_s", 0.0)) >= 5.0,
            (
                f"{config.get('candidate_image_max_width')}x"
                f"{config.get('candidate_image_max_height')} "
                f"JPEG quality={config.get('candidate_jpeg_quality')}, "
                f"resend={config.get('candidate_image_resend_interval_s')} s"
            ),
        ),
    ]

    append_live_flight_controller_checks(checks, config)

    try:
        with open(UNITY_STATE_RECEIVER, "r", encoding="utf-8") as stream:
            unity_state_source = stream.read()
        checks.append(
            (
                "Unity WGS84 drives Cesium globe anchor",
                "/uav/telemetry_json" in unity_state_source
                and "CesiumGlobeAnchor" in unity_state_source
                and "SetPositionLongitudeLatitudeHeight" in unity_state_source,
                UNITY_STATE_RECEIVER,
            )
        )
    except Exception as exc:
        checks.append(("Unity map receiver is readable", False, str(exc)))

    try:
        with open(UNITY_TARGET_UI, "r", encoding="utf-8") as stream:
            unity_target_source = stream.read()
        with open(UNITY_IMAGE_RECEIVER, "r", encoding="utf-8") as stream:
            unity_image_source = stream.read()
        preview_start = unity_target_source.index(
            "private Texture ResolvePreviewTexture()"
        )
        preview_end = unity_target_source.index(
            "private void RequestPreviewDecode", preview_start
        )
        preview_source = unity_target_source[preview_start:preview_end]
        checks.append(
            (
                "Unity right panel never falls back to the live image",
                "liveImageReceiver.LatestTexture" not in preview_source
                and "return null;" in preview_source,
                UNITY_TARGET_UI,
            )
        )
        checks.append(
            (
                "Unity live image preserves the complete camera aspect ratio",
                "displayWidth / sourceAspect" in unity_image_source
                and "new Rect(0.0f, 0.0f, 1.0f, 1.0f)" in unity_image_source,
                "400x225 full-frame display; no crop",
            )
        )
        checks.append(
            (
                "Unity keeps CANCEL enabled during final visual servo",
                'latestApproach.state == "FINAL_APPROACH"' in unity_target_source,
                UNITY_TARGET_UI,
            )
        )
        checks.append(
            (
                "Unity shows direct fixed-position descent",
                "ALTITUDE WINDOW: 80-130 m" in unity_target_source
                and "DIRECT DESCENT TO " in unity_target_source
                and "POSITION LOCKED - DESCENDING" in unity_target_source,
                UNITY_TARGET_UI,
            )
        )
    except Exception as exc:
        checks.append(("Unity target-only preview is readable", False, str(exc)))

    try:
        with open(UNITY_COMMAND_PUBLISHER, "r", encoding="utf-8") as stream:
            unity_command_source = stream.read()
        with open(UNITY_CANDIDATE_SUBSCRIBER, "r", encoding="utf-8") as stream:
            unity_candidate_source = stream.read()
        checks.append(
            (
                "Unity drains ROS command status to the newest message",
                "while (pendingStatusJson.Count > 0)" in unity_command_source,
                UNITY_COMMAND_PUBLISHER,
            )
        )
        checks.append(
            (
                "Unity drains candidate packets to the newest state",
                "while (pendingJsonMessages.Count > 0)" in unity_candidate_source,
                UNITY_CANDIDATE_SUBSCRIBER,
            )
        )
    except Exception as exc:
        checks.append(("Unity ROS queue sources are readable", False, str(exc)))

    try:
        with open(MISSION_BRIDGE, "r", encoding="utf-8") as stream:
            mission_source = stream.read()
        with open(TELEMETRY_BRIDGE, "r", encoding="utf-8") as stream:
            telemetry_source = stream.read()
        checks.append(
            (
                "Orin freezes CONFIRM position and descends detector-independently",
                "direct_fixed_position_descent_after_confirm" in mission_source
                and "def _fixed_position_descent_tick" in mission_source
                and '"FIXED_POSITION_DESCENT"' in mission_source
                and "send_reposition(" in mission_source
                and "MAV_FRAME_GLOBAL_RELATIVE_ALT_INT" in telemetry_source,
                f"{MISSION_BRIDGE} + {TELEMETRY_BRIDGE}",
            )
        )
        checks.append(
            (
                "Orin fixed descent explicitly holds current heading",
                'hold_yaw_rad = float(self.get().get("yaw") or 0.0)'
                in telemetry_source
                and "POSITION_TARGET_TYPEMASK_YAW_IGNORE" in telemetry_source,
                f"{MISSION_BRIDGE} + {TELEMETRY_BRIDGE}",
            )
        )
        checks.append(
            (
                "Orin gates re-identification and stops divergent correction",
                "single_target_states.add(READY_FOR_UNITY_CONFIRM)" in mission_source
                and "candidates[:] =" in mission_source
                and "staircase_reid_min_appearance_score" in mission_source
                and "stopped before runaway" in mission_source,
                MISSION_BRIDGE,
            )
        )
    except Exception as exc:
        checks.append(("Orin visual-servo sources are readable", False, str(exc)))

    try:
        with open(RECORDER_PATH, "r", encoding="utf-8") as stream:
            recorder_source = stream.read()
        required_record_topics = (
            "/d455i/color/image_raw/compressed",
            "/d455i/color/image_annotated/compressed",
            "/d455i/color/camera_info",
            "/unity/detections_json",
            "/unity/target_command",
            "/unity/target_command_status",
            "/uav/gps",
            "/uav/imu",
            "/uav/velocity_ned",
            "/uav/battery",
            "/uav/telemetry_json",
            "/uav/mavlink_json",
        )
        missing_topics = [
            topic for topic in required_record_topics if topic not in recorder_source
        ]
        checks.append(
            (
                "rosbag records RGB, mission, and flight-controller data",
                not missing_topics,
                "all required topics"
                if not missing_topics
                else "missing " + ",".join(missing_topics),
            )
        )
    except Exception as exc:
        checks.append(("recorder is readable", False, str(exc)))

    try:
        with open(RUNNER_PATH, "r", encoding="utf-8") as stream:
            runner_source = stream.read()
        checks.append(
            (
                "outdoor mission runner enforces preflight",
                "./preflight_d455_car_approach.py" in runner_source
                and "D455_APPROACH_INDOOR_SAFE" in runner_source,
                RUNNER_PATH,
            )
        )
    except Exception as exc:
        checks.append(("mission runner is readable", False, str(exc)))

    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}")
    success = all(passed for _label, passed, _detail in checks)
    print(
        "\nLOCAL PREFLIGHT PASS"
        if success
        else "\nLOCAL PREFLIGHT FAILED -- do not start the automatic mission"
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
