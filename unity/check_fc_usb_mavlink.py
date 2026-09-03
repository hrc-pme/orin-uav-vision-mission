#!/usr/bin/env python3
"""Check flight-controller MAVLink over USB without requiring GPS fix."""

import argparse
import time

from pymavlink import mavutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--heartbeat-timeout", type=float, default=5.0)
    parser.add_argument("--listen-seconds", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Connecting MAVLink {args.port} @ {args.baud} ...", flush=True)
    conn = mavutil.mavlink_connection(
        args.port,
        baud=args.baud,
        autoreconnect=False,
        source_system=255,
        source_component=0,
    )
    heartbeat = conn.wait_heartbeat(timeout=args.heartbeat_timeout)
    if heartbeat is None:
        print("FAIL: no MAVLink HEARTBEAT. This is not a GPS problem.")
        return 1

    print(
        "PASS: HEARTBEAT "
        f"sys={conn.target_system} comp={conn.target_component} "
        f"mode={mavutil.mode_string_v10(heartbeat)} "
        f"type={heartbeat.type} autopilot={heartbeat.autopilot}"
    )
    deadline = time.time() + args.listen_seconds
    counts = {}
    while time.time() < deadline:
        msg = conn.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        msg_type = msg.get_type()
        counts[msg_type] = counts.get(msg_type, 0) + 1
        if msg_type == "GLOBAL_POSITION_INT":
            print(
                "GLOBAL_POSITION_INT "
                f"lat={getattr(msg, 'lat', 0) / 1e7:.7f} "
                f"lon={getattr(msg, 'lon', 0) / 1e7:.7f} "
                f"rel_alt_m={getattr(msg, 'relative_alt', 0) / 1000.0:.2f}"
            )
        elif msg_type == "GPS_RAW_INT":
            print(
                "GPS_RAW_INT "
                f"fix={getattr(msg, 'fix_type', 0)} "
                f"sats={getattr(msg, 'satellites_visible', 0)}"
            )
        elif msg_type == "ATTITUDE":
            print(
                "ATTITUDE "
                f"roll={getattr(msg, 'roll', 0.0):.3f} "
                f"pitch={getattr(msg, 'pitch', 0.0):.3f} "
                f"yaw={getattr(msg, 'yaw', 0.0):.3f}"
            )
        elif msg_type == "SYS_STATUS":
            voltage = getattr(msg, "voltage_battery", -1)
            print(f"SYS_STATUS voltage={voltage / 1000.0 if voltage >= 0 else -1:.2f}V")
    print("message_counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
