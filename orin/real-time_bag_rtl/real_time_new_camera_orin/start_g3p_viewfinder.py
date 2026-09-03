#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import sys
import time


def log(message):
    print(f"[g3p-control] {message}", flush=True)


def parse_json_stream(text):
    decoder = json.JSONDecoder()
    items = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        try:
            item, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        items.append(item)
        pos = end
    return items


def recv_available(sock, timeout_sec):
    sock.settimeout(timeout_sec)
    chunks = []
    while True:
        try:
            data = sock.recv(65535)
        except socket.timeout:
            break
        if not data:
            break
        chunks.append(data)
        if len(data) < 65535:
            break
    return b"".join(chunks).decode("utf-8", errors="replace")


def send_command(sock, payload, wait_sec=0.8):
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("ascii")
    sock.sendall(raw)
    time.sleep(wait_sec)
    text = recv_available(sock, wait_sec)
    return text, parse_json_stream(text)


def last_response(items, msg_id):
    for item in reversed(items):
        if isinstance(item, dict) and item.get("msg_id") == msg_id and "rval" in item:
            return item
    return None


def get_token(sock):
    text, items = send_command(sock, {"msg_id": 257, "token": 0}, wait_sec=1.0)
    response = last_response(items, 257)
    if not response or response.get("rval") != 0:
        raise RuntimeError(f"無法開啟 7878 控制 session，回應：{text or '(no response)'}")
    token = int(response.get("param", 0))
    if token <= 0:
        raise RuntimeError(f"7878 控制 session token 無效：{response}")
    return token


def get_setting(sock, token, name):
    _, items = send_command(sock, {"msg_id": 1, "type": name, "token": token}, wait_sec=0.8)
    response = last_response(items, 1)
    if response and response.get("rval") == 0:
        return response.get("param")
    return None


def get_all_settings(sock, token):
    _, items = send_command(sock, {"msg_id": 3, "token": token}, wait_sec=1.8)
    response = last_response(items, 3)
    if not response or response.get("rval") != 0:
        return {}
    result = {}
    for entry in response.get("param", []):
        if isinstance(entry, dict):
            result.update(entry)
    return result


def battery_percent(value):
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def rval_summary(response):
    if not response:
        return "no response"
    return f"rval={response.get('rval')} msg_id={response.get('msg_id')}"


def start_viewfinder(host, port, timeout_sec):
    with socket.create_connection((host, port), timeout=timeout_sec) as sock:
        token = get_token(sock)
        log(f"7878 control session opened, token={token}")

        _, device_items = send_command(sock, {"msg_id": 11, "token": token}, wait_sec=0.9)
        device = last_response(device_items, 11)
        if device and device.get("rval") == 0:
            model = device.get("model", "unknown")
            chip = device.get("chip", "unknown")
            api_ver = device.get("api_ver", "unknown")
            log(f"camera API: model={model}, chip={chip}, api={api_ver}")

        settings = get_all_settings(sock, token)
        app_status = get_setting(sock, token, "app_status") or settings.get("app_status")
        camera_mode = get_setting(sock, token, "camera_mode") or settings.get("camera_mode")
        battery = get_setting(sock, token, "Battery_level") or settings.get("Battery_level")
        firmware = settings.get("sw_version")

        if firmware:
            log(f"firmware={firmware}")
        if app_status:
            log(f"app_status={app_status}")
        if camera_mode:
            log(f"camera_mode={camera_mode}")
        if battery:
            log(f"Battery_level={battery}")
            percent = battery_percent(battery)
            if percent is not None and percent <= 10:
                log("WARNING: camera battery looks very low; video pipeline may refuse to start.")

        reset_param = os.environ.get("G3P_RESETVF_PARAM", "start")
        stop_first = os.environ.get("G3P_RESETVF_STOP_FIRST", "1") == "1" and reset_param == "start"
        commands = [
            ("set camera_mode normal_record", {"msg_id": 2, "type": "camera_mode", "param": "normal_record", "token": token}, 2.0),
        ]
        if os.environ.get("G3P_SEND_STREAM_OUT_TYPE", "0") == "1":
            commands.append(
                ("set stream_out_type rtsp", {"msg_id": 2, "type": "stream_out_type", "param": "rtsp", "token": token}, 1.0)
            )
        if stop_first:
            commands.append(
                ("BOSS_RESETVF stop", {"msg_id": 259, "param": "stop", "token": token}, 2.0)
            )
        commands.append(("BOSS_RESETVF", {"msg_id": 259, "param": reset_param, "token": token}, 3.0))

        reset_response = None
        for label, payload, wait_sec in commands:
            text, items = send_command(sock, payload, wait_sec=wait_sec)
            response = last_response(items, payload["msg_id"])
            if payload["msg_id"] == 259:
                reset_response = response
            if response:
                log(f"{label}: {rval_summary(response)}")
            elif text:
                log(f"{label}: non-standard response: {text}")
            else:
                log(f"{label}: no response")

        if reset_response and reset_response.get("rval") == 0:
            log("RTSP viewfinder start command accepted.")
            return 0

        if reset_param == "start":
            log("BOSS_RESETVF param=start was sent; camera may restart the RTSP encoder even without rval=0.")
            return 0

        if reset_response and reset_response.get("rval") == -21:
            log(
                "WARNING: camera rejected RTSP viewfinder start with rval=-21. "
                "Common causes: low battery, camera menu/busy state, or sensor/video pipeline not ready."
            )
            return 21

        log(f"WARNING: RTSP viewfinder start was not confirmed: {rval_summary(reset_response)}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Start G3P/Ambarella RTSP viewfinder over TCP 7878.")
    parser.add_argument("--host", default="192.168.144.135")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--soft-fail",
        action="store_true",
        help="Return 0 even if the camera rejects the viewfinder start command.",
    )
    args = parser.parse_args()

    try:
        result = start_viewfinder(args.host, args.port, args.timeout)
    except OSError as exc:
        log(f"control port {args.host}:{args.port} is not reachable: {exc}")
        return 0 if args.soft_fail else 1
    except Exception as exc:
        log(f"control failed: {exc}")
        return 0 if args.soft_fail else 1

    if args.soft_fail:
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
