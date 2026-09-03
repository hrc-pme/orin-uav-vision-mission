# Xin UAV — Jetson Orin NX Vision and Mission Control

Onboard computer-vision software for a UAV running on an NVIDIA Jetson Orin NX. The repository contains two operational workflows:

1. a RealSense D455 vehicle-approach bridge that publishes camera, detection, and flight-controller data to a Unity dashboard; and
2. the Presidential Cup competition mission, which detects a white vehicle with a red roof cross from an RTSP camera and can command ArduPilot RTL.

> [!CAUTION]
> This software can send real flight-control commands. Test with propellers removed, use the indoor-safe mode first, keep a pilot ready to take over, and verify every camera, GPS, battery, MAVLink, altitude, and model setting before flight. `AUTO_RTL=1` enables an actual RTL command.

## Contents

- [System overview](#system-overview)
- [Tested platform](#tested-platform)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Model files](#model-files)
- [Unity D455 vehicle-approach workflow](#unity-d455-vehicle-approach-workflow)
- [Presidential Cup workflow](#presidential-cup-workflow)
- [Outputs and ROS 2 topics](#outputs-and-ros-2-topics)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Safety and operational limits](#safety-and-operational-limits)
- [Documentation and license](#documentation-and-license)

## System overview

```text
Unity vehicle-approach workflow

Intel RealSense D455 ─┐
                      ├─> Jetson Orin NX ── ROS 2 / rosbridge ──> Unity dashboard
AP6 flight controller ┘         │
                                └─> rosbag2 + CSV + JPEG recording

Presidential Cup workflow

G3P RTSP camera ──────┐
                      ├─> Jetson Orin NX ──> target event / preview / rosbag2
AP6 flight controller ┘         │
                                └─> optional ArduPilot RTL command
```

The Unity workflow uses RGB detection and projects image detections onto the ground using camera intrinsics, camera mounting geometry, and flight-controller attitude. D455 depth is intentionally disabled at the mission altitudes configured in this project.

The competition workflow runs a vehicle detector first, then searches within vehicle regions for the white-vehicle/red-cross mission marker. It records mission evidence and supports both a non-RTL observation mode and a latched automatic-RTL mode.

## Tested platform

The current deployment was verified on:

| Component | Configuration |
| --- | --- |
| Onboard computer | NVIDIA Jetson Orin NX Engineering Reference Developer Kit |
| Architecture | aarch64 |
| Operating system | Ubuntu 22.04.5 LTS |
| Jetson Linux | R36.5.0 |
| Python | 3.10 |
| ROS | ROS 2 Humble |
| Unity camera | Intel RealSense D455, serial `234322304658` |
| Competition camera | G3P RTSP camera, default `rtsp://192.168.144.135/live` |
| Flight controller | AP6/ArduPilot over MAVLink |

ROS 2 Humble targets Ubuntu 22.04. For Jetson-specific RealSense installation and kernel guidance, use the upstream [ROS 2 Humble documentation](https://docs.ros.org/en/humble/Installation.html) and [librealsense Jetson installation guide](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md).

## Repository layout

```text
xin_uav/
├── README.md
├── .gitignore
├── install_torch_jetson.sh
├── orin/
│   ├── requirements.txt
│   ├── orin_d455i_precision_tracker.py       # shared camera/MAVLink/ROS helpers
│   ├── red_white/
│   │   └── detect_red_white_x.py             # red/white marker detector
│   └── real-time_bag_rtl/
│       └── real_time_new_camera_orin/
│           ├── red_white_target_server_new_camera.py
│           ├── run_white_car_red_cross_new_camera_orin.sh
│           ├── setup_g3p_network.sh
│           └── start_g3p_viewfinder.py
└── unity/
    ├── config_d455_car_approach.yaml
    ├── d455_car_approach_bridge.py
    ├── unity_dashboard_bridge.py
    ├── preflight_d455_car_approach.py
    ├── run_d455_car_approach.sh
    ├── record_d455_car_approach.sh
    ├── record_experiment.sh
    ├── experiment_csv_logger.py
    ├── run_d455_car_approach_indoor.sh
    ├── run_d455_map_preview_indoor.sh
    ├── D455_CAR_APPROACH_RUNBOOK.md
    └── THESIS_SYSTEM_DESIGN.md
```

The competition server imports `orin/orin_d455i_precision_tracker.py` and `orin/red_white/detect_red_white_x.py`; these files are shared dependencies and must remain in their current relative locations.

### Local archive

Historical source, backups, model weights, test media, and flight recordings were preserved outside this GitHub-ready tree at `/home/hrc/uav`. The archive keeps the original paths relative to `xin_uav` (for example, the old `unity/recordings` directory is stored as `/home/hrc/uav/unity/recordings`). It is not required by the active launchers and should not be included in a normal GitHub upload.

## Installation

### 1. Platform software

Install a JetPack release compatible with the Orin NX, then install ROS 2 Humble and the rosbridge/rosbag packages required by the launchers. The scripts expect ROS at:

```text
/opt/ros/humble/setup.bash
```

The host also needs `ffmpeg`/`ffprobe`, `iproute2`, `util-linux`, a CUDA-enabled PyTorch build for Jetson, and working NVIDIA drivers. Do not replace a working Jetson PyTorch installation with a generic x86 or CPU-only wheel.

Useful checks:

```bash
source /opt/ros/humble/setup.bash
python3 --version
ros2 --help >/dev/null
ffprobe -version
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`install_torch_jetson.sh` is provided as a deployment helper, but review its JetPack/CUDA assumptions before running it on a different image.

### 2. Python environment

ROS Python packages are installed by the operating system, so a virtual environment must be able to see system packages:

```bash
cd /home/hrc/xin_uav
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r orin/requirements.txt
python3 -m pip install -r unity/requirements.txt
```

On Jetson, `pyrealsense2` may need to come from the JetPack-compatible librealsense build rather than PyPI. Verify the camera before continuing:

```bash
rs-enumerate-devices
python3 -c 'import pyrealsense2 as rs; print(rs.context().devices)'
```

To install the repository's D455 udev rule:

```bash
cd /home/hrc/xin_uav/unity
./install_realsense_permissions.sh
```

Reconnect the camera after installing a new rule.

### 3. Hardware access

Confirm the expected devices and network route:

```bash
ls -l /dev/ttyACM0 /dev/serial/by-id/ 2>/dev/null
ip -4 address
ip route get 192.168.144.135
ffprobe -rtsp_transport tcp -v error \
  -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0:s=x rtsp://192.168.144.135/live
```

The competition launcher expects a 1920×1080 stream when `REQUIRE_1080P=1`.

## Model files

Model weights and TensorRT engines are deliberately excluded from Git because they are large, machine-specific artifacts. Supply the trained VisDrone model separately.

Default locations used by both workflows:

```text
/home/hrc/Downloads/VisDrone_train1_5090best.pt
/home/hrc/Downloads/VisDrone_train1_5090best.engine
```

For the Unity workflow, edit these keys in `unity/config_d455_car_approach.yaml` if the files are elsewhere:

```yaml
engine_path: /absolute/path/to/model.engine
fallback_model_path: /absolute/path/to/model.pt
```

For the competition workflow, override the path without editing source:

```bash
VEHICLE_MODEL=/absolute/path/to/model.engine \
./run_white_car_red_cross_new_camera_orin.sh
```

TensorRT engines are normally tied to the TensorRT/CUDA/GPU environment that built them. Rebuild the engine on the target Orin NX when the software stack or model changes.

## Unity D455 vehicle-approach workflow

### Network and Unity

The default field setup uses:

| Device | Address | Purpose |
| --- | --- | --- |
| Unity computer | `10.0.0.7` | Operator dashboard |
| Jetson Orin NX | `10.0.0.8` | Perception, telemetry, and command bridge |
| rosbridge | `ws://10.0.0.8:9090` | Unity-to-ROS WebSocket |

The Unity project itself is not included in this repository. The preflight script expects its source checkout at `/home/hrc/unity_windows_review/UAVDashboard`. If it is stored elsewhere, update the `UNITY_*` constants near the top of `unity/preflight_d455_car_approach.py`.

### Required preflight

From the Unity workflow directory:

```bash
cd /home/hrc/xin_uav/unity
./preflight_d455_car_approach.py
```

The preflight is read-only with respect to vehicle control. It validates the mission configuration, required files, D455 access, flight-controller heartbeat, GPS quality, and battery state. Do not bypass a failed outdoor preflight.

### Start the live bridge

```bash
cd /home/hrc/xin_uav/unity
./run_d455_car_approach.sh
```

Operational behavior configured by the current launcher and YAML:

- the program detects and locks one `car` target at 80–130 m relative to HOME;
- before Unity `CONFIRM`, it performs detection/locking only and does not move automatically;
- after `CONFIRM`, it freezes the accepted GPS position and heading;
- it descends toward 20 m using the configured fixed-position safety path;
- it never arms or takes off automatically; and
- it requires the vehicle to remain within the configured image-center gate when confirmed.

Review every value in `unity/config_d455_car_approach.yaml` before flight. In particular, verify camera mounting angles, MAVLink port, battery limits, permitted modes, altitude limits, speed limits, and distance gates.

### Indoor-safe validation

Use this mode with propellers removed before any outdoor test:

```bash
cd /home/hrc/xin_uav/unity
./run_d455_car_approach_indoor.sh
```

It keeps the D455, TensorRT inference, Unity image path, and telemetry available while forcing automatic verification and MAVLink flight commands off.

For a Unity map preview with fake WGS84 data and flight commands still disabled:

```bash
./run_d455_map_preview_indoor.sh
```

### Record a run

Run the recorder in a second terminal while the live bridge is active:

```bash
cd /home/hrc/xin_uav/unity
./record_d455_car_approach.sh
```

Recordings are written below `unity/recordings/` and are ignored by Git.

## Presidential Cup workflow

Working directory:

```bash
cd /home/hrc/xin_uav/orin/real-time_bag_rtl/real_time_new_camera_orin
```

### Observation and tuning mode — no automatic RTL

This mode reports repeated target updates and does not command RTL:

```bash
VEHICLE_CONF=0.60 \
ROOF_RED_VEHICLE_CONF=0.60 \
MIN_WHITE_VEHICLE_RATIO=0.30 \
ROOF_RED_MIN_RATIO=0.05 \
MISSION_CONFIRM_HITS=3 \
AUTO_RTL=0 \
MISSION_LATCH_ONCE=0 \
EMIT_UPDATES=1 \
./run_white_car_red_cross_new_camera_orin.sh
```

### Competition mode — automatic RTL enabled

> [!WARNING]
> The following command enables automatic RTL after a confirmed mission target. Use it only during an authorized flight with a valid home position and verified pilot takeover.

```bash
VEHICLE_CONF=0.60 \
ROOF_RED_VEHICLE_CONF=0.60 \
MIN_WHITE_VEHICLE_RATIO=0.30 \
ROOF_RED_MIN_RATIO=0.015 \
MISSION_CONFIRM_HITS=3 \
AUTO_RTL=1 \
MISSION_LATCH_ONCE=1 \
EMIT_UPDATES=0 \
./run_white_car_red_cross_new_camera_orin.sh
```

The competition launcher defaults to:

- RTSP camera `192.168.144.135`, with the Orin camera interface configured as `192.168.144.10/24`;
- TCP port `5001` for the target server;
- CUDA inference on device `0`;
- 1920×1080 camera preflight and a 1280-pixel detector input;
- MAVLink through the configured `/dev/serial/by-id/...` AP6 device;
- ROS 2 bag recording under `/home/hrc/recordings/red_white_target_bag_rtl`; and
- strict white-vehicle plus red-roof-marker confirmation (white-only fallback is forced off).

To test software paths without a real camera, flight controller, model, bag, or RTL command:

```bash
FAKE_CAMERA=1 \
MOCK_FLIGHT=1 \
DISABLE_VEHICLE_MODEL=1 \
AUTO_RTL=0 \
RECORD_ROSBAG=0 \
G3P_SETUP_NETWORK=0 \
G3P_STRICT_PREFLIGHT=0 \
./run_white_car_red_cross_new_camera_orin.sh
```

## Outputs and ROS 2 topics

### Unity recorder

Each Unity recording session contains a rosbag2 database, a manifest, CSV summaries, compressed camera images, candidate crops, detections, commands, telemetry, GPS, IMU, battery data, and parsed MAVLink JSON.

Important topics include:

```text
/d455i/color/image_annotated/compressed
/d455i/color/image_raw/compressed
/unity/detections_json
/unity/target_command
/unity/target_command_status
/unity/approach_state
/uav/gps
/uav/imu
/uav/velocity_ned
/uav/battery
/uav/telemetry_json
/uav/mavlink_json
```

### Competition recorder

Each competition session contains `mission_summary.csv`, `events.jsonl`, and an optional compressed rosbag. Important topics include:

```text
/g3p/color/image_raw/compressed
/g3p/color/image_annotated/compressed
/uav/gps/fix
/uav/imu
/uav/velocity_ned
/uav/battery
/uav/flight_state
/uav/car_detections
/uav/target/fix
/uav/target/geolocation
```

Inspect a completed bag with:

```bash
ros2 bag info /path/to/session/rosbag
```

## Configuration reference

Common competition environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAMERA_SOURCE` | `rtsp` | `rtsp`, `d455`, `v4l2`, or `gstreamer` |
| `RTSP_URL` | `rtsp://192.168.144.135/live` | Competition camera stream |
| `G3P_IFACE` | auto-detected | Ethernet interface connected to the camera |
| `VEHICLE_MODEL` | engine, then PT fallback | Detector model path |
| `VEHICLE_CONF` | `0.60` | Main vehicle confidence threshold |
| `MIN_WHITE_VEHICLE_RATIO` | `0.30` | Minimum white-pixel ratio inside a vehicle ROI |
| `ROOF_RED_MIN_RATIO` | `0.015` | Minimum red-pixel ratio in the roof search region |
| `MISSION_CONFIRM_HITS` | `3` | Confirmations required before mission activation |
| `AUTO_RTL` | `1` | Enable or disable the real RTL command |
| `MISSION_LATCH_ONCE` | `1` | Trigger at most once per process |
| `EMIT_UPDATES` | `0` | Emit continuous updates instead of first sighting only |
| `RECORD_ROSBAG` | `1` | Enable official rosbag2 recording |
| `RECORD_ROOT` | `/home/hrc/recordings/red_white_target_bag_rtl` | Output root |
| `MAVLINK` | AP6 `/dev/serial/by-id/...` path | Flight-controller serial device |
| `BAUD` | `115200` | MAVLink serial rate |

Environment assignments must appear immediately before the launcher command, as shown in the examples. Use `Ctrl+C` for a clean shutdown so rosbag metadata can be finalized.

## Troubleshooting

### CUDA preflight fails

Verify that the PyTorch build matches the installed JetPack/CUDA stack:

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

### Model not found

Place the PT/engine file in the default location, set `VEHICLE_MODEL` for the competition launcher, or update the two model paths in the Unity YAML. Model files are intentionally not committed.

### D455 is not detected or reports permission errors

Use a USB 3 connection, run `rs-enumerate-devices`, install the udev rule, reconnect the camera, and follow the upstream Jetson-specific librealsense guide. Do not install multiple conflicting librealsense builds.

### Unity cannot connect

Check that rosbridge is listening on TCP 9090 and that the Unity computer can reach the Orin:

```bash
ss -ltnp 'sport = :9090'
ping -c 3 10.0.0.7
```

Unity should connect to `ws://10.0.0.8:9090`, and both systems must use the expected ROS domain (`ROS_DOMAIN_ID=42` in the launchers).

### Competition RTSP preflight fails

Check the physical Ethernet link, selected interface, route, and stream resolution. Override `G3P_IFACE` if auto-detection chooses the wrong interface:

```bash
G3P_IFACE=eno1 ./setup_g3p_network.sh
```

### TCP port 5001 is already in use

Stop the stale process or select another port:

```bash
ss -ltnp 'sport = :5001'
PORT=5002 ./run_white_car_red_cross_new_camera_orin.sh
```

### MAVLink heartbeat, GPS, or battery preflight fails

Do not fly. Confirm the serial device, baud rate, wiring, shared ground, ArduPilot serial protocol, GPS fix, home position, battery telemetry, and that no other process owns the serial port.

## Safety and operational limits

- This repository is research/competition software, not a certified flight-control system.
- The pilot and applicable competition rules remain authoritative.
- Never test automatic commands indoors with propellers installed.
- Confirm the camera is mounted at the angles configured in software; a wrong extrinsic calibration produces wrong ground coordinates.
- Verify the ArduPilot HOME position and RTL behavior independently before enabling `AUTO_RTL=1`.
- Maintain a manual RC takeover path and a safe geofence.
- Archive the exact config, model checksum, rosbag, flight-controller log, weather, and test conditions for every flight.
- A clean software preflight does not prove that the airframe, propulsion, navigation, or RF links are safe.

## Documentation and license

Additional project notes:

- [`unity/D455_CAR_APPROACH_RUNBOOK.md`](unity/D455_CAR_APPROACH_RUNBOOK.md) — field checklist and detailed operating procedure (Chinese)
- [`unity/THESIS_SYSTEM_DESIGN.md`](unity/THESIS_SYSTEM_DESIGN.md) — system-design and validation notes (Chinese)

No open-source license has been selected yet. Without a license, copyright remains with the project owner and reuse rights are not granted automatically. Add a `LICENSE` file before publishing if you intend others to copy, modify, or redistribute the code.
