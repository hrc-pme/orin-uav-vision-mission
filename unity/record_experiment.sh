#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
cd /home/hrc/xin_uav/unity

exec 9>/tmp/unity_dashboard_record.lock
if ! flock -n 9; then
  echo "An experiment recorder is already running; stop it with Ctrl+C first." >&2
  exit 1
fi

mkdir -p recordings
STAMP="$(date +%Y%m%d_%H%M%S)"
RECORD_PREFIX="${RECORD_PREFIX:-uav_dashboard}"
RECORD_TARGET_DESCRIPTION="${RECORD_TARGET_DESCRIPTION:-white-car candidates}"
OUT_DIR="recordings/${RECORD_PREFIX}_${STAMP}"
EXTRA_TOPICS=()
if [[ -n "${RECORD_EXTRA_TOPICS:-}" ]]; then
  read -r -a EXTRA_TOPICS <<< "$RECORD_EXTRA_TOPICS"
fi
mkdir -p "$OUT_DIR"

cleanup() {
  if [[ -n "${CSV_PID:-}" ]]; then
    kill "$CSV_PID" 2>/dev/null || true
    wait "$CSV_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Recording experiment to: $OUT_DIR"
echo "  rosbag2: $OUT_DIR/rosbag2"
echo "  csv:     $OUT_DIR/*.csv"
echo "  images:  $OUT_DIR/compressed_images/raw/*.jpg"
echo "           $OUT_DIR/compressed_images/annotated/*.jpg"
echo "  crops:   $OUT_DIR/candidate_images/*.jpg"
echo "  mavlink: $OUT_DIR/mavlink.csv and /uav/mavlink_json in rosbag2"
cat > "$OUT_DIR/manifest.txt" <<EOF
Unity UAV experiment recording
Started: $(date --iso-8601=seconds)
ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-unset}

Main replay bag:
  $OUT_DIR/rosbag2

CSV summary files:
  images.csv              raw/annotated compressed RGB timestamp + saved jpg path
  detections.csv          ${RECORD_TARGET_DESCRIPTION}, WGS84/TWD97/error, crop path
  debug_detections.csv    every model car detection + white-score gate fields
  commands.csv            Unity CONFIRM/CANCEL command received on ROS
  status.csv              Orin NX command status RECEIVED/ACCEPTED/EXECUTING/ARRIVED/REJECTED
  telemetry.csv           MAVLink/GPS/mode/armed/height/battery/readiness
  mavlink.csv             every parsed MAVLink message received from AP6 as JSON
  gps.csv                 /uav/gps NavSatFix samples
  imu.csv                 /uav/imu IMU samples
  summary_events.csv      compact event timeline

Recorded important topics:
  /d455i/color/image_annotated/compressed
  /d455i/color/image_raw/compressed
  /d455i/color/camera_info
  /d455i/depth/image_rect_raw
  /d455i/imu
  /unity/detections_json
  /unity/target_command
  /unity/target_command_status
  ${RECORD_EXTRA_TOPICS:-}
  /uav/gps
  /uav/imu
  /uav/velocity_ned
  /uav/battery
  /uav/telemetry_json
  /uav/mavlink_json
  /tf
  /tf_static
  /rosout
  /parameter_events
EOF

python3 experiment_csv_logger.py --out-dir "$OUT_DIR" &
CSV_PID=$!

ros2 bag record \
  --include-unpublished-topics \
  -o "$OUT_DIR/rosbag2" \
  /d455i/color/image_annotated/compressed \
  /d455i/color/image_raw/compressed \
  /d455i/color/camera_info \
  /d455i/depth/image_rect_raw \
  /d455i/imu \
  /unity/detections_json \
  /unity/target_command \
  /unity/target_command_status \
  /uav/gps \
  /uav/imu \
  /uav/velocity_ned \
  /uav/battery \
  /uav/telemetry_json \
  /uav/mavlink_json \
  /tf \
  /tf_static \
  /rosout \
  /parameter_events \
  "${EXTRA_TOPICS[@]}"
