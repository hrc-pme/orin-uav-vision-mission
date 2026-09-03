#!/usr/bin/env bash
# Red-white mission target detector for the replacement camera, with official
# ROS 2 bag recording and optional auto RTL.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${PYTHON:-python3}"
FAST_PASS="${FAST_PASS:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5001}"
CAMERA_SERIAL="${CAMERA_SERIAL:-234322304658}"
CAMERA_SOURCE="${CAMERA_SOURCE:-rtsp}"      # d455 | v4l2 | gstreamer | rtsp
RTSP_URL="${RTSP_URL:-rtsp://192.168.144.135/live}"
G3P_CAMERA_IP="${G3P_CAMERA_IP:-192.168.144.135}"
G3P_HOST_IP="${G3P_HOST_IP:-192.168.144.10}"
G3P_IFACE="${G3P_IFACE:-}"
G3P_SETUP_NETWORK="${G3P_SETUP_NETWORK:-1}"
G3P_START_VIEWFINDER="${G3P_START_VIEWFINDER:-1}"
G3P_PROBE_ATTEMPTS="${G3P_PROBE_ATTEMPTS:-8}"
G3P_PROBE_SLEEP_S="${G3P_PROBE_SLEEP_S:-3}"
G3P_PROBE_TIMEOUT_S="${G3P_PROBE_TIMEOUT_S:-8}"
G3P_VIEWFINDER_RECOVERY_ATTEMPTS="${G3P_VIEWFINDER_RECOVERY_ATTEMPTS:-3}"
G3P_VIEWFINDER_SETTLE_S="${G3P_VIEWFINDER_SETTLE_S:-12}"
G3P_STRICT_PREFLIGHT="${G3P_STRICT_PREFLIGHT:-1}"
REQUIRE_1080P="${REQUIRE_1080P:-1}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}" # V4L2 device or numeric OpenCV index
SENSOR_ID="${SENSOR_ID:-0}"                 # CSI/Argus sensor id for default GStreamer pipeline
CAMERA_PIPELINE="${CAMERA_PIPELINE:-}"      # optional when CAMERA_SOURCE=gstreamer
CAMERA_FX="${CAMERA_FX:-0.0}"
CAMERA_FY="${CAMERA_FY:-0.0}"
CAMERA_PPX="${CAMERA_PPX:-0.0}"
CAMERA_PPY="${CAMERA_PPY:-0.0}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-10}"
# Competition altitude is about 60 m; use RGB-only detection and ground-ray geolocation by default.
COLOR_ONLY="${COLOR_ONLY:-1}"
STRICT_NO_DEPTH_NOTE="depth disabled by COLOR_ONLY=1 for 80m low-latency race mode"
CAMERA_ROLL_DEG="${CAMERA_ROLL_DEG:-0.0}"
CAMERA_PITCH_DEG="${CAMERA_PITCH_DEG:--90.0}"
CAMERA_YAW_DEG="${CAMERA_YAW_DEG:-0.0}"
CAMERA_X_M="${CAMERA_X_M:-0.0}"
CAMERA_Y_M="${CAMERA_Y_M:-0.0}"
CAMERA_Z_M="${CAMERA_Z_M:-0.0}"
MAVLINK="${MAVLINK:-/dev/serial/by-id/usb-ALIGN_AP6-M490-ds_330032000D51333331333534-if00}"
BAUD="${BAUD:-115200}"
RECORD_ROOT="${RECORD_ROOT:-/home/hrc/recordings/red_white_target_bag_rtl}"
VEHICLE_PT_MODEL="${VEHICLE_PT_MODEL:-/home/hrc/Downloads/VisDrone_train1_5090best.pt}"
VEHICLE_ENGINE_MODEL="${VEHICLE_ENGINE_MODEL:-/home/hrc/Downloads/VisDrone_train1_5090best.engine}"
if [[ -z "${VEHICLE_MODEL:-}" ]]; then
  if [[ -f "$VEHICLE_ENGINE_MODEL" ]]; then
    VEHICLE_MODEL="$VEHICLE_ENGINE_MODEL"
  else
    VEHICLE_MODEL="$VEHICLE_PT_MODEL"
  fi
fi
VEHICLE_CLASSES="${VEHICLE_CLASSES:-car,van,truck,bus}"
VEHICLE_CONF=${VEHICLE_CONF:-0.60}
VEHICLE_IMGSZ="${VEHICLE_IMGSZ:-1280}"
# TensorRT engine is fixed-shape 1280x1280. Smaller VEHICLE_IMGSZ will crash inference.
if [[ $VEHICLE_MODEL == *.engine ]] && [[ $VEHICLE_IMGSZ != 1280 ]]; then
  echo "[warn] TensorRT engine requires VEHICLE_IMGSZ=1280; overriding VEHICLE_IMGSZ=${VEHICLE_IMGSZ} -> 1280" >&2
  VEHICLE_IMGSZ=1280
fi
VEHICLE_DEVICE="${VEHICLE_DEVICE:-0}"
VEHICLE_TILES="${VEHICLE_TILES:-1}"
VEHICLE_TILE_OVERLAP="${VEHICLE_TILE_OVERLAP:-0.25}"
DRAW_VEHICLE_CANDIDATES="${DRAW_VEHICLE_CANDIDATES:-0}"
DRAW_REJECTED_MARKER_CANDIDATES="${DRAW_REJECTED_MARKER_CANDIDATES:-0}"
DRAW_RAW_VEHICLE_CANDIDATES="${DRAW_RAW_VEHICLE_CANDIDATES:-1}"
DEBUG_VEHICLE_CONF=${DEBUG_VEHICLE_CONF:-0.40}
DEBUG_MIN_VEHICLE_WHITE_RATIO="${DEBUG_MIN_VEHICLE_WHITE_RATIO:-0.30}"
DEBUG_MIN_VEHICLE_AREA_RATIO="${DEBUG_MIN_VEHICLE_AREA_RATIO:-0.0002}"
DEBUG_MAX_VEHICLE_AREA_RATIO="${DEBUG_MAX_VEHICLE_AREA_RATIO:-0.25}"
VEHICLE_ROI_FIRST="${VEHICLE_ROI_FIRST:-1}"
MIN_WHITE_VEHICLE_RATIO="${MIN_WHITE_VEHICLE_RATIO:-0.30}"
WHITE_VEHICLE_FALLBACK="0"  # strict: never RTL on white vehicle alone
WHITE_VEHICLE_FALLBACK_CONF="${WHITE_VEHICLE_FALLBACK_CONF:-0.80}"
WHITE_VEHICLE_FALLBACK_WHITE_RATIO="${WHITE_VEHICLE_FALLBACK_WHITE_RATIO:-0.55}"
WHITE_VEHICLE_FALLBACK_MIN_AREA_RATIO="${WHITE_VEHICLE_FALLBACK_MIN_AREA_RATIO:-0.0005}"
WHITE_VEHICLE_FALLBACK_MAX_AREA_RATIO="${WHITE_VEHICLE_FALLBACK_MAX_AREA_RATIO:-0.08}"
ROOF_RED_FALLBACK="${ROOF_RED_FALLBACK:-1}"
ROOF_RED_VEHICLE_CONF="${ROOF_RED_VEHICLE_CONF:-0.60}"
ROOF_RED_ROI_SCALE="${ROOF_RED_ROI_SCALE:-1.80}"
ROOF_RED_MIN_RATIO="${ROOF_RED_MIN_RATIO:-0.015}"
ROOF_RED_MIN_PIXELS="${ROOF_RED_MIN_PIXELS:-8}"
ROOF_RED_H1_MAX="${ROOF_RED_H1_MAX:-18}"
ROOF_RED_H2_MIN="${ROOF_RED_H2_MIN:-155}"
ROOF_RED_S_MIN="${ROOF_RED_S_MIN:-22}"
ROOF_RED_V_MIN="${ROOF_RED_V_MIN:-35}"
MIN_MARKER_SCORE="${MIN_MARKER_SCORE:-0.10}"
MIN_X_SCORE="${MIN_X_SCORE:-0.10}"
MIN_RED_RATIO="${MIN_RED_RATIO:-$ROOF_RED_MIN_RATIO}"
MIN_X_WHITE_RATIO="${MIN_X_WHITE_RATIO:-0.03}"
MIN_MARKER_VEHICLE_AREA_RATIO="${MIN_MARKER_VEHICLE_AREA_RATIO:-0.025}"
MAX_MARKER_VEHICLE_AREA_RATIO="${MAX_MARKER_VEHICLE_AREA_RATIO:-0.45}"
MIN_ROI_MARKER_AREA="${MIN_ROI_MARKER_AREA:-20}"
MARKER_ROI_UPSCALE="${MARKER_ROI_UPSCALE:-4.5}"
VEHICLE_ROI_EXPAND="${VEHICLE_ROI_EXPAND:-1.20}"
MAX_MARKERS_PER_VEHICLE="${MAX_MARKERS_PER_VEHICLE:-4}"
CONFIRM_HITS="${CONFIRM_HITS:-1}"
MISSION_CONFIRM_HITS="${MISSION_CONFIRM_HITS:-3}"
REQUIRE_VEHICLE_MODEL_MATCH="${REQUIRE_VEHICLE_MODEL_MATCH:-1}"
MIN_GPS_FIX="${MIN_GPS_FIX:-6}"
TRACK_TTL_S="${TRACK_TTL_S:-1.0}"
DISPLAY_HOLD_S="${DISPLAY_HOLD_S:-0.0}"
TRACK_MIN_IOU="${TRACK_MIN_IOU:-0.05}"
TRACK_GEO_BREAK_M="${TRACK_GEO_BREAK_M:-4.0}"
REID_MEMORY_S="${REID_MEMORY_S:-90.0}"
MISSION_LATCH_ONCE="${MISSION_LATCH_ONCE:-1}"

if [[ "$FAST_PASS" == "1" ]]; then
  FPS="${FPS:-10}"
  VEHICLE_TILES="${VEHICLE_TILES:-1}"
  VEHICLE_IMGSZ="${VEHICLE_IMGSZ:-1280}"
VEHICLE_CONF=${VEHICLE_CONF:-0.60}
  VEHICLE_DEVICE="${VEHICLE_DEVICE:-0}"
DEBUG_VEHICLE_CONF=${DEBUG_VEHICLE_CONF:-0.40}
  CONFIRM_HITS="${CONFIRM_HITS:-1}"
  MISSION_CONFIRM_HITS="${MISSION_CONFIRM_HITS:-3}"
  TRACK_TTL_S="${TRACK_TTL_S:-1.0}"
  MARKER_ROI_UPSCALE="${MARKER_ROI_UPSCALE:-4.5}"
fi

RECORD_ROSBAG="${RECORD_ROSBAG:-1}"
BAG_MAX_SIZE="${BAG_MAX_SIZE:-2000000000}"
BAG_MAX_DURATION="${BAG_MAX_DURATION:-0}"
BAG_COMPRESSION_FORMAT="${BAG_COMPRESSION_FORMAT:-zstd}"
BAG_COMPRESSION_MODE="${BAG_COMPRESSION_MODE:-file}"
ROS2_WARMUP="${ROS2_WARMUP:-3.0}"
ROS_IMAGE_FPS="${ROS_IMAGE_FPS:-2.0}"
ROS_JPEG_QUALITY="${ROS_JPEG_QUALITY:-80}"
JPEG_QUALITY="${JPEG_QUALITY:-30}"
GROUND_PREVIEW_SCALE="${GROUND_PREVIEW_SCALE:-0.25}"
GROUND_PREVIEW_FPS="${GROUND_PREVIEW_FPS:-2.0}"

AUTO_RTL="${AUTO_RTL:-1}"
AUTO_RTL_REQUIRE_VALID_TARGET="${AUTO_RTL_REQUIRE_VALID_TARGET:-0}"
RTL_MODE_PARAM1="${RTL_MODE_PARAM1:-1}"
RTL_CUSTOM_MODE="${RTL_CUSTOM_MODE:-6}"
RTL_COMMAND_REPEATS="${RTL_COMMAND_REPEATS:-30}"
RTL_COMMAND_INTERVAL_S="${RTL_COMMAND_INTERVAL_S:-0.50}"
RTL_CONFIRM_TIMEOUT_S="${RTL_CONFIRM_TIMEOUT_S:-15.0}"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[error] ROS 2 setup not found: $ROS_SETUP" >&2
  exit 1
fi
if [[ ! -f "$VEHICLE_MODEL" && "${DISABLE_VEHICLE_MODEL:-0}" != "1" ]]; then
  echo "[error] vehicle model not found: $VEHICLE_MODEL" >&2
  exit 1
fi

# ROS setup files may inspect optional unset AMENT_* variables.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[error] ros2 command unavailable after sourcing $ROS_SETUP" >&2
  exit 1
fi

echo "[preflight] checking TCP port $PORT..."
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "[error] TCP port $PORT is already in use. Stop the old server or set PORT=5002." >&2
  ss -ltnp "sport = :$PORT" >&2 || true
  exit 6
fi

mkdir -p "$RECORD_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="${SESSION_DIR:-${RECORD_ROOT}/${STAMP}}"
BAG_PATH="${BAG_PATH:-${SESSION_DIR}/rosbag}"
mkdir -p "$SESSION_DIR"

TOPICS=(
  /d455/color/image_annotated/compressed
  /g3p/color/image_raw/compressed
  /g3p/color/image_annotated/compressed
  /d455/imu
  /uav/gps/fix
  /uav/imu
  /uav/velocity_ned
  /uav/battery
  /uav/flight_state
  /uav/car_detections
  /uav/target/fix
  /uav/target/geolocation
)

APP_PID=""
BAG_PID=""
ROUTE_KEEPER_PID=""
CLEANED_UP=0

cleanup() {
  local status="${1:-0}"
  if (( CLEANED_UP )); then
    return
  fi
  CLEANED_UP=1
  trap - INT TERM EXIT

  if [[ -n "$ROUTE_KEEPER_PID" ]] && kill -0 "$ROUTE_KEEPER_PID" 2>/dev/null; then
    echo "[stop] G3P route keeper PID $ROUTE_KEEPER_PID"
    kill "$ROUTE_KEEPER_PID" 2>/dev/null || true
    wait "$ROUTE_KEEPER_PID" 2>/dev/null || true
  fi
  if [[ "$CAMERA_SOURCE" == "rtsp" && -x "$SCRIPT_DIR/start_g3p_viewfinder.py" ]]; then
    echo "[stop] G3P RTSP viewfinder"
    G3P_RESETVF_PARAM=stop G3P_RESETVF_STOP_FIRST=0 \
      "$PYTHON" "$SCRIPT_DIR/start_g3p_viewfinder.py" --host "$G3P_CAMERA_IP" --port 7878 --soft-fail 2>/dev/null || true
  fi
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    echo "[stop] target server PID $APP_PID"
    kill -INT "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "$BAG_PID" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
    echo "[stop] ros2 bag PID $BAG_PID (finalizing metadata/storage)"
    kill -INT "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
  fi

  if [[ "$RECORD_ROSBAG" == "1" && -n "$BAG_PID" ]]; then
    if [[ -f "$BAG_PATH/metadata.yaml" ]]; then
      echo "[info] rosbag summary:"
      ros2 bag info "$BAG_PATH" || true
    else
      echo "[warn] rosbag metadata not found: $BAG_PATH/metadata.yaml" >&2
      if (( status == 0 )); then status=5; fi
    fi
  fi

  echo "[done] session: $SESSION_DIR"
  echo "[done] CSV    : $SESSION_DIR/mission_summary.csv"
  echo "[done] events : $SESSION_DIR/events.jsonl"
  if [[ "$RECORD_ROSBAG" == "1" && -n "$BAG_PID" ]]; then
    echo "[done] rosbag : $BAG_PATH"
  fi
  exit "$status"
}

trap 'cleanup 130' INT TERM
trap 'cleanup $?' EXIT

start_g3p_route_keeper() {
  if [[ "$CAMERA_SOURCE" != "rtsp" || "$G3P_SETUP_NETWORK" != "1" || -n "$ROUTE_KEEPER_PID" ]]; then
    return
  fi
  if [[ -z "${G3P_IFACE:-}" ]] || ! ip link show "$G3P_IFACE" >/dev/null 2>&1; then
    echo "[g3p-route] route keeper skipped; camera interface is unknown" >&2
    return
  fi
  (
    while true; do
      current_dev="$(ip route get "$G3P_CAMERA_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
      current_src="$(ip route get "$G3P_CAMERA_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
      if [[ "$current_dev" != "$G3P_IFACE" || "$current_src" != "$G3P_HOST_IP" ]]; then
        echo "[g3p-route] restoring route to $G3P_CAMERA_IP via $G3P_IFACE src $G3P_HOST_IP (was dev=${current_dev:-?} src=${current_src:-?})" >&2
        if [[ "$(id -u)" -eq 0 ]]; then
          ip link set "$G3P_IFACE" up || true
          ip -4 addr show "$G3P_IFACE" | grep -q "${G3P_HOST_IP}/24" || ip addr add "${G3P_HOST_IP}/24" dev "$G3P_IFACE" || true
          ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" src "$G3P_HOST_IP" || \
            ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" || true
        else
          sudo ip link set "$G3P_IFACE" up || true
          ip -4 addr show "$G3P_IFACE" | grep -q "${G3P_HOST_IP}/24" || sudo ip addr add "${G3P_HOST_IP}/24" dev "$G3P_IFACE" || true
          sudo ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" src "$G3P_HOST_IP" || \
            sudo ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" || true
        fi
      fi
      sleep "${G3P_ROUTE_KEEPALIVE_S:-2}"
    done
  ) &
  ROUTE_KEEPER_PID=$!
  echo "[g3p-route] route keeper PID $ROUTE_KEEPER_PID dev=$G3P_IFACE src=$G3P_HOST_IP"
}

echo "============================================================"
echo " STRICT white-car + red-white-X target detector + ROS 2 bag + auto RTL"
echo " Session   : $SESSION_DIR"
echo " ROS bag   : $BAG_PATH"
echo " Bag split : ${BAG_MAX_SIZE} bytes / ${BAG_MAX_DURATION} sec"
echo " Image     : compressed annotated + original 1080P JPEG, ${ROS_IMAGE_FPS} fps, quality ${ROS_JPEG_QUALITY}"
echo " Camera    : source=${CAMERA_SOURCE}, video_device=${VIDEO_DEVICE}, color_only=${COLOR_ONLY} (depth disabled when 1)"
if [[ "$CAMERA_SOURCE" == "rtsp" ]]; then
  echo " RTSP      : ${RTSP_URL}"
fi
if [[ "$CAMERA_SOURCE" == "gstreamer" ]]; then
  echo " Pipeline  : ${CAMERA_PIPELINE}"
fi
echo " Geometry  : camera_rpy=${CAMERA_ROLL_DEG},${CAMERA_PITCH_DEG},${CAMERA_YAW_DEG} deg offset_frd=${CAMERA_X_M},${CAMERA_Y_M},${CAMERA_Z_M} m"
echo " MAVLink   : $MAVLINK @ $BAUD"
echo " Trigger   : vehicle_roi_first=${VEHICLE_ROI_FIRST}, confirm_hits=${MISSION_CONFIRM_HITS}, require_vehicle_model=${REQUIRE_VEHICLE_MODEL_MATCH}"
echo " Fast pass : ${FAST_PASS}"
echo " Vehicle   : model=${VEHICLE_MODEL}, conf=${VEHICLE_CONF}, imgsz=${VEHICLE_IMGSZ}, tiles=${VEHICLE_TILES}, white_ratio>=${MIN_WHITE_VEHICLE_RATIO}"
echo " Fallback  : white_vehicle=${WHITE_VEHICLE_FALLBACK} (strict script forces white-only fallback off), conf>=${WHITE_VEHICLE_FALLBACK_CONF}, white>=${WHITE_VEHICLE_FALLBACK_WHITE_RATIO}, area=${WHITE_VEHICLE_FALLBACK_MIN_AREA_RATIO}-${WHITE_VEHICLE_FALLBACK_MAX_AREA_RATIO} (marker logic kept)"
echo " Roof red  : enabled=${ROOF_RED_FALLBACK}, vehicle_conf>=${ROOF_RED_VEHICLE_CONF}, roi_scale=${ROOF_RED_ROI_SCALE}, red_ratio>=${ROOF_RED_MIN_RATIO}, red_pixels>=${ROOF_RED_MIN_PIXELS}, hsv=0-${ROOF_RED_H1_MAX}/${ROOF_RED_H2_MIN}-179 S>=${ROOF_RED_S_MIN} V>=${ROOF_RED_V_MIN}"
echo " Debug     : draw_target_candidates=${DRAW_VEHICLE_CANDIDATES}, draw_rejected_markers=${DRAW_REJECTED_MARKER_CANDIDATES}, draw_raw_vehicle_candidates=${DRAW_RAW_VEHICLE_CANDIDATES}, debug_vehicle_conf>=${DEBUG_VEHICLE_CONF}, white>=${DEBUG_MIN_VEHICLE_WHITE_RATIO}"
echo " Marker    : score>=${MIN_MARKER_SCORE}, x>=${MIN_X_SCORE}, red>=${MIN_RED_RATIO}, white_x>=${MIN_X_WHITE_RATIO}, roi_area>=${MIN_ROI_MARKER_AREA}, roi_upscale=${MARKER_ROI_UPSCALE}, roi_expand=${VEHICLE_ROI_EXPAND}"
echo " Auto RTL  : ${AUTO_RTL}  trigger_on_detection=1 rtl_requires_precise_target=${AUTO_RTL_REQUIRE_VALID_TARGET}"
echo " Geo gate  : precise_target_min_fix=${MIN_GPS_FIX}; strict precision mode requires valid target for RTL"
echo " RTL cmd   : MAV_CMD_DO_SET_MODE param1=${RTL_MODE_PARAM1} param2=${RTL_CUSTOM_MODE}, repeats=${RTL_COMMAND_REPEATS}, timeout=${RTL_CONFIRM_TIMEOUT_S}s"
echo "============================================================"

echo "[preflight] checking CUDA PyTorch..."
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch is unavailable")
x = torch.ones((64, 64), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
if not bool(torch.isfinite(y).all()):
    raise SystemExit("CUDA tensor test produced invalid values")
print(f"[preflight] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

start_rosbag() {
  if [[ "$RECORD_ROSBAG" != "1" || -n "$BAG_PID" ]]; then
    return
  fi
  bag_args=(
    ros2 bag record
    --storage sqlite3
    --output "$BAG_PATH"
    --max-bag-size "$BAG_MAX_SIZE"
    --max-bag-duration "$BAG_MAX_DURATION"
  )
  if [[ "$BAG_COMPRESSION_MODE" != "none" ]]; then
    bag_args+=(--compression-mode "$BAG_COMPRESSION_MODE")
    bag_args+=(--compression-format "$BAG_COMPRESSION_FORMAT")
  fi
  bag_args+=("${TOPICS[@]}")
  "${bag_args[@]}" &
  BAG_PID=$!
}

args=(
  "$PYTHON" "$SCRIPT_DIR/red_white_target_server_new_camera.py"
  --host "$HOST"
  --port "$PORT"
  --camera-source "$CAMERA_SOURCE"
  --camera-serial "$CAMERA_SERIAL"
  --video-device "$VIDEO_DEVICE"
  --rtsp-url "$RTSP_URL"
  --camera-fx "$CAMERA_FX"
  --camera-fy "$CAMERA_FY"
  --camera-ppx "$CAMERA_PPX"
  --camera-ppy "$CAMERA_PPY"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --mavlink "$MAVLINK"
  --baud "$BAUD"
  --session-dir "$SESSION_DIR"
  --camera-roll-deg "$CAMERA_ROLL_DEG"
  --camera-pitch-deg "$CAMERA_PITCH_DEG"
  --camera-yaw-deg "$CAMERA_YAW_DEG"
  --camera-x-m "$CAMERA_X_M"
  --camera-y-m "$CAMERA_Y_M"
  --camera-z-m "$CAMERA_Z_M"
  --vehicle-model "$VEHICLE_MODEL"
  --vehicle-classes "$VEHICLE_CLASSES"
  --vehicle-conf "$VEHICLE_CONF"
  --vehicle-imgsz "$VEHICLE_IMGSZ"
  --vehicle-device "$VEHICLE_DEVICE"
  --vehicle-tiles "$VEHICLE_TILES"
  --vehicle-tile-overlap "$VEHICLE_TILE_OVERLAP"
  --debug-vehicle-conf "$DEBUG_VEHICLE_CONF"
  --debug-min-vehicle-white-ratio "$DEBUG_MIN_VEHICLE_WHITE_RATIO"
  --debug-min-vehicle-area-ratio "$DEBUG_MIN_VEHICLE_AREA_RATIO"
  --debug-max-vehicle-area-ratio "$DEBUG_MAX_VEHICLE_AREA_RATIO"
  --min-marker-score "$MIN_MARKER_SCORE"
  --min-x-score "$MIN_X_SCORE"
  --min-red-ratio "$MIN_RED_RATIO"
  --min-x-white-ratio "$MIN_X_WHITE_RATIO"
  --min-white-vehicle-ratio "$MIN_WHITE_VEHICLE_RATIO"
  --white-vehicle-fallback-conf "$WHITE_VEHICLE_FALLBACK_CONF"
  --white-vehicle-fallback-white-ratio "$WHITE_VEHICLE_FALLBACK_WHITE_RATIO"
  --white-vehicle-fallback-min-area-ratio "$WHITE_VEHICLE_FALLBACK_MIN_AREA_RATIO"
  --white-vehicle-fallback-max-area-ratio "$WHITE_VEHICLE_FALLBACK_MAX_AREA_RATIO"
  --roof-red-vehicle-conf "$ROOF_RED_VEHICLE_CONF"
  --roof-red-roi-scale "$ROOF_RED_ROI_SCALE"
  --roof-red-min-ratio "$ROOF_RED_MIN_RATIO"
  --roof-red-min-pixels "$ROOF_RED_MIN_PIXELS"
  --roof-red-h1-max "$ROOF_RED_H1_MAX"
  --roof-red-h2-min "$ROOF_RED_H2_MIN"
  --roof-red-s-min "$ROOF_RED_S_MIN"
  --roof-red-v-min "$ROOF_RED_V_MIN"
  --min-marker-vehicle-area-ratio "$MIN_MARKER_VEHICLE_AREA_RATIO"
  --max-marker-vehicle-area-ratio "$MAX_MARKER_VEHICLE_AREA_RATIO"
  --min-roi-marker-area "$MIN_ROI_MARKER_AREA"
  --marker-roi-upscale "$MARKER_ROI_UPSCALE"
  --vehicle-roi-expand "$VEHICLE_ROI_EXPAND"
  --max-markers-per-vehicle "$MAX_MARKERS_PER_VEHICLE"
  --confirm-hits "$CONFIRM_HITS"
  --mission-confirm-hits "$MISSION_CONFIRM_HITS"
  --min-gps-fix "$MIN_GPS_FIX"
  --track-ttl-s "$TRACK_TTL_S"
  --display-hold-s "$DISPLAY_HOLD_S"
  --track-min-iou "$TRACK_MIN_IOU"
  --track-geo-break-m "$TRACK_GEO_BREAK_M"
  --reid-memory-s "$REID_MEMORY_S"
  --ros2
  --ros2-compressed-only
  --ros2-warmup "$ROS2_WARMUP"
  --ros-image-fps "$ROS_IMAGE_FPS"
  --ros-jpeg-quality "$ROS_JPEG_QUALITY"
  --ground-preview-scale "$GROUND_PREVIEW_SCALE"
  --ground-preview-fps "$GROUND_PREVIEW_FPS"
  --rtl-mode-param1 "$RTL_MODE_PARAM1"
  --rtl-custom-mode "$RTL_CUSTOM_MODE"
  --rtl-command-repeats "$RTL_COMMAND_REPEATS"
  --rtl-command-interval-s "$RTL_COMMAND_INTERVAL_S"
  --rtl-confirm-timeout-s "$RTL_CONFIRM_TIMEOUT_S"
)

if [[ "$CAMERA_SOURCE" == "rtsp" && "$G3P_SETUP_NETWORK" == "1" ]]; then
  iface_has_carrier() {
    [[ -e "/sys/class/net/$1/carrier" ]] && [[ "$(cat "/sys/class/net/$1/carrier" 2>/dev/null)" == "1" ]]
  }

  pick_g3p_iface() {
    if [[ -n "$G3P_IFACE" ]]; then
      echo "$G3P_IFACE"
      return 0
    fi
    if iface_has_carrier eno1; then
      echo eno1
      return 0
    fi
    for iface_path in /sys/class/net/enx*; do
      [[ -e "$iface_path" ]] || continue
      iface="${iface_path##*/}"
      if iface_has_carrier "$iface"; then
        echo "$iface"
        return 0
      fi
    done
    for iface_path in /sys/class/net/*; do
      iface="${iface_path##*/}"
      case "$iface" in
        lo|docker*|br-*|virbr*|veth*|tailscale*|wg*|wl*|wlan*|wifi*|p2p-*|usb0|usb1|usb2)
          continue
          ;;
      esac
      if [[ "$(cat "/sys/class/net/$iface/type" 2>/dev/null)" == "1" ]] && iface_has_carrier "$iface"; then
        echo "$iface"
        return 0
      fi
    done
    echo eno1
  }

  G3P_IFACE="$(pick_g3p_iface)"
  echo "[g3p] using camera interface: ${G3P_IFACE}"

  probe_rtsp_resolution() {
    timeout "$G3P_PROBE_TIMEOUT_S" ffprobe -rtsp_transport tcp -v error \
      -select_streams v:0 \
      -show_entries stream=width,height \
      -of csv=p=0:s=x "$RTSP_URL" 2>/dev/null | head -1 || true
  }

  wait_rtsp_resolution() {
    local attempt probe
    for ((attempt = 1; attempt <= G3P_PROBE_ATTEMPTS; attempt++)); do
      probe="$(probe_rtsp_resolution)"
      if [[ -n "$probe" ]]; then
        echo "$probe"
        return 0
      fi
      if (( attempt < G3P_PROBE_ATTEMPTS )); then
        echo "[g3p] RTSP probe unavailable; waiting ${G3P_PROBE_SLEEP_S}s (${attempt}/${G3P_PROBE_ATTEMPTS})" >&2
        sleep "$G3P_PROBE_SLEEP_S"
      fi
    done
    return 1
  }

  for iface_path in /sys/class/net/*; do
    other_iface="${iface_path##*/}"
    if [[ "$other_iface" != "$G3P_IFACE" ]] && ip -4 addr show "$other_iface" 2>/dev/null | grep -q "${G3P_HOST_IP}/24"; then
      echo "[g3p] removing stale ${G3P_HOST_IP}/24 from ${other_iface}"
      if [[ "$(id -u)" -eq 0 ]]; then
        ip addr del "${G3P_HOST_IP}/24" dev "$other_iface" || true
      else
        sudo ip addr del "${G3P_HOST_IP}/24" dev "$other_iface" || true
      fi
    fi
  done

  if ip link show "$G3P_IFACE" >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      ip link set "$G3P_IFACE" up || true
    else
      sudo ip link set "$G3P_IFACE" up || true
    fi
    if ! ip -4 addr show "$G3P_IFACE" | grep -q "${G3P_HOST_IP}/24"; then
      echo "[g3p] adding ${G3P_HOST_IP}/24 to ${G3P_IFACE}"
      if [[ "$(id -u)" -eq 0 ]]; then
        ip addr add "${G3P_HOST_IP}/24" dev "$G3P_IFACE" || true
      else
        sudo ip addr add "${G3P_HOST_IP}/24" dev "$G3P_IFACE" || true
      fi
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
      ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" src "$G3P_HOST_IP" || true
    else
      sudo ip route replace "${G3P_CAMERA_IP%.*}.0/24" dev "$G3P_IFACE" src "$G3P_HOST_IP" || true
    fi
    ip -4 -br addr show "$G3P_IFACE" || true
    ip route get "$G3P_CAMERA_IP" || true
  else
    echo "[g3p] warning: interface ${G3P_IFACE} not found; set G3P_IFACE=<iface>" >&2
  fi

  initial_probe="$(probe_rtsp_resolution || true)"
  echo "[g3p] initial RTSP probe resolution: ${initial_probe:-unavailable}"
  if [[ -x "$SCRIPT_DIR/setup_g3p_network.sh" ]]; then
    if [[ "$initial_probe" != "1920x1080" ]]; then
      echo "[g3p] running setup_g3p_network.sh and requesting RTSP viewfinder"
      G3P_CAMERA_IP="$G3P_CAMERA_IP" \
      G3P_HOST_IP="$G3P_HOST_IP" \
      G3P_IFACE="$G3P_IFACE" \
      G3P_START_VIEWFINDER="$G3P_START_VIEWFINDER" \
      "$SCRIPT_DIR/setup_g3p_network.sh" || echo "[g3p] setup script returned non-zero; continuing with fallback" >&2
    else
      echo "[g3p] RTSP already 1920x1080; skipping viewfinder reset"
    fi
  fi
  if [[ "$REQUIRE_1080P" == "1" ]]; then
    probe="$(wait_rtsp_resolution || true)"
    if [[ -z "$probe" && "$G3P_START_VIEWFINDER" == "1" && -x "$SCRIPT_DIR/start_g3p_viewfinder.py" ]]; then
      for ((recover = 1; recover <= G3P_VIEWFINDER_RECOVERY_ATTEMPTS; recover++)); do
        echo "[g3p] RTSP still unavailable; viewfinder recovery ${recover}/${G3P_VIEWFINDER_RECOVERY_ATTEMPTS}"
        G3P_RESETVF_PARAM="${G3P_RESETVF_PARAM:-start}" \
          "$PYTHON" "$SCRIPT_DIR/start_g3p_viewfinder.py" --host "$G3P_CAMERA_IP" --port 7878 --soft-fail || true
        echo "[g3p] waiting ${G3P_VIEWFINDER_SETTLE_S}s for RTSP encoder"
        sleep "$G3P_VIEWFINDER_SETTLE_S"
        probe="$(wait_rtsp_resolution || true)"
        [[ -n "$probe" ]] && break
      done
    fi
    echo "[g3p] RTSP probe resolution: ${probe:-unavailable}"
    if [[ -n "$probe" && "$probe" != "1920x1080" ]]; then
      echo "[error] RTSP stream is not 1920x1080. Refusing to run lower-quality detection." >&2
      echo "        Check camera SecStream_resolution=1080P and restart viewfinder." >&2
      exit 7
    elif [[ -z "$probe" ]]; then
      echo "[warn] RTSP did not return frames during preflight; starting app so the camera thread can keep reconnecting." >&2
      if [[ "$G3P_STRICT_PREFLIGHT" == "1" ]]; then
        echo "[error] G3P_STRICT_PREFLIGHT=1 and RTSP is unavailable." >&2
        exit 7
      fi
    fi
  fi
fi

if [[ "$CAMERA_SOURCE" == "gstreamer" && -z "$CAMERA_PIPELINE" ]]; then
  CAMERA_PIPELINE="nvarguscamerasrc sensor-id=${SENSOR_ID} ! video/x-raw(memory:NVMM),width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1,format=NV12 ! nvvidconv ! video/x-raw,format=I420"
fi
if [[ "$CAMERA_SOURCE" == "gstreamer" ]]; then
  args+=(--camera-pipeline "$CAMERA_PIPELINE")
fi

if [[ "${COLOR_ONLY:-0}" == "1" ]]; then
  args+=(--color-only)
fi
if [[ "${FAKE_CAMERA:-0}" == "1" ]]; then
  args+=(--fake-camera)
fi
if [[ "${DISABLE_VEHICLE_MODEL:-0}" == "1" ]]; then
  args+=(--disable-vehicle-model)
fi
if [[ "${VEHICLE_HALF:-1}" == "0" ]]; then
  args+=(--no-vehicle-half)
fi
if [[ "$DRAW_VEHICLE_CANDIDATES" == "1" ]]; then
  args+=(--draw-vehicle-candidates)
fi
if [[ "$DRAW_REJECTED_MARKER_CANDIDATES" == "1" ]]; then
  args+=(--draw-rejected-marker-candidates)
fi
if [[ "$DRAW_RAW_VEHICLE_CANDIDATES" == "1" ]]; then
  args+=(--draw-raw-vehicle-candidates)
fi
if [[ "$VEHICLE_ROI_FIRST" == "0" ]]; then
  args+=(--no-vehicle-roi-first)
fi
if [[ "$REQUIRE_VEHICLE_MODEL_MATCH" == "0" ]]; then
  args+=(--no-require-vehicle-model-match)
fi
if [[ "$WHITE_VEHICLE_FALLBACK" == "1" ]]; then
  args+=(--white-vehicle-fallback)
else
  args+=(--no-white-vehicle-fallback)
fi
if [[ "$ROOF_RED_FALLBACK" == "1" ]]; then
  args+=(--roof-red-fallback)
else
  args+=(--no-roof-red-fallback)
fi
if [[ "${MOCK_FLIGHT:-0}" == "1" ]]; then
  args+=(--mock-flight)
fi
if [[ "${EMIT_UPDATES:-0}" == "1" ]]; then
  args+=(--emit-updates)
fi
if [[ "${ENABLE_GPS_REID:-0}" == "1" ]]; then
  args+=(--enable-gps-reid)
fi
if [[ "${ENABLE_PIXEL_REID:-0}" == "1" ]]; then
  args+=(--enable-pixel-reid)
fi
if [[ "${DRAW_PREDICTED_TRACKS:-0}" == "1" ]]; then
  args+=(--draw-predicted-tracks)
fi
if [[ "$MISSION_LATCH_ONCE" == "0" ]]; then
  args+=(--no-mission-latch-once)
fi
if [[ "$AUTO_RTL" == "1" ]]; then
  args+=(--auto-rtl)
else
  args+=(--no-auto-rtl)
fi
if [[ "$AUTO_RTL_REQUIRE_VALID_TARGET" == "0" ]]; then
  args+=(--no-auto-rtl-require-valid-target)
fi

start_g3p_route_keeper

echo "[preflight] checking target server syntax..."
"$PYTHON" -m py_compile "$SCRIPT_DIR/red_white_target_server_new_camera.py"

start_rosbag

"${args[@]}" &
APP_PID=$!

set +e
wait "$APP_PID"
APP_STATUS=$?
set -e

if (( APP_STATUS != 0 )); then
  echo "[error] target server exited with status $APP_STATUS" >&2
fi
cleanup "$APP_STATUS"
