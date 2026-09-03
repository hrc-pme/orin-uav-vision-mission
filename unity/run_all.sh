#!/usr/bin/env bash
set -e

ROS_SETUP=/opt/ros/humble/setup.bash
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup not found: $ROS_SETUP" >&2
  exit 1
fi

source "$ROS_SETUP"
cd /home/hrc/xin_uav/unity
CONFIG="${CONFIG:-config.yaml}"
BRIDGE_PROGRAM="${BRIDGE_PROGRAM:-unity_dashboard_bridge.py}"
FASTDDS_PROFILE=/home/hrc/xin_uav/unity/fastdds_udp_only.xml
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# Avoid stale/corrupted Fast DDS shared-memory segments after camera process
# restarts. All local ROS 2 traffic for this dashboard uses loopback UDP.
if [[ -f "$FASTDDS_PROFILE" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTDDS_PROFILE"
  export FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_PROFILE"
fi

exec 9>/tmp/unity_dashboard_live.lock
if ! flock -n 9; then
  echo "Another Unity dashboard live bridge is already running." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${ROSBRIDGE_PID:-}" ]]; then
    kill "$ROSBRIDGE_PID" 2>/dev/null || true
    wait "$ROSBRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ss -ltn "sport = :9090" | grep -q LISTEN; then
  echo "Using existing rosbridge at ws://0.0.0.0:9090"
elif ros2 pkg prefix rosbridge_server >/dev/null 2>&1; then
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
    default_call_service_timeout:=5.0 \
    call_services_in_new_thread:=true \
    send_action_goals_in_new_thread:=true &
  ROSBRIDGE_PID=$!
  echo "Started rosbridge at ws://0.0.0.0:9090"
else
  echo "rosbridge_server is not installed." >&2
  exit 1
fi

python3 "$BRIDGE_PROGRAM" --config "$CONFIG" --mode live
