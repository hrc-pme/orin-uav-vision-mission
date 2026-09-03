#!/usr/bin/env bash
set -euo pipefail

cd /home/hrc/xin_uav/unity

stop_old_publishers() {
  local pattern
  local pids
  local patterns=(
    "[u]nity_dashboard_bridge.py"
    "[d]455_car_approach_bridge.py"
    "[s]imulate_unity_confirm_from_bag.py"
    "[r]os2 bag play"
    "[r]un_bag_.*[.]sh"
  )

  for pattern in "${patterns[@]}"; do
    pids="$(pgrep -f "$pattern" || true)"
    if [[ -n "$pids" ]]; then
      echo "Stopping previous dashboard process(es): $pids"
      kill -TERM $pids 2>/dev/null || true
    fi
  done

  for _attempt in {1..100}; do
    if flock -n /tmp/unity_dashboard_live.lock -c true; then
      return
    fi
    sleep 0.2
  done

  echo "Previous Unity bridge did not stop within 20 seconds." >&2
  exit 1
}

stop_old_publishers

echo "INDOOR SAFE mode: D455 + TensorRT + Unity compressed annotated image."
echo "Automatic verification and all MAVLink flight commands are forced OFF."

D455_APPROACH_INDOOR_SAFE=1 \
CONFIG=config_d455_car_approach.yaml \
BRIDGE_PROGRAM=d455_car_approach_bridge.py \
exec ./run_all.sh
