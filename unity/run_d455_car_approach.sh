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
      echo "Stopping stale publisher(s): $pids"
      kill -TERM $pids 2>/dev/null || true
    fi
  done

  # The previous run owns this lock until it has also stopped rosbridge.
  # Waiting here prevents a fast indoor -> outdoor switch from failing with
  # "Another Unity dashboard live bridge is already running."
  for _attempt in {1..100}; do
    if flock -n /tmp/unity_dashboard_live.lock -c true; then
      return
    fi
    sleep 0.2
  done

  echo "Previous Unity bridge did not stop within 20 seconds; refusing to start a second instance." >&2
  exit 1
}

stop_old_publishers

if [[ "${D455_APPROACH_INDOOR_SAFE:-0}" == "1" ]]; then
  echo "INDOOR_SAFE: outdoor GPS/battery gate skipped; all flight commands remain disabled."
else
  echo "Running mandatory outdoor hardware preflight..."
  ./preflight_d455_car_approach.py
fi

echo "D455 CAR APPROACH mode"
echo "  Lock-one-car altitude gate: 80--130 m above home"
echo "  Before CONFIRM: detect/lock only; no automatic movement"
echo "  After Unity CONFIRM: freeze current GPS position and heading"
echo "  Descend directly to 20 m with no visual horizontal correction"
echo "  After CONFIRM no further detection is required; tracker loss is ignored"
echo "  CONFIRM accepts the locked car within 60% of the image center"
echo "  Detection target: any model class 'car' (all colors)"
echo "This program never ARM or TAKEOFF automatically. Keep RC takeover ready."

CONFIG=config_d455_car_approach.yaml \
BRIDGE_PROGRAM=d455_car_approach_bridge.py \
exec ./run_all.sh
