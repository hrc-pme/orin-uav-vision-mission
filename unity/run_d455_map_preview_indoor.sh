#!/usr/bin/env bash
set -euo pipefail

cd /home/hrc/xin_uav/unity

echo "INDOOR MAP PREVIEW: fake WGS84 is used only to verify the Unity Cesium city."
echo "Automatic verification and all MAVLink flight commands remain forced OFF."

D455_APPROACH_MAP_PREVIEW=1 \
exec ./run_d455_car_approach_indoor.sh
