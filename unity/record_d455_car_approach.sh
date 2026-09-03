#!/usr/bin/env bash
set -euo pipefail

cd /home/hrc/xin_uav/unity
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
RECORD_PREFIX=d455_car_approach \
RECORD_TARGET_DESCRIPTION="all-color car candidates" \
RECORD_EXTRA_TOPICS="/unity/approach_state" \
exec ./record_experiment.sh
