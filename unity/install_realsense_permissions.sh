#!/usr/bin/env bash
set -e

cd /home/hrc/xin_uav/unity
sudo install -m 0644 99-realsense-d455.rules /etc/udev/rules.d/99-realsense-d455.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
sudo udevadm trigger --subsystem-match=hidraw

echo "RealSense udev 權限已安裝。請拔掉再插回 D455，然後執行 ./run_all.sh。"
