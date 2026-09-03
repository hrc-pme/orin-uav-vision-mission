#!/usr/bin/env bash
set -euo pipefail

CAMERA_IP="${G3P_CAMERA_IP:-192.168.144.135}"
PRIMARY_HOST_IP="${G3P_HOST_IP:-192.168.144.10}"
FALLBACK_HOST_IP="${G3P_FALLBACK_HOST_IP:-192.168.144.20}"
PREFIX="${G3P_PREFIX:-24}"
REQUESTED_IFACE="${G3P_IFACE:-}"

log() {
  printf '[g3p-network] %s\n' "$*" >&2
}

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

is_excluded_iface() {
  case "$1" in
    lo|docker*|br-*|virbr*|veth*|tailscale*|wg*|wl*|wlan*|wifi*|p2p-*|usb0|usb1|usb2|l4tbr0)
      return 0
      ;;
  esac
  [ -d "/sys/class/net/$1/wireless" ]
}

is_ethernet_iface() {
  local iface="$1"
  [ -e "/sys/class/net/$iface/type" ] || return 1
  [ "$(cat "/sys/class/net/$iface/type" 2>/dev/null)" = "1" ] || return 1
  ! is_excluded_iface "$iface"
}

iface_path() {
  readlink -f "/sys/class/net/$1" 2>/dev/null || true
}

iface_carrier() {
  cat "/sys/class/net/$1/carrier" 2>/dev/null || echo 0
}

iface_operstate() {
  cat "/sys/class/net/$1/operstate" 2>/dev/null || echo unknown
}

list_candidates() {
  for path in /sys/class/net/*; do
    local iface="${path##*/}"
    is_ethernet_iface "$iface" || continue
    printf '%s\n' "$iface"
  done
}

choose_iface() {
  if [ -n "$REQUESTED_IFACE" ]; then
    is_ethernet_iface "$REQUESTED_IFACE" || {
      log "Requested G3P_IFACE=$REQUESTED_IFACE is not a usable Ethernet interface."
      return 1
    }
    printf '%s\n' "$REQUESTED_IFACE"
    return 0
  fi

  mapfile -t candidates < <(list_candidates)
  if [ "${#candidates[@]}" -eq 0 ]; then
    log "No Ethernet interface found. Wi-Fi/docker/bridge/veth interfaces are ignored."
    return 1
  fi

  log "Ethernet candidates:"
  carrier_ifaces=()
  usb_carrier_ifaces=()
  for iface in "${candidates[@]}"; do
    local carrier oper ips path
    carrier="$(iface_carrier "$iface")"
    oper="$(iface_operstate "$iface")"
    ips="$(ip -4 -br addr show "$iface" | awk '{for (i=3; i<=NF; i++) printf "%s%s", $i, (i<NF ? " " : "\n")}')"
    path="$(iface_path "$iface")"
    log "  $iface carrier=$carrier oper=$oper ip=${ips:-none} path=$path"
    if [ "$carrier" = "1" ] || [ "$oper" = "up" ]; then
      carrier_ifaces+=("$iface")
      if printf '%s' "$path" | grep -q '/usb'; then
        usb_carrier_ifaces+=("$iface")
      fi
    fi
  done

  if [ "${#usb_carrier_ifaces[@]}" -eq 1 ]; then
    printf '%s\n' "${usb_carrier_ifaces[0]}"
    return 0
  fi

  if [ "${#carrier_ifaces[@]}" -eq 1 ]; then
    printf '%s\n' "${carrier_ifaces[0]}"
    return 0
  fi

  if [ "${#carrier_ifaces[@]}" -eq 0 ]; then
    log "No Ethernet interface has carrier. Check camera power, RJ45 wiring, and USB Ethernet adapter."
    return 1
  fi

  log "Multiple Ethernet interfaces have carrier: ${carrier_ifaces[*]}"
  log "Set G3P_IFACE=<interface> and run again."
  return 1
}

selected_iface="$(choose_iface)"
log "Selected interface: $selected_iface"

if ip -4 addr show scope global | grep -q "192\.168\.144\.135/${PREFIX}"; then
  log "192.168.144.135/${PREFIX} is the camera IP and is already assigned on this host. Refusing to continue."
  exit 1
fi

host_ip="$PRIMARY_HOST_IP"
if ip -4 addr show scope global | grep -q "${PRIMARY_HOST_IP}/${PREFIX}" &&
   ! ip -4 addr show "$selected_iface" | grep -q "${PRIMARY_HOST_IP}/${PREFIX}"; then
  host_ip="$FALLBACK_HOST_IP"
  log "$PRIMARY_HOST_IP/${PREFIX} is already used on another interface. Using $host_ip/${PREFIX}."
fi

if ip -4 addr show "$selected_iface" | grep -q "192\.168\.144\.[0-9]\+/${PREFIX}"; then
  log "$selected_iface already has a 192.168.144.x/${PREFIX} address."
else
  log "Adding temporary address $host_ip/${PREFIX} to $selected_iface."
  run_root ip addr add "$host_ip/${PREFIX}" dev "$selected_iface"
fi

ip -br addr show "$selected_iface"

log "Forcing route ${CAMERA_IP%.*}.0/${PREFIX} through $selected_iface from $host_ip."
run_root ip route replace "${CAMERA_IP%.*}.0/${PREFIX}" dev "$selected_iface" src "$host_ip"
ip route get "$CAMERA_IP"

route_dev="$(ip route get "$CAMERA_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
if [ "$route_dev" != "$selected_iface" ]; then
  log "Route to $CAMERA_IP uses ${route_dev:-unknown}, not $selected_iface."
  exit 1
fi

if ping -c 1 -W 1 -I "$selected_iface" "$CAMERA_IP" >/dev/null 2>&1; then
  log "Ping to $CAMERA_IP succeeded."
else
  log "Ping to $CAMERA_IP failed. Continuing because some cameras block ICMP."
fi

if command -v nc >/dev/null 2>&1; then
  if timeout 4 nc -z -w 2 "$CAMERA_IP" 554 >/dev/null 2>&1; then
    log "RTSP TCP port 554 is reachable."
  else
    log "RTSP TCP port 554 is not reachable yet."
  fi

  if [ "${G3P_START_VIEWFINDER:-1}" = "1" ] &&
     timeout 4 nc -z -w 2 "$CAMERA_IP" 7878 >/dev/null 2>&1; then
    control_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_g3p_viewfinder.py"
    if [ -x "$control_script" ] && command -v /usr/bin/python3 >/dev/null 2>&1; then
      log "G3P control port 7878 is reachable. Checking camera state and requesting RTSP viewfinder."
      if ! /usr/bin/python3 "$control_script" --host "$CAMERA_IP" --port 7878 --soft-fail; then
        log "G3P control command failed unexpectedly. Continuing to RTSP/ROS checks."
      fi
    else
      log "G3P control script is not executable or /usr/bin/python3 is missing: $control_script"
    fi
  fi
fi

if [ "${G3P_PROBE_RTSP:-0}" = "1" ] && command -v ffprobe >/dev/null 2>&1; then
  if timeout 8 ffprobe -rtsp_transport tcp -v error \
    -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate \
    -of default=noprint_wrappers=1 "rtsp://${CAMERA_IP}/live" >/tmp/g3p_ffprobe_network.out 2>/tmp/g3p_ffprobe_network.err; then
    log "RTSP probe succeeded:"
    sed 's/^/[g3p-network]   /' /tmp/g3p_ffprobe_network.out
  else
    log "RTSP probe did not succeed yet:"
    sed 's/^/[g3p-network]   /' /tmp/g3p_ffprobe_network.err || true
  fi
fi

log "Network setup complete for $selected_iface."
