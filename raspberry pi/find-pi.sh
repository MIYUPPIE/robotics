#!/usr/bin/env bash
# find-pi.sh - locate the Raspberry Pi on the LAN after boot.
# Pi 4 Ethernet MAC OUI is dc:a6:32 (also matches b8:27:eb/e4:5f:01/d8:3a:dd/2c:cf:67).
set -u

SUBNET="${1:-192.168.1.0/24}"
PI_OUI='dc:a6:32|b8:27:eb|e4:5f:01|28:cd:c1|d8:3a:dd|2c:cf:67'
PI_USER="${PI_USER:-okhub}"

echo "=== ping-sweeping $SUBNET ($(date +%T)) ==="
nmap -sn "$SUBNET" >/dev/null 2>&1

echo "=== live hosts ==="
ip neigh | grep -ivE "INCOMPLETE|FAILED" | sort -t. -k4 -n

echo
echo "=== Raspberry Pi (by MAC OUI) ==="
PI_LINE=$(ip neigh | grep -iE "$PI_OUI" | grep -ivE "INCOMPLETE|FAILED" | head -1)
if [ -n "$PI_LINE" ]; then
  PI_IP=$(echo "$PI_LINE" | awk '{print $1}')
  echo ">>> FOUND Pi at $PI_IP  ($PI_LINE)"
  echo ">>> ssh ${PI_USER}@${PI_IP}"
else
  echo ">>> no Pi MAC on the subnet yet"
fi

echo
echo "=== mDNS names (avahi-daemon must be up on the Pi) ==="
for H in okpi.local okhub-2.local okhub.local raspberrypi.local; do
  R=$(getent hosts "$H" 2>/dev/null | awk '{print $1}')
  [ -n "$R" ] && echo "  $H -> $R   (ssh ${PI_USER}@${H})"
done
