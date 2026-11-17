#!/usr/bin/env bash
# Runs ON THE VPS (as root). Replaces the plant-classifier service with a static
# "not maintained" page and deletes the large training data and runtime models.
#
# Env (all optional):
#   SERVICE   systemd unit to replace          (default: plant-classifier)
#   APP_DIR   git checkout root (or the app dir itself)  (default: unset -> only caches are cleaned)
#   APP_USER  user whose caches are deleted    (default: owner of APP_DIR, else none)
#   PORT      port nginx proxies to            (default: 8000)
set -euo pipefail

SERVICE="${SERVICE:-plant-classifier}"
APP_DIR="${APP_DIR:-}"
APP_USER="${APP_USER:-}"
PORT="${PORT:-8000}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/${SERVICE}-maintenance"

[[ "$SERVICE" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo "bad SERVICE: $SERVICE" >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "bad PORT: $PORT" >&2; exit 1; }
# ROOTLESS=true: run as an ordinary account, as a systemd *user* service (no sudo anywhere).
# Needs once, by root: `loginctl enable-linger <account>` (service runs without a login, starts at boot).
if [ -n "${ROOTLESS:-}" ] && [ "${ROOTLESS}" != "false" ]; then ROOTLESS=1; else ROOTLESS=""; fi
if [ -z "$ROOTLESS" ]; then
  [ "$(id -u)" -eq 0 ] || { echo "must run as root (or set ROOTLESS=true to run without sudo)" >&2; exit 1; }
  SCTL="systemctl"; JCTL="journalctl"
  UNIT="/etc/systemd/system/${SERVICE}.service"
else
  [ "$(id -u)" -ne 0 ] || { echo "ROOTLESS needs an ordinary account, not root" >&2; exit 1; }
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
  if [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true)" != "yes" ]; then
    echo "lingering is off for $(id -un). A root user must run once:  loginctl enable-linger $(id -un)" >&2; exit 1
  fi
  [ -S "$XDG_RUNTIME_DIR/bus" ] || { echo "the user systemd manager is not running; is lingering active?" >&2; exit 1; }
  SCTL="systemctl --user"; JCTL="journalctl --user"
  UNIT="$HOME/.config/systemd/user/${SERVICE}.service"
  mkdir -p "$HOME/.config/systemd/user"
  if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
    echo "a system-wide '$SERVICE' service is still running (from a root deploy) and holds the port." >&2
    echo "A root user must stop it once:  systemctl disable --now $SERVICE" >&2; exit 1
  fi
fi
command -v python3 >/dev/null || { echo "python3 not found on VPS" >&2; exit 1; }

[ -z "$ROOTLESS" ] || DEST="$HOME/.local/share/${SERVICE}-maintenance"

echo "==> Stopping old service"
$SCTL stop "$SERVICE" 2>/dev/null || true

echo "==> Installing maintenance page to $DEST"
install -d -m 755 "$DEST"
install -m 644 "$SRC/index.html" "$DEST/index.html"
install -m 644 "$SRC/server.py" "$DEST/server.py"

echo "==> Overwriting $UNIT"
rm -rf "${UNIT}.d"   # drop overrides of the old deploy
if [ -z "$ROOTLESS" ]; then
  HARDEN="DynamicUser=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes"; WANTED=multi-user.target
else
  HARDEN=""; WANTED=default.target
fi
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Plant Classifier - retired (maintenance page)
After=network.target

[Service]
Environment=HOST=127.0.0.1 PORT=${PORT}
ExecStart=$(command -v python3) -I ${DEST}/server.py
Restart=always
RestartSec=2
NoNewPrivileges=yes
${HARDEN}

[Install]
WantedBy=${WANTED}
UNIT_EOF
$SCTL daemon-reload
$SCTL enable "$SERVICE"            # starts on every boot
$SCTL restart "$SERVICE"

echo "==> Deleting training data and runtime models"
safe_rm() {
  # refuse empty / root / very short paths
  local p="$1"
  if [ -z "$p" ] || [ "$p" = "/" ] || [ "${#p}" -lt 8 ]; then
    echo "refusing to delete '$p'" >&2; return 0
  fi
  if [ -e "$p" ]; then du -sh "$p" 2>/dev/null || true; rm -rf -- "$p"; fi
}

if [ -n "$APP_DIR" ] && [ -d "$APP_DIR" ]; then
  APP_DIR="$(cd "$APP_DIR" && pwd)"
  # APP_DIR is the git checkout root; the app itself lives in its plant-classifier-app/ subfolder
  if [ -d "$APP_DIR/plant-classifier-app" ]; then APP_DIR="$APP_DIR/plant-classifier-app"; fi
  [ -n "$APP_USER" ] || APP_USER="$(stat -c %U "$APP_DIR")"
  [ -z "$ROOTLESS" ] || APP_USER="$(id -un)"
  safe_rm "$APP_DIR/models"                 # CLIP/MLP/FAISS artefacts (.pkl, .index, .json)
  safe_rm "$APP_DIR/static/sample_images"   # gallery images from prepare_data.py
  safe_rm "$APP_DIR/venv"                   # torch + open_clip etc.
  safe_rm "$APP_DIR/.venv"
  safe_rm "$APP_DIR/data/raw"
  safe_rm "$APP_DIR/data/dataset"
else
  echo "APP_DIR not set or missing; skipping app-directory cleanup"
fi

if [ -n "$APP_USER" ] && id "$APP_USER" >/dev/null 2>&1; then
  HOME_DIR="$(getent passwd "$APP_USER" | cut -d: -f6)"
  [ -z "$ROOTLESS" ] || HOME_DIR="$HOME"
  if [ -n "$HOME_DIR" ] && [ "$HOME_DIR" != "/" ]; then
    safe_rm "$HOME_DIR/.cache/huggingface"  # PlantVillage dataset + CLIP weights
    safe_rm "$HOME_DIR/.cache/torch"
    safe_rm "$HOME_DIR/.cache/open_clip"
    safe_rm "$HOME_DIR/.cache/pip"
  fi
fi
if [ -z "$ROOTLESS" ]; then   # the service may also have run as root
  safe_rm /root/.cache/huggingface
  safe_rm /root/.cache/torch
  safe_rm /root/.cache/open_clip
fi

if [ -z "$ROOTLESS" ]; then
echo "==> Making sure nginx (reverse proxy) is up and survives reboots"
if systemctl list-unit-files nginx.service >/dev/null 2>&1 && systemctl cat nginx >/dev/null 2>&1; then
  systemctl enable nginx
  nginx -t && systemctl reload nginx || systemctl restart nginx
else
  echo "nginx not installed; serving directly on 127.0.0.1:${PORT} only"
fi
else
  echo "rootless: nginx is not touched (a root user manages it)"
fi

echo "==> Verifying"
$SCTL is-enabled "$SERVICE" | grep -qx enabled
$SCTL is-active "$SERVICE" | grep -qx active
for i in 1 2 3 4 5 6 7 8 9 10; do
  code="$(curl -s -o /tmp/.undeploy_body -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)"
  [ "$code" = "410" ] && grep -q "not maintained anymore" /tmp/.undeploy_body && { rm -f /tmp/.undeploy_body; echo "OK: maintenance page is live (HTTP 410)"; exit 0; }
  sleep 1
done
rm -f /tmp/.undeploy_body
echo "maintenance page did not come up on port ${PORT}" >&2
$JCTL -u "$SERVICE" -n 30 --no-pager >&2 || true
exit 1
