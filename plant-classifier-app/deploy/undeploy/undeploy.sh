#!/usr/bin/env bash
# Runs ON THE VPS (as root). Replaces the plant-classifier service with a static
# "not maintained" page and deletes the large training data and runtime models.
#
# Env (all optional):
#   SERVICE   systemd unit to replace          (default: plant-classifier)
#   APP_DIR   deployed app checkout            (default: unset -> data cleanup limited to caches)
#   APP_USER  user whose caches are deleted    (default: owner of APP_DIR, else none)
#   PORT      port nginx proxies to            (default: 8000)
set -euo pipefail

SERVICE="${SERVICE:-plant-classifier}"
APP_DIR="${APP_DIR:-}"
APP_USER="${APP_USER:-}"
PORT="${PORT:-8000}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/${SERVICE}-maintenance"
UNIT="/etc/systemd/system/${SERVICE}.service"

[ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }
[[ "$SERVICE" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo "bad SERVICE: $SERVICE" >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "bad PORT: $PORT" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found on VPS" >&2; exit 1; }

echo "==> Stopping old service"
systemctl stop "$SERVICE" 2>/dev/null || true

echo "==> Installing maintenance page to $DEST"
install -d -m 755 "$DEST"
install -m 644 "$SRC/index.html" "$DEST/index.html"
install -m 644 "$SRC/server.py" "$DEST/server.py"

echo "==> Overwriting $UNIT"
rm -rf "/etc/systemd/system/${SERVICE}.service.d"   # drop overrides of the old deploy
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Plant Classifier - retired (maintenance page)
After=network.target

[Service]
Environment=HOST=127.0.0.1 PORT=${PORT}
ExecStart=$(command -v python3) -I ${DEST}/server.py
DynamicUser=yes
Restart=always
RestartSec=2
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT_EOF
systemctl daemon-reload
systemctl enable "$SERVICE"            # starts on every boot
systemctl restart "$SERVICE"

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
  [ -n "$APP_USER" ] || APP_USER="$(stat -c %U "$APP_DIR")"
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
  if [ -n "$HOME_DIR" ] && [ "$HOME_DIR" != "/" ]; then
    safe_rm "$HOME_DIR/.cache/huggingface"  # PlantVillage dataset + CLIP weights
    safe_rm "$HOME_DIR/.cache/torch"
    safe_rm "$HOME_DIR/.cache/open_clip"
    safe_rm "$HOME_DIR/.cache/pip"
  fi
fi
# the service may also have run as root
safe_rm /root/.cache/huggingface
safe_rm /root/.cache/torch
safe_rm /root/.cache/open_clip

echo "==> Making sure nginx (reverse proxy) is up and survives reboots"
if systemctl list-unit-files nginx.service >/dev/null 2>&1 && systemctl cat nginx >/dev/null 2>&1; then
  systemctl enable nginx
  nginx -t && systemctl reload nginx || systemctl restart nginx
else
  echo "nginx not installed; serving directly on 127.0.0.1:${PORT} only"
fi

echo "==> Verifying"
systemctl is-enabled "$SERVICE" | grep -qx enabled
systemctl is-active "$SERVICE" | grep -qx active
for i in 1 2 3 4 5 6 7 8 9 10; do
  code="$(curl -s -o /tmp/.undeploy_body -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)"
  [ "$code" = "410" ] && grep -q "not maintained anymore" /tmp/.undeploy_body && { rm -f /tmp/.undeploy_body; echo "OK: maintenance page is live (HTTP 410)"; exit 0; }
  sleep 1
done
rm -f /tmp/.undeploy_body
echo "maintenance page did not come up on port ${PORT}" >&2
journalctl -u "$SERVICE" -n 30 --no-pager >&2 || true
exit 1
