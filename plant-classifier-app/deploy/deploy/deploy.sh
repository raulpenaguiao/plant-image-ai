#!/usr/bin/env bash
# Runs ON THE VPS (as root), AFTER the workflow has rsynced the code into $APP_DIR/plant-classifier-app.
# Installs dependencies, trains the models if missing, then swaps the service to the real app
# (replacing the unit that undeploy.sh wrote, if any). The old service/maintenance page keeps
# answering until the new app is ready; a failed deploy never takes the site down.
#
# Env:
#   APP_DIR   git checkout root (the folder that contains plant-classifier-app)   (required)
#   APP_USER  user the app runs as              (default: owner of the app folder; never root)
#   SERVICE   systemd unit                       (default: plant-classifier)
#   PORT      port nginx proxies to              (default: 8000)
set -euo pipefail

SERVICE="${SERVICE:-plant-classifier}"
PORT="${PORT:-8000}"
APP_DIR="${APP_DIR:?APP_DIR is required}"
UNIT="/etc/systemd/system/${SERVICE}.service"

[ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }
[[ "$SERVICE" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo "bad SERVICE: $SERVICE" >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "bad PORT: $PORT" >&2; exit 1; }
APP_DIR="$(cd "$APP_DIR" && pwd)"
APP="$APP_DIR/plant-classifier-app"
[ -f "$APP/requirements.txt" ] || { echo "no code at $APP (did the rsync step run?)" >&2; exit 1; }

APP_USER="${APP_USER:-$(stat -c %U "$APP")}"
if [ "$APP_USER" = "root" ] || ! id "$APP_USER" >/dev/null 2>&1; then
  echo "APP_USER must be an existing non-root user (got '$APP_USER'); set the VPS_APP_USER secret" >&2; exit 1
fi
chown -R "$APP_USER": "$APP"
run() { runuser -u "$APP_USER" -- "$@"; }   # run as the app user, in the current directory
cd "$APP"

echo "==> Python virtualenv and dependencies"
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
[ -x venv/bin/python ] || run python3 -m venv venv || { echo "python3-venv missing? apt install python3-venv" >&2; exit 1; }
want="$(sha256sum requirements.txt | cut -d' ' -f1)"
if [ "$(cat venv/.requirements.sha256 2>/dev/null || true)" != "$want" ]; then
  run venv/bin/pip install --no-cache-dir --upgrade pip
  run venv/bin/pip install --no-cache-dir -r requirements.txt
  printf '%s' "$want" | run tee venv/.requirements.sha256 >/dev/null
else
  echo "requirements unchanged; skipping pip"
fi

echo "==> Models"
run mkdir -p static models
if [ -s models/linear_probe.pkl ] && [ -s models/labels.json ] && [ -s models/faiss.index ] && [ -s models/faiss_meta.json ]; then
  echo "models present; skipping training"
else
  echo "models missing: downloading data and training (this takes a while)"
  run nice -n 10 venv/bin/python scripts/prepare_data.py
fi

echo "==> Switching $SERVICE to the app (replacing $UNIT)"
rm -rf "/etc/systemd/system/${SERVICE}.service.d"
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Plant Classifier
After=network.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP}
ExecStart=${APP}/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT_EOF
systemctl daemon-reload
systemctl enable "$SERVICE"          # starts at every boot
systemctl restart "$SERVICE"

echo "==> Waiting for the app (model load can take a minute)"
ok=""
for i in $(seq 1 90); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/pics-list" 2>/dev/null; then ok=1; break; fi
  systemctl is-active --quiet "$SERVICE" || break
  sleep 2
done
if [ -z "$ok" ]; then
  echo "the app did not come up on port ${PORT}:" >&2
  journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
  exit 1
fi
code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/")"
[ "$code" = "200" ] || { echo "GET / returned $code, expected 200" >&2; exit 1; }
systemctl is-enabled "$SERVICE" | grep -qx enabled

rm -rf "/opt/${SERVICE}-maintenance"   # the page undeploy.sh installed, if any

echo "==> nginx (reverse proxy)"
if systemctl cat nginx >/dev/null 2>&1; then
  systemctl enable nginx
  nginx -t && systemctl reload nginx || systemctl restart nginx
else
  echo "nginx not installed; the app is only on 127.0.0.1:${PORT}"
fi
echo "OK: the app is live on 127.0.0.1:${PORT}"
