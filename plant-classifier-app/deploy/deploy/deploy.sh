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
APP_DIR="$(cd "$APP_DIR" && pwd)"
APP="$APP_DIR/plant-classifier-app"
[ -f "$APP/requirements.txt" ] || { echo "no code at $APP (did the rsync step run?)" >&2; exit 1; }

if [ -n "$ROOTLESS" ]; then
  run() { "$@"; }                           # already the app user
else
  APP_USER="${APP_USER:-$(stat -c %U "$APP")}"
  if [ "$APP_USER" = "root" ] || ! id "$APP_USER" >/dev/null 2>&1; then
    echo "APP_USER must be an existing non-root user (got '$APP_USER'); set the VPS_APP_USER secret" >&2; exit 1
  fi
  chown -R "$APP_USER": "$APP"
  run() { runuser -u "$APP_USER" -- "$@"; }   # run as the app user, in the current directory
fi
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
rm -rf "${UNIT}.d"
if [ -z "$ROOTLESS" ]; then USERLINE="User=${APP_USER}"; WANTED=multi-user.target; else USERLINE=""; WANTED=default.target; fi
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Plant Classifier
After=network.target

[Service]
${USERLINE}
WorkingDirectory=${APP}
ExecStart=${APP}/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=${WANTED}
UNIT_EOF
$SCTL daemon-reload
$SCTL enable "$SERVICE"          # starts at every boot
$SCTL restart "$SERVICE"

echo "==> Waiting for the app (model load can take a minute)"
ok=""
for i in $(seq 1 90); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/pics-list" 2>/dev/null; then ok=1; break; fi
  $SCTL is-active --quiet "$SERVICE" || break
  sleep 2
done
if [ -z "$ok" ]; then
  echo "the app did not come up on port ${PORT}:" >&2
  $JCTL -u "$SERVICE" -n 40 --no-pager >&2 || true
  exit 1
fi
code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/")"
[ "$code" = "200" ] || { echo "GET / returned $code, expected 200" >&2; exit 1; }
$SCTL is-enabled "$SERVICE" | grep -qx enabled

rm -rf "/opt/${SERVICE}-maintenance" "$HOME/.local/share/${SERVICE}-maintenance"   # the page undeploy.sh installed, if any

if [ -z "$ROOTLESS" ]; then
echo "==> nginx (reverse proxy)"
if systemctl cat nginx >/dev/null 2>&1; then
  systemctl enable nginx
  nginx -t && systemctl reload nginx || systemctl restart nginx
else
  echo "nginx not installed; the app is only on 127.0.0.1:${PORT}"
fi
else
  echo "rootless: nginx is not touched (a root user manages it)"
fi
echo "OK: the app is live on 127.0.0.1:${PORT}"
