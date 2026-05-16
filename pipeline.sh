#!/bin/bash
# pipeline.sh
# Convenience wrapper for the schollab caiman+FAST pipeline.
#
# The actual pipeline is launched by registration.py (GUI → job JSON → systemd).
# This script handles the pre-flight setup and provides status/attach/stop modes.
#
# Usage:
#   bash pipeline.sh              # launch GUI to select folders and start pipeline
#   bash pipeline.sh --attach     # follow live output (journalctl)
#   bash pipeline.sh --status     # show service status + last log lines
#   bash pipeline.sh --stop       # stop a running pipeline

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="schollab-pipeline"
FAST_LOG_DIR="$REPO_DIR/fast/logs"

CAIMAN_PYTHON="$HOME/miniforge3/envs/caiman/bin/python"
REGISTRATION_SCRIPT="$REPO_DIR/caiman/registration.py"

# ── parse args ────────────────────────────────────────────────────────────────

MODE="start"
for arg in "$@"; do
	case "$arg" in
		--attach) MODE="attach" ;;
		--status) MODE="status" ;;
		--stop)   MODE="stop"   ;;
	esac
done

# ── status mode ───────────────────────────────────────────────────────────────

if [ "$MODE" = "status" ]; then
	echo "── systemd service ──────────────────────────────────"
	systemctl --user status "$UNIT_NAME" 2>/dev/null || echo "Service not active"
	echo ""
	echo "── Last 20 FAST log lines ───────────────────────────"
	tail -20 "$(ls -t "$FAST_LOG_DIR"/_pipeline_log_*.txt 2>/dev/null | head -1)" 2>/dev/null \
		|| echo "No FAST log file found"
	exit 0
fi

# ── attach mode ───────────────────────────────────────────────────────────────

if [ "$MODE" = "attach" ]; then
	echo "Following pipeline output — Ctrl+C to detach (pipeline keeps running)"
	echo ""
	journalctl --user -f -u "$UNIT_NAME"
	exit 0
fi

# ── stop mode ─────────────────────────────────────────────────────────────────

if [ "$MODE" = "stop" ]; then
	systemctl --user stop "$UNIT_NAME" 2>/dev/null \
		&& echo "Pipeline stopped." \
		|| echo "Pipeline was not running."
	exit 0
fi

# ── start mode ────────────────────────────────────────────────────────────────

# Verify caiman python exists
if [ ! -f "$CAIMAN_PYTHON" ]; then
	echo "ERROR: caiman python not found at $CAIMAN_PYTHON"
	echo "  Check that the caiman conda env is installed in ~/miniforge3/envs/caiman/"
	exit 1
fi

# Enable linger so user manager survives session death.
# Required for systemd user services to outlive the login session.
loginctl enable-linger "$USER"

mkdir -p "$FAST_LOG_DIR"

echo "Starting Schollab pipeline..."
echo "  Caiman python: $CAIMAN_PYTHON"
echo "  Repo:          $REPO_DIR"
echo ""
echo "Opening folder selection GUI..."
echo ""

# Launch the GUI — this blocks until the user clicks Run.
# registration.py writes the job JSON and fires the systemd service, then exits.
"$CAIMAN_PYTHON" "$REGISTRATION_SCRIPT"

echo ""
echo "Useful commands:"
echo "  Follow live output:  bash pipeline.sh --attach"
echo "  Check status:        bash pipeline.sh --status"
echo "  Stop pipeline:       bash pipeline.sh --stop"
echo "  Raw journal:         journalctl --user -f -u $UNIT_NAME"
