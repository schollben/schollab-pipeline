#!/bin/bash
# pipeline.sh
# Single entry point for the schollab caiman+FAST pipeline.
#
# Usage:
#   bash pipeline.sh                       # launch GUI, start pipeline
#   bash pipeline.sh --attach              # follow live output (journalctl)
#   bash pipeline.sh --status              # service status + last log lines
#   bash pipeline.sh --stop                # stop a running pipeline
#   bash pipeline.sh --clean_caiman        # dry run: show caiman artifacts
#   bash pipeline.sh --clean_caiman --confirm  # delete caiman artifacts
#   bash pipeline.sh --clean_fast          # dry run: show FAST artifacts
#   bash pipeline.sh --clean_fast --confirm    # delete FAST artifacts
#   bash pipeline.sh --clean_all           # dry run: show all artifacts
#   bash pipeline.sh --clean_all --confirm     # delete everything

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="schollab-pipeline"
FAST_LOG_DIR="$REPO_DIR/fast/logs"
FAST_CONFIG="$REPO_DIR/fast/pipeline_config.json"

CAIMAN_PYTHON="$HOME/miniforge3/envs/caiman/bin/python"
REGISTRATION_SCRIPT="$REPO_DIR/caiman/registration.py"

# ── parse args ────────────────────────────────────────────────────────────────

MODE="start"
CONFIRM=false

for arg in "$@"; do
	case "$arg" in
		--attach)       MODE="attach"        ;;
		--status)       MODE="status"        ;;
		--stop)         MODE="stop"          ;;
		--clean_caiman) MODE="clean_caiman"  ;;
		--clean_fast)   MODE="clean_fast"    ;;
		--clean_all)    MODE="clean_all"     ;;
		--confirm)      CONFIRM=true         ;;
	esac
done

# ── helpers ───────────────────────────────────────────────────────────────────

# Read folder list and scratch/log dirs from the FAST config
_read_config() {
	if [ ! -f "$FAST_CONFIG" ]; then
		echo "ERROR: pipeline_config.json not found at $FAST_CONFIG"
		exit 1
	fi
	SCRATCH_DIR=$(python3 -c "import json; c=json.load(open('$FAST_CONFIG')); print(c['scratch_dir'])")
	FOLDERS=$(python3    -c "import json; c=json.load(open('$FAST_CONFIG')); [print(f) for f in c['data_folders']]")
}

_delete() {
	local target="$1"
	if [ -e "$target" ] || [ -d "$target" ]; then
		if $CONFIRM; then
			rm -rf "$target"
			echo "  deleted: $target"
		else
			echo "  [DRY-RUN] would delete: $target"
		fi
	fi
}

# ── clean_caiman: remove caiman registration artifacts ────────────────────────

_clean_caiman() {
	_read_config
	echo ""
	echo "════════════════════════════════════════════════════════"
	echo "  Clean caiman artifacts"
	$CONFIRM && echo "  MODE: DELETING" || echo "  MODE: DRY RUN  (add --confirm to delete)"
	echo "════════════════════════════════════════════════════════"
	echo ""

	# Safety check: warn if any folder has registered.h5 but no source TIFs.
	# registered.h5 is irreplaceable if the original TIFs have been deleted.
	if $CONFIRM; then
		UNSAFE_FOLDERS=()
		while IFS= read -r folder; do
			folder="${folder%/}"
			if [ -f "$folder/registered.h5" ]; then
				tif_count=$(find "$folder" -maxdepth 1 -name "*.tif" 2>/dev/null | grep -v References | wc -l)
				if [ "$tif_count" -eq 0 ]; then
					UNSAFE_FOLDERS+=("$folder")
				fi
			fi
		done <<< "$FOLDERS"

		if [ "${#UNSAFE_FOLDERS[@]}" -gt 0 ]; then
			echo "WARNING: The following folders have registered.h5 but NO source TIFs."
			echo "  Deleting registered.h5 here is PERMANENT — the data cannot be recovered."
			echo ""
			for f in "${UNSAFE_FOLDERS[@]}"; do
				echo "  $f"
			done
			echo ""
			read -r -p "Type 'yes' to confirm permanent deletion, or anything else to abort: " ans
			if [ "$ans" != "yes" ]; then
				echo "Aborted."
				exit 1
			fi
			echo ""
		fi
	fi

	while IFS= read -r folder; do
		folder="${folder%/}"
		echo "Folder: $folder"

		# H5 files produced by TIFs→H5 and registration steps
		_delete "$folder/unregistered.h5"
		_delete "$folder/registered.h5"

		# Shift CSVs written by rigid and non-rigid registration
		_delete "$folder/rigid_shifts.csv"
		_delete "$folder/nonrigid_x_shifts.csv"
		_delete "$folder/nonrigid_y_shifts.csv"

		# Sample TIFFs written by register_one_session (4000-frame previews)
		for tif in "$folder"/*_rigid.tif "$folder"/*_nonrigid.tif; do
			[ -e "$tif" ] && _delete "$tif"
		done

		echo ""
	done <<< "$FOLDERS"
}

# ── clean_fast: remove FAST denoising artifacts ───────────────────────────────

_clean_fast() {
	_read_config
	echo ""
	echo "════════════════════════════════════════════════════════"
	echo "  Clean FAST artifacts"
	$CONFIRM && echo "  MODE: DELETING" || echo "  MODE: DRY RUN  (add --confirm to delete)"
	echo "  Scratch: $SCRATCH_DIR"
	echo "  Logs:    $FAST_LOG_DIR"
	echo "════════════════════════════════════════════════════════"
	echo ""

	while IFS= read -r folder; do
		folder="${folder%/}"
		echo "Folder: $folder"

		# Permanent drive: model weights, denoised output, sentinel, configs
		_delete "$folder/checkpoint"
		_delete "$folder/inference.h5"
		_delete "$folder/_fast_complete"
		_delete "$folder/_run_config.json"
		_delete "$folder/_inference_config.json"

		# Example result TIFFs copied to session root by FAST
		for tif in "$folder"/*_registered_*.tif; do
			[ -e "$tif" ] && _delete "$tif"
		done

		# Scratch (tmpfs): registered/, training/, result/ for this folder
		folder_id=$(basename "$folder")
		_delete "$SCRATCH_DIR/$folder_id"

		echo ""
	done <<< "$FOLDERS"

	# FAST log files
	echo "Logs:"
	_delete "$FAST_LOG_DIR/_pipeline_status.json"
	for f in "$FAST_LOG_DIR"/_pipeline_log_*.txt; do
		[ -e "$f" ] && _delete "$f"
	done
	_delete "$FAST_LOG_DIR/gpu_stats.csv"
	_delete "$FAST_LOG_DIR/ram_stats.txt"
	echo ""
}

# ── clean modes ───────────────────────────────────────────────────────────────

if [ "$MODE" = "clean_caiman" ]; then
	_clean_caiman
	echo "Caiman dry run complete. Add --confirm to delete." && ! $CONFIRM && exit 0
	echo "Caiman clean complete." && exit 0
fi

if [ "$MODE" = "clean_fast" ]; then
	_clean_fast
	echo "FAST dry run complete. Add --confirm to delete." && ! $CONFIRM && exit 0
	echo "FAST clean complete." && exit 0
fi

if [ "$MODE" = "clean_all" ]; then
	_clean_caiman
	_clean_fast
	echo "Dry run complete. Add --confirm to delete." && ! $CONFIRM && exit 0
	echo "Full clean complete. Ready for a fresh run:" && exit 0
fi

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

if [ ! -f "$CAIMAN_PYTHON" ]; then
	echo "ERROR: caiman python not found at $CAIMAN_PYTHON"
	echo "  Check that the caiman conda env is installed in ~/miniforge3/envs/caiman/"
	exit 1
fi

# Enable linger so user manager survives session death
loginctl enable-linger "$USER"

mkdir -p "$FAST_LOG_DIR"

echo "Starting Schollab pipeline..."
echo "  Caiman python: $CAIMAN_PYTHON"
echo "  Repo:          $REPO_DIR"
echo ""
echo "Opening folder selection GUI..."
echo ""

# Blocks until user clicks Run; registration.py writes job JSON + launches systemd service
"$CAIMAN_PYTHON" "$REGISTRATION_SCRIPT"

echo ""
echo "Useful commands:"
echo "  Follow live output:  bash pipeline.sh --attach"
echo "  Check status:        bash pipeline.sh --status"
echo "  Stop pipeline:       bash pipeline.sh --stop"
echo "  Raw journal:         journalctl --user -f -u $UNIT_NAME"
