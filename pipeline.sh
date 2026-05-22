#!/bin/bash
# PreProcess2PImage.sh
# Single entry point for the schollab caiman+FAST pipeline.
#
# Usage:
#   bash PreProcess2PImage.sh                       # launch GUI, start pipeline
#   bash PreProcess2PImage.sh --attach              # follow live output (journalctl)
#   bash PreProcess2PImage.sh --status              # service status + last log lines
#   bash PreProcess2PImage.sh --stop                # stop a running pipeline
#   bash PreProcess2PImage.sh --clean_caiman        # scan: list CaImAn artifacts (no delete)
#   bash PreProcess2PImage.sh --clean_caiman --confirm   # scan, prompt, then delete if confirmed
#   bash PreProcess2PImage.sh --clean_caiman --confirm --yes   # non-interactive delete (scripts)
#   bash PreProcess2PImage.sh --clean_caiman -- /path/A /path/B   # explicit session folders
#   bash PreProcess2PImage.sh --clean_caiman --from-job  # use /tmp/pipeline_job.json sessions
#   bash PreProcess2PImage.sh --clean_fast …          # same -- / --from-job / config fallback
#   bash PreProcess2PImage.sh --clean_all …
#   bash PreProcess2PImage.sh --setup                # conda envs (caiman + FAST), linger, scratch tmpfs

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="schollab-PreProcess2PImage"
FAST_LOG_DIR="$REPO_DIR/fast/logs"
FAST_CONFIG="$REPO_DIR/fast/pipeline_config.json"

# Conda install prefix (Miniforge vs Miniconda, etc.). Export SCHOLLAB_CONDA_ROOT on hosts
# where envs live under e.g. ~/miniconda3 — must match registration.py / pipeline_worker.py.
SCHOLLAB_CONDA_ROOT="${SCHOLLAB_CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="${CONDA_BIN:-$SCHOLLAB_CONDA_ROOT/bin/conda}"

# FAST scratch tmpfs size (edit for your machine — must fit in RAM)
SCRATCH_TMPFS_SIZE="120G"

CAIMAN_PYTHON="$SCHOLLAB_CONDA_ROOT/envs/caiman/bin/python"
REGISTRATION_SCRIPT="$REPO_DIR/caiman/registration.py"
JOB_FILE="${JOB_FILE:-/tmp/pipeline_job.json}"

# ── parse args ────────────────────────────────────────────────────────────────

MODE="start"
CONFIRM=false
CLEAN_FROM_JOB=false
CLEAN_YES=false
PASSTHRU=false
CLEAN_PATHS=()

for arg in "$@"; do
	if $PASSTHRU; then
		CLEAN_PATHS+=("$arg")
		continue
	fi
	case "$arg" in
		--)             PASSTHRU=true        ;;
		--from-job)     CLEAN_FROM_JOB=true  ;;
		--yes)          CLEAN_YES=true       ;;
		--attach)       MODE="attach"        ;;
		--status)       MODE="status"        ;;
		--stop)         MODE="stop"          ;;
		--setup)        MODE="setup"         ;;
		--clean_caiman) MODE="clean_caiman"  ;;
		--clean_fast)   MODE="clean_fast"    ;;
		--clean_all)    MODE="clean_all"     ;;
		--confirm)      CONFIRM=true         ;;
	esac
done

# ── helpers ───────────────────────────────────────────────────────────────────

# scratch_dir only (GUI runs do not update data_folders in config).
_read_scratch_config() {
	if [ ! -f "$FAST_CONFIG" ]; then
		echo "ERROR: pipeline_config.json not found at $FAST_CONFIG"
		exit 1
	fi
	SCRATCH_DIR=$(python3 -c "import json; c=json.load(open('$FAST_CONFIG')); print(c['scratch_dir'])")
}

# Legacy full read (unused by --clean; kept for tooling).
_read_config() {
	if [ ! -f "$FAST_CONFIG" ]; then
		echo "ERROR: pipeline_config.json not found at $FAST_CONFIG"
		exit 1
	fi
	SCRATCH_DIR=$(python3 -c "import json; c=json.load(open('$FAST_CONFIG')); print(c['scratch_dir'])")
	FOLDERS=$(python3 -c "import json; c=json.load(open('$FAST_CONFIG')); [print(f) for f in c['data_folders']]")
}

# Session folders for --clean_*: paths after -- , --from-job, or config (warn if fallback).
_resolve_clean_folders() {
	_read_scratch_config
	if [ "${#CLEAN_PATHS[@]}" -gt 0 ]; then
		FOLDERS=$(printf '%s\n' "${CLEAN_PATHS[@]}")
		echo "── Folder source: ${#CLEAN_PATHS[@]} path(s) after -- ─────────────────────"
	elif $CLEAN_FROM_JOB; then
		if [ ! -f "$JOB_FILE" ]; then
			echo "ERROR: job file not found: $JOB_FILE"
			echo "  The GUI writes this from registration.py, or pass folders after --."
			exit 1
		fi
		export JOB_FILE
		FOLDERS=$(python3 <<'PY'
import json, os
p = os.environ["JOB_FILE"]
with open(p, encoding="utf-8") as f:
	d = json.load(f)
for s in d.get("sessions", []):
	print(s)
PY
		)
		echo "── Folder source: sessions from $JOB_FILE ─────────────────────────"
	else
		FOLDERS=$(python3 -c "import json; c=json.load(open('$FAST_CONFIG')); [print(f) for f in c['data_folders']]")
		echo "WARNING: Using data_folders from pipeline_config.json — may not match GUI-selected sessions." >&2
		echo "  Prefer: --from-job or pass session paths after -- ." >&2
		echo "── Folder source: pipeline_config.json data_folders (fallback) ─────────"
	fi
}

# Second prompt unless CLEAN_YES (non-interactive automation).
_clean_prompt_delete() {
	local n="$1"
	if $CLEAN_YES; then
		return 0
	fi
	if ! [ -t 0 ]; then
		echo "ERROR: stdin is not a TTY. Use --yes with --confirm for non-interactive delete."
		exit 1
	fi
	read -r -p "Delete these $n path(s)? [y/N] " ans
	if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
		echo "Aborted (no files removed)."
		exit 0
	fi
}

_clean_run_unsafe_registered_check() {
	local UNSAFE_FOLDERS=()
	local folder f
	local tif_count
	while IFS= read -r folder; do
		folder="${folder%/}"
		[ -z "$folder" ] && continue
		if [ -f "$folder/registered.h5" ]; then
			tif_count=$(find "$folder" -maxdepth 1 -type f \
				-name "*.tif" \
				! -name "*_rigid.tif" ! -name "*_nonrigid.tif" \
				2>/dev/null | grep -v References | wc -l)
			if [ "$tif_count" -eq 0 ]; then
				UNSAFE_FOLDERS+=("$folder")
			fi
		fi
	done <<< "$FOLDERS"

	if [ "${#UNSAFE_FOLDERS[@]}" -eq 0 ]; then
		return 0
	fi
	echo ""
	echo "WARNING: The following folders have registered.h5 but NO acquisition source TIFs"
	echo "  (previews *_rigid.tif / *_nonrigid.tif do not count)."
	echo "  Deleting registered.h5 here is PERMANENT."
	echo ""
	local f
	for f in "${UNSAFE_FOLDERS[@]}"; do
		echo "  $f"
	done
	echo ""
	if ! [ -t 0 ]; then
		echo "ERROR: Unsafe-folder confirmation needs a TTY." >&2
		exit 1
	fi
	read -r -p "Type 'yes' to confirm permanent deletion for these folders: " ans
	if [ "$ans" != "yes" ]; then
		echo "Aborted."
		exit 1
	fi
	echo ""
}

_clean_collect_caiman_paths() {
	shopt -s nullglob
	local out=""
	while IFS= read -r folder; do
		folder="${folder%/}"
		[ -z "$folder" ] && continue
		local p
		for p in \
			"$folder/unregistered.h5" \
			"$folder/registered.h5" \
			"$folder/rigid_shifts.csv" \
			"$folder/nonrigid_x_shifts.csv" \
			"$folder/nonrigid_y_shifts.csv"
		do
			[ -e "$p" ] && out+="$p"$'\n'
		done
		local tif
		for tif in "$folder"/*_rigid.tif "$folder"/*_nonrigid.tif; do
			[ -e "$tif" ] && out+="$tif"$'\n'
		done
	done <<< "$FOLDERS"
	shopt -u nullglob
	printf '%s' "$out" | sort -u
}

_clean_collect_fast_paths() {
	shopt -s nullglob
	local out=""
	while IFS= read -r folder; do
		folder="${folder%/}"
		[ -z "$folder" ] && continue
		local folder_id p tif
		folder_id=$(basename "$folder")
		for p in \
			"$folder/checkpoint" \
			"$folder/inference.h5" \
			"$folder/_fast_complete" \
			"$folder/_run_config.json" \
			"$folder/_inference_config.json"
		do
			[ -e "$p" ] && out+="$p"$'\n'
		done
		for tif in "$folder"/*_registered_*.tif; do
			[ -e "$tif" ] && out+="$tif"$'\n'
		done
		[ -e "$SCRATCH_DIR/$folder_id" ] && out+="$SCRATCH_DIR/$folder_id"$'\n'
	done <<< "$FOLDERS"
	for p in \
		"$FAST_LOG_DIR/_pipeline_status.json" \
		"$FAST_LOG_DIR/gpu_stats.csv" \
		"$FAST_LOG_DIR/ram_stats.txt"
	do
		[ -e "$p" ] && out+="$p"$'\n'
	done
	local f
	for f in "$FAST_LOG_DIR"/_pipeline_log_*.txt; do
		[ -e "$f" ] && out+="$f"$'\n'
	done
	shopt -u nullglob
	printf '%s' "$out" | sort -u
}

_clean_print_manifest() {
	local title="$1"
	local manifest="$2"
	echo ""
	echo "════════════════════════════════════════════════════════"
	echo "  $title"
	echo "  Paths that exist on disk:"
	echo "════════════════════════════════════════════════════════"
	if [ -z "$(echo "$manifest" | sed '/^$/d')" ]; then
		echo "  (none — nothing to remove)"
	else
		while IFS= read -r line; do
			[ -n "$line" ] && echo "  $line"
		done <<< "$manifest"
	fi
	echo ""
}

_clean_execute_manifest() {
	while IFS= read -r p; do
		[ -z "$p" ] && continue
		rm -rf "$p"
		echo "  deleted: $p"
	done <<< "$1"
}

_clean_count_nonempty_lines() {
	echo "$1" | sed '/^$/d' | wc -l | tr -d ' '
}

# Returns 0 if /etc/fstab already has this mount point as field 2
_fstab_has_scratch_mount() {
	local mp="$1"
	awk -v "mp=$mp" '
		/^[[:space:]]*#/ { next }
		NF < 2 { next }
		$2 == mp { found = 1 }
		END { exit found ? 0 : 1 }
	' /etc/fstab 2>/dev/null
}

# Ensure scratch_dir from pipeline_config.json is a mounted tmpfs (creates dir,
# appends /etc/fstab once, mounts). Refuses to overlay tmpfs on a non-empty
# directory that is not already the mount point — avoids hiding existing data.
_ensure_scratch_tmpfs() {
	if [ ! -f "$FAST_CONFIG" ]; then
		echo "ERROR: pipeline_config.json not found at $FAST_CONFIG"
		exit 1
	fi
	local SCRATCH_DIR
	SCRATCH_DIR=$(python3 -c "import json; print(json.load(open('$FAST_CONFIG'))['scratch_dir'])")

	if mountpoint -q "$SCRATCH_DIR" 2>/dev/null; then
		if [ ! -w "$SCRATCH_DIR" ]; then
			echo "ERROR: scratch_dir is mounted but not writable: $SCRATCH_DIR"
			exit 1
		fi
		echo "  Scratch (tmpfs): $SCRATCH_DIR (already mounted)"
		return 0
	fi

	if [ -d "$SCRATCH_DIR" ] && [ -n "$(ls -A "$SCRATCH_DIR" 2>/dev/null)" ]; then
		echo "ERROR: scratch_dir exists, is not mounted, and is not empty:"
		echo "  $SCRATCH_DIR"
		echo "  Empty it or pick a different scratch_dir in pipeline_config.json"
		echo "  before mounting tmpfs (would hide existing files)."
		exit 1
	fi

	echo "  Configuring FAST scratch tmpfs at $SCRATCH_DIR (sudo required once)..."

	_sudo_or_die() {
		if ! sudo "$@"; then
			echo ""
			echo "ERROR: sudo failed. Configure manually, then re-run PreProcess2PImage.sh:"
			echo "  sudo mkdir -p $SCRATCH_DIR"
			echo "  Add to /etc/fstab:"
			echo "    tmpfs	${SCRATCH_DIR}	tmpfs	defaults,size=${SCRATCH_TMPFS_SIZE},mode=0777	0	0"
			echo "  sudo mount ${SCRATCH_DIR}"
			exit 1
		fi
	}

	_sudo_or_die mkdir -p "$SCRATCH_DIR"

	if ! _fstab_has_scratch_mount "$SCRATCH_DIR"; then
		{
			echo ""
			echo "# schollab-pipeline FAST scratch (tmpfs)"
			echo "tmpfs	${SCRATCH_DIR}	tmpfs	defaults,size=${SCRATCH_TMPFS_SIZE},mode=0777	0	0"
		} | _sudo_or_die tee -a /etc/fstab >/dev/null
	fi

	_sudo_or_die mount "$SCRATCH_DIR"

	if ! mountpoint -q "$SCRATCH_DIR" 2>/dev/null; then
		echo "ERROR: $SCRATCH_DIR is not a mountpoint after 'sudo mount'. Check /etc/fstab and syslog."
		exit 1
	fi

	local tfile
	tfile="$SCRATCH_DIR/.schollab_scratch_writable_test.$$"
	if ! touch "$tfile" 2>/dev/null; then
		echo "ERROR: scratch_dir is not writable after mount: $SCRATCH_DIR"
		exit 1
	fi
	rm -f "$tfile"
	echo "  Scratch (tmpfs): $SCRATCH_DIR mounted and writable"
}

# Require Miniforge/conda on disk (no auto-install).
_conda_bin_or_die() {
	if [ ! -x "$CONDA_BIN" ]; then
		echo "ERROR: conda not found or not executable: $CONDA_BIN"
		echo "  Install Miniforge: https://github.com/conda-forge/miniforge"
		echo "  Or set SCHOLLAB_CONDA_ROOT (install prefix) or CONDA_BIN explicitly and re-run."
		exit 1
	fi
}

# Create/update caiman from yml; create FAST if missing; pip install FAST deps.
_setup_conda_envs() {
	local yml req
	yml="$REPO_DIR/caiman_conda_env.yml"
	req="$REPO_DIR/fast_pip_requirements.txt"
	if [ ! -f "$yml" ]; then
		echo "ERROR: $yml not found"
		exit 1
	fi
	if [ ! -f "$req" ]; then
		echo "ERROR: $req not found"
		exit 1
	fi

	echo "── Conda: caiman env ─────────────────────────────────"
	if [ -d "$SCHOLLAB_CONDA_ROOT/envs/caiman" ]; then
		echo "  Updating existing env 'caiman' from caiman_conda_env.yml ..."
		"$CONDA_BIN" env update -f "$yml" --prune
	else
		echo "  Creating env 'caiman' from caiman_conda_env.yml ..."
		"$CONDA_BIN" env create -f "$yml"
	fi

	echo "── Conda: FAST env ────────────────────────────────────"
	if [ ! -d "$SCHOLLAB_CONDA_ROOT/envs/FAST" ]; then
		echo "  Creating env 'FAST' (python 3.11) ..."
		"$CONDA_BIN" create -n FAST python=3.11 -y
	else
		echo "  Env 'FAST' already exists; skipping conda create."
	fi
	echo "  pip install (FAST env) from fast_pip_requirements.txt ..."
	"$CONDA_BIN" run -n FAST pip install -r "$req"
}

# One-shot / refresh: envs, linger, log dir, scratch tmpfs (sudo at end).
_setup_run() {
	echo "Schollab pipeline — setup (conda envs + scratch)"
	echo "  Repo:            $REPO_DIR"
	echo "  Conda prefix:    $SCHOLLAB_CONDA_ROOT"
	echo "  Conda binary:    $CONDA_BIN"
	echo ""
	_conda_bin_or_die
	_setup_conda_envs
	echo ""
	echo "── User systemd linger ────────────────────────────────"
	loginctl enable-linger "$USER"
	echo ""
	mkdir -p "$FAST_LOG_DIR"
	echo "── FAST scratch tmpfs ─────────────────────────────────"
	_ensure_scratch_tmpfs
	echo ""
	echo "Setup complete. Next: bash PreProcess2PImage.sh"
}

# ── clean_caiman: remove caiman registration artifacts ────────────────────────

_clean_caiman() {
	_resolve_clean_folders
	echo "  Scratch: $SCRATCH_DIR"
	local manifest n
	manifest=$(_clean_collect_caiman_paths)
	_clean_print_manifest "Clean CaImAn artifacts" "$manifest"
	n=$(_clean_count_nonempty_lines "$manifest")
	if [ "${n:-0}" -eq 0 ]; then
		echo "Nothing to remove."
		return 0
	fi
	if ! $CONFIRM; then
		echo "Scan only. Re-run with --confirm to delete after reviewing the list above."
		return 0
	fi
	_clean_prompt_delete "$n"
	_clean_run_unsafe_registered_check
	echo "Deleting..."
	_clean_execute_manifest "$manifest"
	echo "CaImAn clean complete."
}

# ── clean_fast: remove FAST denoising artifacts ───────────────────────────────

_clean_fast() {
	_resolve_clean_folders
	echo "  Scratch: $SCRATCH_DIR"
	echo "  Logs:    $FAST_LOG_DIR"
	local manifest n
	manifest=$(_clean_collect_fast_paths)
	_clean_print_manifest "Clean FAST artifacts" "$manifest"
	n=$(_clean_count_nonempty_lines "$manifest")
	if [ "${n:-0}" -eq 0 ]; then
		echo "Nothing to remove."
		return 0
	fi
	if ! $CONFIRM; then
		echo "Scan only. Re-run with --confirm to delete after reviewing the list above."
		return 0
	fi
	_clean_prompt_delete "$n"
	echo "Deleting..."
	_clean_execute_manifest "$manifest"
	echo "FAST clean complete."
}

# ── clean_all: CaImAn + FAST ────────────────────────────────────────────────────

_clean_all() {
	_resolve_clean_folders
	echo "  Scratch: $SCRATCH_DIR"
	echo "  Logs:    $FAST_LOG_DIR"
	local m1 m2 merged n
	m1=$(_clean_collect_caiman_paths)
	m2=$(_clean_collect_fast_paths)
	merged=$(printf '%s\n%s\n' "$m1" "$m2" | sort -u)
	_clean_print_manifest "Clean ALL (CaImAn + FAST)" "$merged"
	n=$(_clean_count_nonempty_lines "$merged")
	if [ "${n:-0}" -eq 0 ]; then
		echo "Nothing to remove."
		return 0
	fi
	if ! $CONFIRM; then
		echo "Scan only. Re-run with --confirm to delete after reviewing the list above."
		return 0
	fi
	_clean_prompt_delete "$n"
	_clean_run_unsafe_registered_check
	echo "Deleting..."
	_clean_execute_manifest "$merged"
	echo "Full clean complete."
}

# ── clean modes ───────────────────────────────────────────────────────────────

if [ "$MODE" = "clean_caiman" ]; then
	_clean_caiman
	exit 0
fi

if [ "$MODE" = "clean_fast" ]; then
	_clean_fast
	exit 0
fi

if [ "$MODE" = "clean_all" ]; then
	_clean_all
	exit 0
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

# ── setup mode ───────────────────────────────────────────────────────────────

if [ "$MODE" = "setup" ]; then
	_setup_run
	exit 0
fi

# ── start mode ────────────────────────────────────────────────────────────────

if [ ! -f "$CAIMAN_PYTHON" ]; then
	echo "ERROR: caiman python not found at $CAIMAN_PYTHON"
	echo "  Install envs (bash PreProcess2PImage.sh --setup) or set SCHOLLAB_CONDA_ROOT to your conda prefix."
	exit 1
fi

# Enable linger so user manager survives session death
loginctl enable-linger "$USER"

mkdir -p "$FAST_LOG_DIR"

# FAST scratch tmpfs — disabled: use disk-backed scratch_dir in pipeline_config.json.
# _ensure_scratch_tmpfs

echo "Starting PreProcess2PImage..."
echo "  Caiman python: $CAIMAN_PYTHON"
echo "  Repo:          $REPO_DIR"
echo ""
echo "Opening folder selection GUI..."
echo ""

# Blocks until user clicks Run; registration.py writes job JSON + launches systemd service
"$CAIMAN_PYTHON" "$REGISTRATION_SCRIPT"

echo ""
echo "Useful commands:"
echo "  Follow live output:  bash PreProcess2PImage.sh --attach"
echo "  Check status:        bash PreProcess2PImage.sh --status"
echo "  Stop pipeline:       bash PreProcess2PImage.sh --stop"
echo "  Raw journal:         journalctl --user -f -u $UNIT_NAME"
