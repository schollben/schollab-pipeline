#!/usr/bin/env bash
# clean_run.sh
# Wipes all pipeline artifacts so the next run starts completely fresh.
#
# Deletes per-folder (permanent drive):
#   checkpoint/          model weights + config
#   inference.h5         denoised output
#   _fast_complete       completion sentinel
#   _run_config.json     training config copy
#   _inference_config.json  inference config copy
#   *.tif                example result TIFFs copied to root
#
# Deletes scratch (tmpfs):
#   /mnt/fast_tmp/<folder>/   registered/, training/, result/
#
# Deletes logs:
#   logs/_pipeline_status.json
#   logs/_pipeline_log_*.txt
#   logs/gpu_stats.csv
#   logs/ram_stats.txt
#
# Usage:
#   bash clean_run.sh              # dry run — shows what would be deleted
#   bash clean_run.sh --confirm    # actually deletes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/pipeline_config.json"
DRY_RUN=true

if [[ "${1:-}" == "--confirm" ]]; then
    DRY_RUN=false
fi

# ── helpers ──────────────────────────────────────────────────────────────────

delete() {
    local target="$1"
    if [ -e "$target" ] || [ -d "$target" ]; then
        if $DRY_RUN; then
            echo "  [DRY-RUN] would delete: $target"
        else
            rm -rf "$target"
            echo "  deleted: $target"
        fi
    fi
}

# ── read config ───────────────────────────────────────────────────────────────

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: pipeline_config.json not found at $CONFIG"
    exit 1
fi

SCRATCH_DIR=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['scratch_dir'])")
FAST_DIR=$(python3    -c "import json; c=json.load(open('$CONFIG')); print(c['fast_dir'])")
FOLDERS=$(python3     -c "import json; c=json.load(open('$CONFIG')); [print(f) for f in c['data_folders']]")

# ── summary ───────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  FAST Pipeline Clean Start"
if $DRY_RUN; then
    echo "  MODE: DRY RUN  (pass --confirm to actually delete)"
else
    echo "  MODE: DELETING"
fi
echo "  Config:  $CONFIG"
echo "  Scratch: $SCRATCH_DIR"
echo "  Logs:    $FAST_DIR/logs"
echo "════════════════════════════════════════════════════════"
echo ""

# ── per-folder cleanup ────────────────────────────────────────────────────────

while IFS= read -r folder; do
    folder="${folder%/}"   # strip trailing slash
    echo "Folder: $folder"

    # Permanent drive artifacts
    delete "$folder/checkpoint"
    delete "$folder/inference.h5"
    delete "$folder/_fast_complete"
    delete "$folder/_run_config.json"
    delete "$folder/_inference_config.json"

    # Example result TIFFs copied to root (named *_registered_*.tif)
    for tif in "$folder"/*_registered_*.tif; do
        delete "$tif"
    done

    # Scratch (tmpfs) for this folder
    folder_id=$(basename "$folder")
    delete "$SCRATCH_DIR/$folder_id"

    echo ""
done <<< "$FOLDERS"

# ── log cleanup ───────────────────────────────────────────────────────────────

echo "Logs:"
delete "$FAST_DIR/logs/_pipeline_status.json"
for f in "$FAST_DIR"/logs/_pipeline_log_*.txt; do
    delete "$f"
done
delete "$FAST_DIR/logs/gpu_stats.csv"
delete "$FAST_DIR/logs/ram_stats.txt"
echo ""

# ── done ──────────────────────────────────────────────────────────────────────

if $DRY_RUN; then
    echo "Dry run complete. Run with --confirm to delete for real:"
    echo ""
    echo "  bash $0 --confirm"
else
    echo "Clean complete. Ready for a fresh run:"
    echo ""
    echo "  bash pipeline.sh   # from repo root"
fi
echo ""