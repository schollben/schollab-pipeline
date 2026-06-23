# PreProcess2PImages

End-to-end calcium imaging pre processing pipeline: CaImAn motion correction → FAST denoising.

## Structure

```
PreProcess2PImages/
├── caiman/                  CaImAn motion correction
│   ├── registration.py      Entry point: GUI → job file → systemd service
│   ├── registration_gui.py  wxPython folder picker + step checkboxes
│   ├── config.json          CaImAn runtime settings
│   ├── tif_to_h5.py         TIF stack → HDF5 conversion
│   └── h5_to_tif.py         HDF5 → TIF utility
├── fast/                    FAST denoising
│   ├── denoising.py         Entry point: reads config.json
│   ├── config.json          Data folders + hyperparameters
│   ├── datasets/            PyTorch dataset + augmentation
│   ├── models/              U-Net architecture
│   └── utils/               H5/TIFF utilities, config loader
├── tools/
│   ├── scan_sessions.py     Audit recording sessions before running
│   └── schedule_batch.py    CLI: schedule / list / cancel batch jobs
├── workers/
│   ├── pipeline_worker.py    Headless per-folder caiman→FAST (systemd)
│   └── pipeline_dispatcher.py  Scheduled batch entry (lock + defer)
├── PreProcess2PImages.sh              Launcher: --setup, GUI + status/attach/stop + clean
├── caiman_conda_env.yml     CaImAn conda environment spec
└── fast_pip_requirements.txt  FAST pip requirements
```

## Environments

**Prerequisite:** [Miniforge](https://github.com/conda-forge/miniforge), Miniconda, or another conda install whose prefix contains `bin/conda` and `envs/`.

- **Default prefix** is `~/miniforge3` (paths like `$HOME/miniforge3/envs/caiman/bin/python`).
- On a machine where conda lives under **`~/miniconda3`** (or anywhere else), set:

  ```bash
  export SCHOLLAB_CONDA_ROOT="$HOME/miniconda3"
  ```

  before `bash PreProcess2PImages.sh` / the GUI. `PreProcess2PImages.sh` derives **`CONDA_BIN`** from `SCHOLLAB_CONDA_ROOT` unless you override **`CONDA_BIN`** directly. The GUI passes **`SCHOLLAB_CONDA_ROOT`** into the systemd worker so FAST uses the same prefix.

Two conda environments — kept separate due to dependency conflicts:

| Env | Used for | Python path (default prefix) |
|-----|----------|------------------------------|
| `caiman` | Motion correction | `$SCHOLLAB_CONDA_ROOT/envs/caiman/bin/python` |
| `FAST` | Denoising | `$SCHOLLAB_CONDA_ROOT/envs/FAST/bin/python` |

**Preferred setup** (creates/updates both envs, enables systemd linger, creates log dir, configures scratch tmpfs — may prompt for **sudo** at the end):

```bash
bash PreProcess2PImages.sh --setup
```

Manual fallback (equivalent ingredients):

```bash
conda env create -f caiman_conda_env.yml   # or: conda env update -f caiman_conda_env.yml --prune
conda create -n FAST python=3.11 -y        # once
conda run -n FAST pip install -r fast_pip_requirements.txt
```

### CaImAn CPU controls (`caiman/config.json`)

CaImAn runtime settings live in [`caiman/config.json`](caiman/config.json). The most important CPU controls are:

- `n_processes`: CaImAn worker process count (default `2`; increase only if RAM/swap stay stable).
- `threads`: default numerical-library thread caps applied before NumPy/CaImAn imports.

For one-off runs, override process count from the shell:

```bash
CAIMAN_N_PROCESSES=6 bash PreProcess2PImages.sh
```

CaImAn thread caps such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` respect existing environment values; otherwise the values in `caiman/config.json` are used.

Only `n_processes` and `threads` are active today; sample-output and motion-parameter config entries are reserved for follow-up cleanup.

### FAST CPU controls (`fast/config.json`)

FAST gets its own subprocess environment from [`fast/config.json`](fast/config.json), so its thread caps can differ from CaImAn:

- `threads`: numerical-library thread caps passed only to the FAST subprocess.
- `num_workers`: PyTorch data-loading workers; this is separate from BLAS/OpenMP thread caps.
- `tiff_chunk_size`: frames per TIFF stack used for FAST inference. Lower this if `systemd-oomd` kills FAST during `step3_inference`.
- `h5_write_batch_frames`: frames loaded per batch when merging FAST result TIFFs into `inference.h5` in Step 4 (default `128`).

The worker logs the FAST thread environment before launching `fast/denoising.py`, which makes `journalctl --user -u schollab-PreProcess2PImages` useful for checking the active CPU settings.

### FAST scratch directory (`scratch_dir` in `fast/config.json`)

FAST expands `$HOME` / `~` in `fast_dir` and `scratch_dir`, so the default config follows the current Linux user instead of a hardcoded lab account. Override machine-specific locations when needed:

```bash
export FAST_DIR="$HOME/Documents/FAST"
export FAST_SCRATCH_DIR="$HOME/Documents/scratch"
```

FAST uses `scratch_dir` (or `FAST_SCRATCH_DIR`) for fast intermediate I/O. On **`bash PreProcess2PImages.sh --setup`** or the first normal **`bash PreProcess2PImages.sh`** run, the script prompts for **sudo** to:

- create the mount point directory,
- append a **tmpfs** line to `/etc/fstab` (if not already present),
- run `sudo mount` so it matches reboot behavior.

Tune the RAM cap by editing `SCRATCH_TMPFS_SIZE` at the top of `PreProcess2PImages.sh`. If `scratch_dir` already exists as a **non-empty** directory and is not mounted, the script exits rather than overlay tmpfs and hide files.

**Note:** `bash PreProcess2PImages.sh` (GUI start) may have tmpfs setup commented out in favor of a disk-backed `scratch_dir`; **`--setup`** still configures tmpfs if you use that workflow. Check comments at the bottom of [`PreProcess2PImages.sh`](PreProcess2PImages.sh).

## Performance tuning guide

Choose settings from dataset size, host RAM/CPU/GPU, and how much swap pressure you can tolerate. Edit [`caiman/config.json`](caiman/config.json) and [`fast/config.json`](fast/config.json), or use the runtime overrides below for one-off runs.

### Quick machine check

```bash
free -h          # RAM + swap
nproc            # logical CPUs
nvidia-smi       # GPU (FAST needs CUDA)
df -h /mnt/bigdata "$HOME/Documents/scratch"
```

| Host RAM | Typical safe starting point |
|---|---|
| 32–64 GB | Profile A (`n_processes=2`, `h5_write_batch_frames=128`) |
| 64–128 GB | Profile B (`n_processes=4`, `h5_write_batch_frames=256`) |
| 128+ GB | Profile C only after stable test runs |

Keep `threads.* = 1` in both configs unless you deliberately want more BLAS parallelism per process (usually increases RAM and swap).

### File-size tiers (rule of thumb)

Assume 512×512 `uint16` frames unless your data differs.

| Dataset shape | Approx frames | CaImAn (2 workers) | FAST (GPU) | Notes |
|---|---|---|---|---|
| Small (1–5 TIFFs, ~1k frames each) | ~1k–5k | ~5–20 min | ~10–40 min | Use for config validation |
| Medium (10–30 TIFFs, ~1k frames) | ~10k–30k | ~20–50 min | ~30–90 min | Typical lab session |
| Large (40+ TIFFs, ~1k frames) | ~40k+ | ~45–120 min | ~1–4+ h | Watch swap; prefer Profile A first |

Memory intuition: one 1000-frame 512×512 stack is ~0.5 GB when fully loaded. Peak RAM spikes usually come from full-stack loads (CaImAn rewrite, FAST Step 3/4), not from raw TIFF count alone. Lower `tiff_chunk_size` and `h5_write_batch_frames` before raising `n_processes`.

### Recommended profiles

**Profile A — safe / low memory** (swap fills, `systemd-oomd` kills, or first run on a new host)

| Setting | Value |
|---|---|
| `caiman` → `n_processes` | `2` |
| `fast` → `tiff_chunk_size` | `1000` (try `500` if Step 3 OOMs) |
| `fast` → `h5_write_batch_frames` | `128` (try `64` if Step 4 OOMs) |
| `fast` → `num_workers` | `8` |
| `fast` → `epochs` | `25` |

**Profile B — balanced** (64–128 GB RAM, stable swap)

| Setting | Value |
|---|---|
| `caiman` → `n_processes` | `4` |
| `fast` → `tiff_chunk_size` | `1000` |
| `fast` → `h5_write_batch_frames` | `256` |
| `fast` → `num_workers` | `16` |

**Profile C — high throughput** (128+ GB RAM, low swap on two test folders)

| Setting | Value |
|---|---|
| `caiman` → `n_processes` | `6–8` |
| `fast` → `h5_write_batch_frames` | `512` |
| `fast` → `num_workers` | `16` |

### Scratch and I/O

| Situation | Recommendation |
|---|---|
| Fast local SSD/NVMe for scratch | Set `scratch_dir` to that path, or `export FAST_SCRATCH_DIR=...` |
| tmpfs via `--setup` | Good for intermediate TIFF/H5 I/O; size is capped by `SCRATCH_TMPFS_SIZE` in `PreProcess2PImages.sh` — do not make tmpfs larger than RAM you can spare |
| Scratch on same disk as huge datasets | Slower but safe; prefer smaller `h5_write_batch_frames` if I/O is the bottleneck |
| Rerun with valid checkpoint | `"skip_training": true` in `fast/config.json` saves training time |

### Config knobs (summary)

| Knob | Where | Effect | Increase when… | Decrease when… |
|---|---|---|---|---|
| `n_processes` / `CAIMAN_N_PROCESSES` | CaImAn | Parallel motion correction | Large host, low swap | Swap pressure, OOM |
| `save_sample` / `sample_frames` | CaImAn | Preview TIFF export | Debugging | Extra disk/time |
| `tiff_chunk_size` | FAST | Frames per inference stack | — | Step 3 OOM or high swap |
| `h5_write_batch_frames` | FAST | Step 4 merge batch size | Stable host, fast Step 4 | Step 4 OOM after inference |
| `num_workers` | FAST | DataLoader parallelism | Large training sets | Memory pressure |
| `epochs` | FAST | Training length | Quality matters more than time | Quick iteration |
| `skip_training` | FAST | Skip training, reuse checkpoint | Valid checkpoint for this data | Model mismatch risk |
| `threads` (both configs) | CaImAn / FAST | BLAS thread caps per subprocess | Rarely | Almost always keep at `1` |

### Runtime overrides (no file edit)

```bash
export SCHOLLAB_CONDA_ROOT="$HOME/miniconda3"   # if not using ~/miniforge3
export CAIMAN_N_PROCESSES=2
export FAST_DIR="$HOME/Documents/FAST"
export FAST_SCRATCH_DIR="$HOME/Documents/scratch"
```

### Symptoms → what to change

| Symptom | Likely stage | Try first |
|---|---|---|
| Swap hits 100% during CaImAn | Motion correction / H5 rewrite | `CAIMAN_N_PROCESSES=2`, keep `threads=1` |
| `oomd` during FAST inference | Step 3 | Lower `tiff_chunk_size` (e.g. `500`) |
| Inference finishes, then OOM | Step 4 | Lower `h5_write_batch_frames` (e.g. `64`) |
| Run completes but very slow | I/O or oversubscribed CPU | Check scratch path; avoid raising both `n_processes` and `num_workers` at once |
| Need faster rerun on same data | Training | `skip_training: true` if checkpoint is valid |

### Logs and completion

```bash
bash PreProcess2PImages.sh --attach
journalctl --user -u schollab-PreProcess2PImages -f
```

Healthy markers: CaImAn `motion correction starting/finished`, FAST `START`/`DONE step4_h5_export`, `Written: .../_fast_complete`, `pipeline_worker: all folders complete`. FAST `print` output can appear out of order in the journal due to buffering — use stage log lines above for progress.

**Timing:** When a run ends, systemd prints `Consumed … CPU time`. That is **total core-seconds** across all CaImAn worker processes and FAST subprocesses — not wall clock. A ~1.5 h run with `n_processes=16` can legitimately show ~5 h CPU. Use `bash PreProcess2PImages.sh --status` for **wall elapsed** (from `/tmp/pipeline_run_<run_id>.timing.json`).

### Tuning workflow

1. Pick one representative folder (medium size if possible).
2. Run Profile A with `--attach`.
3. If swap stays manageable, step up to Profile B on the next folder.
4. Only use Profile C after two consecutive successful runs with low swap.

### Cleanup (`--clean_caiman`, `--clean_fast`, `--clean_all`)

Each mode **scans disk first**, prints a **single manifest** of paths that exist and would be removed, then exits unless you also pass **`--confirm`**.

- **`--confirm`**: after the manifest, you get a **`Delete N path(s)? [y/N]`** prompt (needs a TTY). **`--yes`** skips that prompt for scripting (unsafe-folder rules still require a TTY and typed `yes` when applicable).
- **Folder list** (precedence):
  1. Paths after **`--`**: `bash PreProcess2PImages.sh --clean_caiman -- /data/session1 /data/session2`
  2. **`--from-job`**: read `sessions` from **`/tmp/pipeline_job.json`** (GUI job file); override path with env **`JOB_FILE`** if needed.
  3. Else **`data_folders`** in [`fast/config.json`](fast/config.json) (a **WARNING** is printed — may not match GUI-selected folders).

**`scratch_dir`** for FAST scratch subdirs is always taken from `fast/config.json`, even when folders come from the job file.

| Mode | Removes (per folder, when present) | Does **not** remove |
|------|-------------------------------------|---------------------|
| **clean_caiman** | `unregistered.h5`, `registered.h5`, rigid/nonrigid shift CSVs, `*_rigid.tif` / `*_nonrigid.tif` | Raw acquisition TIFFs (`TSeries_*`, `file_*`, …) |
| **clean_fast** | `checkpoint/`, `inference.h5`, `_fast_complete`, `_run_config.json`, `_inference_config.json`, example `*_registered_*.tif`, scratch subdir named like the session folder, shared FAST log files under `fast/logs/` | Raw TIFFs / CaImAn H5s |
| **clean_all** | Union of the above (one manifest) | Same |

If a folder has **`registered.h5`** but **no acquisition TIFs** (previews excluded), you must type **`yes`** before deletion proceeds.

## Running the pipeline

The registration GUI pre-checks **TIFs→.H5** and **First Rigid** by default for each folder so motion correction runs and `registered.h5` exists for FAST if you leave defaults; adjust the columns per folder as needed.

**Skip CaImAn (FAST only):** check this box at the top of the GUI to grey out all CaImAn columns and run denoising only. Each folder must already have `registered.h5`. Remove `_fast_complete` to force a FAST re-run. The GUI warns if any selected folder lacks `registered.h5`.

```bash
bash PreProcess2PImages.sh --setup                 # conda envs + linger + scratch tmpfs (new machine)
bash PreProcess2PImages.sh                        # open GUI, launch pipeline
bash PreProcess2PImages.sh --attach               # follow live output
bash PreProcess2PImages.sh --status               # service status + last log lines
bash PreProcess2PImages.sh --stop                 # stop a running pipeline

# Cleanup (scan manifest, then optional delete)
bash PreProcess2PImages.sh --clean_caiman
bash PreProcess2PImages.sh --clean_caiman --confirm              # interactive y/N after manifest
bash PreProcess2PImages.sh --clean_caiman --confirm --yes       # skip y/N (scripts); unsafe prompt still needs TTY
bash PreProcess2PImages.sh --clean_caiman --from-job --confirm --yes
bash PreProcess2PImages.sh --clean_all --confirm -- /path/to/Exp1 /path/to/Exp2

bash PreProcess2PImages.sh --clean_fast
bash PreProcess2PImages.sh --clean_fast --confirm
bash PreProcess2PImages.sh --clean_all --confirm
```

The pipeline runs as a systemd user service — it survives display/GDM crashes.

## Per-folder flow

For each selected folder:
1. CaImAn: TIF stacks → `unregistered.h5` → motion correction → `registered.h5`
2. FAST: reads `registered.h5` → trains U-Net → inference → `inference.h5` + `_fast_complete`

## Why folders are skipped (and when)

The pipeline is designed to continue to the next folder when one folder is incomplete or invalid. This is intentional so one bad session does not block all queued sessions.

### CaImAn-side skip conditions

- **No source acquisition TIFFs while `TIFs→H5` is checked:** CaImAn prints `Skipping TIFs→H5 step — registered.h5 left untouched.` and does not overwrite existing H5 files.
- **`TIFs→H5` checked but no motion step checked:** only `unregistered.h5` is produced; CaImAn warns that FAST will skip because `registered.h5` is never created.
- **CaImAn raises an exception for a folder:** worker logs `Skipping FAST for this folder and continuing.` and moves to the next folder.

### Worker gate before FAST

After CaImAn finishes a folder, the worker checks for `registered.h5`.

- **If `registered.h5` is missing:** worker logs `[FAST] Skipping — registered.h5 not found` and skips FAST for that folder.
- **Why this happens most often:** only `TIFs→H5` was selected (no First Rigid / Addl. Rigid / NoRMCorre), or CaImAn failed before writing `registered.h5`.

### FAST auto-skip / resume logic (per folder)

Inside `fast/denoising.py`, `process_folder()` uses this order:

1. **`_fast_complete` exists:** folder is considered fully complete, FAST skips entirely (`SKIPPING (complete)`).
2. **Checkpoint config exists:** FAST skips Steps 1-2 (TIFF export + training) and resumes from inference.
3. **Checkpoint exists but `registered/` TIFF folder is missing or empty:** FAST re-runs Step 1 only, then continues with inference/export.
4. **No checkpoint:** FAST runs full pipeline.

### Practical meaning

- `_fast_complete` is the final success sentinel; if present, reruns are intentionally skipped.
- Removing `_fast_complete` allows FAST to run the folder again (resume behavior depends on checkpoint files).
- To avoid accidental skips, keep at least one motion-correction column enabled in the GUI defaults.

## Pipeline run logs

Each pipeline run (immediate or scheduled) writes three files under **`log/`** at the repo root, sharing one timestamp from `run_id` (e.g. `20260528-143012`):

| File | Contents |
|------|----------|
| `log/{ts}.log` | **Combined summary** — one block per data folder (CaImAn + FAST ops, artifacts, wall time, OVERALL line) |
| `log/{ts}.verbose.log` | Full stdout/stderr tee (journal mirror) |
| `log/{ts}.fast.log` | FAST step detail (memory stats, tracebacks) |

`bash PreProcess2PImages.sh --status` tails the latest summary log. Standalone `python fast/denoising.py` (outside the worker) still defaults to `fast/logs/_pipeline_log_*.txt`.

## Scheduled batch runs

Optional feature — **Run** in the GUI behaves exactly as before (immediate systemd launch). Scheduling adds batch metadata and the same `log/` files as immediate runs.

Queue multiple folders (with per-folder CaImAn step checkboxes) to run **sequentially** starting at a chosen time.

Immediate runs and hand-written job files need `sessions`, `process_selections`, `skip_caiman`, `run_id`, `unit_name`. Log paths are added automatically at launch if omitted.

### GUI

1. Pick folders and step checkboxes as usual (**multiple folders in one picker session** — each gets its own row).
2. Click **Run** for immediate start (unchanged behavior).
3. *Optional:* set **Start at** under “Optional: schedule batch”, then click **Schedule batch**.

### CLI

```bash
# After GUI wrote /tmp/pipeline_job.json, or hand-authored job JSON:
python tools/schedule_batch.py --job /tmp/pipeline_job.json --at "2026-06-19T02:00:00"

bash PreProcess2PImages.sh --schedule-at "2026-06-19T02:00:00"   # uses /tmp/pipeline_job.json
bash PreProcess2PImages.sh --list-scheduled
bash PreProcess2PImages.sh --cancel-scheduled 20260619-020000-a1b2
bash PreProcess2PImages.sh --status   # includes scheduled batch list
```

Run immediately from a job file:

```bash
python tools/schedule_batch.py --job /path/to/batch.json --now
```

### Overlap policy

If a batch is still running when the next scheduled batch fires, the new batch is **deferred 5 minutes** and retried until the machine is free (global lock at `/tmp/pipeline_batch.lock`).

Persistent job copies live under `{fast_dir}/scheduled_jobs/` so timers survive reboot (requires `loginctl enable-linger` from `--setup`).

---

## Alternate launchers

The GUI produces a job file at `/tmp/pipeline_job.json`.

**Immediate run (legacy / default):**
```json
{
  "sessions": ["/mnt/bigdata/BRUKER/TSeries-001", "..."],
  "process_selections": [[true, false, true, false], ...],
  "skip_caiman": false,
  "run_id": "20250618143000",
  "unit_name": "schollab-PreProcess2PImages-20250618143000"
}
```

**Scheduled batch (additional fields):**
```json
{
  "batch_id": "20260619-020000-a1b2",
  "sessions": ["/mnt/bigdata/BRUKER/TSeries-001", "..."],
  "process_selections": [[true, false, true, false], ...],
  "skip_caiman": false,
  "run_id": "20250618143000",
  "unit_name": "schollab-PreProcess2PImages-20250618143000",
  "run_log_path": "/path/to/schollab-pipeline/log/20250618-143000.log",
  "verbose_log_path": "/path/to/schollab-pipeline/log/20250618-143000.verbose.log",
  "fast_log_path": "/path/to/schollab-pipeline/log/20250618-143000.fast.log",
  "batch_log_path": "/path/to/schollab-pipeline/log/20250618-143000.verbose.log",
  "scheduled_at": "2026-06-19T02:00:00"
}
```
Set `"skip_caiman": true` (or pass `--skip-caiman` to the worker) to skip CaImAn and run FAST only when `registered.h5` already exists:
```bash
python workers/pipeline_worker.py /tmp/pipeline_job.json --skip-caiman
```
Any tool that writes this file can trigger the pipeline — the GUI is one option,
not a requirement. A CLI or web UI can produce the same file.
