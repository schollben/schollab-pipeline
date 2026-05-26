# PreProcess2PImages

End-to-end calcium imaging pre processing pipeline: CaImAn motion correction → FAST denoising.

## Structure

```
PreProcess2PImages/
├── caiman/                  CaImAn motion correction
│   ├── registration.py      Entry point: GUI → job file → systemd service
│   ├── registration_gui.py  wxPython folder picker + step checkboxes
│   ├── config.json          CaImAn runtime + motion-correction settings
│   ├── tif_to_h5.py         TIF stack → HDF5 conversion
│   └── h5_to_tif.py         HDF5 → TIF utility
├── fast/                    FAST denoising
│   ├── denoising.py         Entry point: reads config.json
│   ├── config.json          Data folders + hyperparameters
│   ├── datasets/            PyTorch dataset + augmentation
│   ├── models/              U-Net architecture
│   └── utils/               H5/TIFF utilities, config loader
├── tools/
│   └── scan_sessions.py     Audit recording sessions before running
├── workers/
│   └── pipeline_worker.py    Headless per-folder caiman→FAST (systemd only)
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

### FAST scratch directory (`scratch_dir` in `fast/config.json`)

FAST uses `scratch_dir` (often `/mnt/fast_tmp`) for fast intermediate I/O. On **`bash PreProcess2PImages.sh --setup`** or the first normal **`bash PreProcess2PImages.sh`** run, the script prompts for **sudo** to:

- create the mount point directory,
- append a **tmpfs** line to `/etc/fstab` (if not already present),
- run `sudo mount` so it matches reboot behavior.

Tune the RAM cap by editing `SCRATCH_TMPFS_SIZE` at the top of `PreProcess2PImages.sh`. If `scratch_dir` already exists as a **non-empty** directory and is not mounted, the script exits rather than overlay tmpfs and hide files.

**Note:** `bash PreProcess2PImages.sh` (GUI start) may have tmpfs setup commented out in favor of a disk-backed `scratch_dir`; **`--setup`** still configures tmpfs if you use that workflow. Check comments at the bottom of [`PreProcess2PImages.sh`](PreProcess2PImages.sh).

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

## Alternate launchers

The GUI produces a job file at `/tmp/pipeline_job.json`:
```json
{
  "sessions": ["/mnt/bigdata/BRUKER/TSeries-001", "..."],
  "process_selections": [[true, false, true, false], ...]
}
```
Any tool that writes this file can trigger the pipeline — the GUI is one option,
not a requirement. A CLI or web UI can produce the same file.
