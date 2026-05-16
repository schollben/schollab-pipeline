# schollab-pipeline

End-to-end calcium imaging pipeline: CaImAn motion correction → FAST denoising.

## Structure

```
schollab-pipeline/
├── caiman/                  CaImAn motion correction
│   ├── registration.py      Entry point: GUI → job file → systemd service
│   ├── registration_gui.py  wxPython folder picker + step checkboxes
│   ├── tif_to_h5.py         TIF stack → HDF5 conversion
│   └── h5_to_tif.py         HDF5 → TIF utility
├── fast/                    FAST denoising
│   ├── denoising.py         Entry point: reads pipeline_config.json
│   ├── pipeline_config.json Data folders + hyperparameters
│   ├── datasets/            PyTorch dataset + augmentation
│   ├── models/              U-Net architecture
│   └── utils/               H5/TIFF utilities, config loader
├── tools/
│   └── scan_sessions.py     Audit recording sessions before running
├── pipeline_worker.py       Headless per-folder caiman → FAST loop
├── pipeline.sh              Launcher: setup + GUI + status/attach/stop
├── caiman_conda_env.yml     CaImAn conda environment spec
└── fast_pip_requirements.txt  FAST pip requirements
```

## Environments

Two conda environments — kept separate due to dependency conflicts:

| Env | Used for | Python path |
|-----|----------|-------------|
| `caiman` | Motion correction | `~/miniforge3/envs/caiman/bin/python` |
| `FAST` | Denoising | `~/miniforge3/envs/FAST/bin/python` |

Set up:
```bash
conda env create -f caiman_conda_env.yml
pip install -r fast_pip_requirements.txt  # inside FAST env
```

## Running the pipeline

```bash
bash pipeline.sh                        # open GUI, launch pipeline
bash pipeline.sh --attach               # follow live output
bash pipeline.sh --status               # service status + last log lines
bash pipeline.sh --stop                 # stop a running pipeline

# Cleanup
bash pipeline.sh --clean_caiman         # dry run: show caiman artifacts
bash pipeline.sh --clean_caiman --confirm   # delete caiman artifacts
bash pipeline.sh --clean_fast           # dry run: show FAST artifacts
bash pipeline.sh --clean_fast --confirm     # delete FAST artifacts
bash pipeline.sh --clean_all            # dry run: show all artifacts
bash pipeline.sh --clean_all --confirm      # delete everything
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
