---
title: SEM Contrast Enhancement
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app/streamlit_app.py
pinned: false
---

# SEM Contrast Enhancement Settings

This repository condenses an SEM-preprocessing workflow into a single,
reproducible package plus a Streamlit demo. The goal is to showcase some algorithms, like Frangi, dirt inpainting, CLAHE/bilateral, DoG fusion, to preprocess SEM images of the microstructures as an initial step to feed a large amount of SEM images into a machine learning pipeline.

---
## 1. Architecture Overview

| Layer | Description |
| --- | --- |
| `sem_line_enhancer/loader.py` | Normalizes TIFF/PNG/JPG inputs to float32 `[0,1]`, removing metadata noise. |
| `sem_line_enhancer/enhancers.py` | Implements the four core algorithms:<br>1) **Frangi** multi-scale ridge enhancement.<br>2) **Difference-of-Gaussians (DoG)** for mid-frequency contrast.<br>3) **CLAHE + bilateral filtering** for grains/base texture.<br>4) **Dirt blob detection & Telea inpainting** (regionprops + OpenCV). |
| `sem_line_enhancer/pipeline.py` | Orchestrates the Std-aware preprocessing pipeline and dual-path outputs (`lines`, `base`, `fused`). |
| `sem_line_enhancer/presets.py` | Stores experiment-backed hyperparameter sets for different SEM classes. |
| `sem_line_enhancer/cli.py` | CLI entrypoint for preprocessing (`.npy`) + PNG export, with optional MLflow logging. |
| `app/streamlit_app.py` | Upload/sample interface showing Original + Lines/Base/Fused panels and downloads. |

Two presets are shipped:
- **`lines`** – tuned for something like ferroelastic line or other line-shape features visibility (Frangi+DoG-heavy, stronger dirt removal).
- **`boundaries`** – tuned for grain-only micrographs (lighter Frangi weights, CLAHE emphasis).

Sample PNGs (`examples/`) match both classes so the app/CI can run without raw data.

---
## 2. Getting Started

```bash
# Install with dev extras (ruff/pytest already listed)
pip install -e .[dev]

# Quick CLI smoke test (line-focused preset)
python -m sem_line_enhancer.cli preprocess \
  --input examples \
  --output tmp \
  --limit 1 \
  --cpu-only \
  --preset lines

# Export the generated .npy files to PNG
python -m sem_line_enhancer.cli export \
  --input tmp \
  --output tmp_png \
  --types lines base fused

# Launch Streamlit app
streamlit run app/streamlit_app.py
```

The app lets users pick a preset, load one of the sample images (either the
bundled PNGs under `examples/` or the synthetic fallbacks shipped inside the app
for Spaces), or upload their own image. Results can be downloaded as `.npy`
arrays for downstream SAM/MatSAM workflows, which have not been described here.

---
## 3. Experimentation & MLOps Hooks

### Grid Search
- Script: `gridsearch_single_preprocessing.py`
- Adjustable grids near the top (`FRANGI_*`, `DIRT_*`, `CLAHE_*`, `BILAT_*`). Two example experiments are documented in `docs/LOCAL_TESTING.md`:
  1. **Experiment 1** – broad Frangi/CLAHE sweep (lines + grains).
  2. **Experiment 2** – dirt-focused sweep (grain-only SEMs).
- Outputs land in `data/diagnostics/` (`gridsearch_stats.csv`, best-result panels, `best results.txt`).
- Promote winners by editing `sem_line_enhancer/presets.py`, then rerun the CLI/app to verify visually.

### MLflow (optional)
CLI flag `--mlflow` logs preset name, hyperparameters, and metadata to the
server defined in `MLFLOW_TRACKING_URI`. Example:
```bash
python -m sem_line_enhancer.cli preprocess \
  --input data/raw \
  --output data/preprocessed \
  --preset boundaries \
  --mlflow --mlflow-run-name boundary_sweep
```

### Continuous Integration
`.github/workflows/ci.yaml` runs on every push/PR:
1. Checks out the repo.
2. Installs the package (`pip install -e .[dev] streamlit opencv-python-headless`).
3. Executes a CLI smoke test on the sample data.

This keeps the app/demo reproducible without requiring private datasets.

---
## 4. Deployment Notes

### Local
- Use the CLI for batch preprocessing → `.npy` or PNG outputs.
- Run Streamlit locally (`streamlit run app/streamlit_app.py`) inside the Conda env (`ymno3_gpu`). No external services required.

### GitHub
- Repo name: `SEM-contrast-enhancement-settings-`. Push changes, CI runs automatically.
- `.gitignore` excludes raw data, diagnostics, and caches so only code/config/docs live in source control.

### Hugging Face Space (Streamlit)
1. Create a new HF Space (Streamlit template).
2. Point it to this repo or upload the files (ensure dependencies include `streamlit`, `opencv-python-headless`, `numpy`, etc.).
3. Optional secrets (MLflow tracking URI, AWS creds) can be stored via the HF Secrets tab—no tokens are committed.

Once deployed, the HF space mirrors the local app: preset dropdown, sample images, upload, download outputs.

---
## 5. Repository Layout
```
.
├── sem_line_enhancer/
│   ├── cli.py                # CLI entrypoint (preprocess/export/presets)
│   ├── loader.py             # SEMImageLoader
│   ├── pipeline.py           # SEMPreprocessor with dual outputs
│   ├── enhancers.py          # Frangi, DoG, dirt removal, etc.
│   ├── presets.py            # `lines` + `boundaries` parameter sets
│   └── ...
├── app/
│   ├── streamlit_app.py      # UI with preset/sample support
│   └── README.md
├── docs/LOCAL_TESTING.md     # Command cookbook + experiment notes
├── gridsearch_single_preprocessing.py
├── resize_images.py          # Optional 512×512 tiling utility
├── examples/                 # Sample PNGs used by app/CI
├── .github/workflows/ci.yaml
├── requirements.txt
└── README.md (this file)
```

---

Feel free to fork or adapt to your own SEM datasets—the repo is intentionally modular and lightweight so students, or PhDs, can inspect the full workflow. Enjoy! 🎯
