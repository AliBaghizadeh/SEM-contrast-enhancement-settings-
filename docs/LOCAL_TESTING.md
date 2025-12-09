## Local smoke test (current canonical config)

```
python -m sem_line_enhancer.cli preprocess ^
  --input data ^
  --output data/preprocessed_test ^
  --limit 2 ^
  --cpu-only ^
  --preset lines
```

- Processes the first two `.tif` files in `data/` (49 examples provided in this repo).
- Writes `_lines.npy`, `_base.npy`, `_fused.npy` per image into `data/preprocessed_test/`.
- GPU kernels (CuPy / OpenCL) are attempted by default; `--cpu-only` avoids CUDA/NVRTC issues.

## With MLflow logging (same preset)

```
python -m sem_line_enhancer.cli preprocess ^
  --input data ^
  --output data/preprocessed_test ^
  --mlflow --mlflow-run-name local_debug ^
  --cpu-only ^
  --preset boundaries
```

Set `MLFLOW_TRACKING_URI` to your EC2-tracked server before running.

## Export `.npy` outputs to PNG (lines/base/fused)

```
python -m sem_line_enhancer.cli export ^
  --input data/preprocessed_test ^
  --output data/preprocessed_png ^
  --types lines base fused ^
  --verbose
```

Generates grayscale PNGs for quick inspection. Adjust `--types` if you only want
`lines`, `base`, or `fused` images. Use the exported PNGs to visually verify
line contrast, grain clarity, and dirt removal before committing changes.

## Experiments & tuning workflow

### Experiment 1 – Broad Frangi/CLAHE sweep (completed)

- `FRANGI_SCALES_GRID = [(0.5, 1.0, 2.0), (0.3, 0.7, 1.5), (0.2, 0.5, 0.8)]`
- `FRANGI_ALPHA_GRID = [0.2, 0.35, 0.45, 0.55]`
- `DIRT_THRESHOLD_GRID = [0.12, 0.15]`, `DIRT_MAX_SIZE_GRID = [30, 40]`
- `CLAHE_CLIP_GRID = [15, 25]`, `CLAHE_TILE_GRID = [4, 8]`
- **Finding:** best configs cluster around `(0.5, 1.0, 2.0)` with `alpha=0.2`, CLAHE clip 25 / tile 4, dirt threshold ≈0.12–0.15. These values seeded `PIPELINE_PARAMS` in `sem_line_enhancer/cli.py`.

### Experiment 2 – Dirt-focused sweep (current)

- Frangi locked to `(0.5, 1.0, 2.0)` with `alpha=0.2`.
- `DIRT_THRESHOLD_GRID = [0.05, 0.10, 0.12, 0.15, 0.20]`
- `DIRT_MAX_SIZE_GRID = [10, 15, 20, 30, 40]`
- CLAHE / bilateral fixed at the winning values from Experiment 1.
- **Goal:** Find a threshold + max-size pair that removes bright dirt blobs without harming the line map.

### Running the grid search

```
python gridsearch_single_preprocessing.py
```

- Edit the grids near the top of the script before each experiment.
- Outputs land in `data/diagnostics/`:
  * `gridsearch_stats.csv` – every evaluated configuration.
  * `gridsearch_panels/best_results/` – best-per-image `.npy` arrays and diagnostic PNGs.
  * `best results.txt` – manual summary of the current top configs.
- Once a configuration dominates, update `PIPELINE_PARAMS` in `sem_line_enhancer/cli.py`,
  rerun the smoke test and export commands above, inspect the PNGs, and then commit.

## Streamlit app (preview)

- Run locally: `streamlit run app/streamlit_app.py` (inside the `ymno3_gpu` env).
- Choose a preset (`lines` vs `boundaries`), then upload a `.tif`/`.png`/`.jpg` **or** pick one of the bundled samples in `examples/`.
- Preview Original + Lines/Base/Fused panels, download the `.npy` arrays, and inspect JSON metadata for the selected preset.
- The app imports the same preset dictionaries used by the CLI, so changes stay in sync.
- Deployment plan: push this repo with `streamlit` dependencies to a Hugging Face Space (Streamlit runtime) to host the free demo.
