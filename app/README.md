# Streamlit SEM Line Enhancer

This app wraps the `sem_line_enhancer` package so users can:

1. Upload a SEM image (`.tif`, `.png`, `.jpg`).
2. Run the canonical preprocessing pipeline (Frangi + dirt removal + base/fused blend) on CPU.
3. Visualize original vs. `lines`, `base`, and `fused` outputs, plus download the resulting `.npy` arrays.
4. Inspect dirt-removal metrics (threshold, max-size, etc.) that came from the latest grid-search experiments.

## Architecture

- `sem_line_enhancer.loader.SEMImageLoader` standardizes inputs (grayscale float32 `[0, 1]`).
- `sem_line_enhancer.pipeline.SEMPreprocessor.preprocess_dual` generates `(lines, base, fused)`.
- Streamlit UI orchestrates the flow: upload → preprocess → display → optional download.
- The CLI/MLflow workflow remains unchanged; the app simply imports the same package so behavior stays consistent.

## Running locally

```
streamlit run app/streamlit_app.py
```

Make sure your Conda env (`ymno3_gpu`) is active and contains Streamlit (`pip install streamlit` if needed).

## Deployment plan (HF Spaces)

- Commit `app/streamlit_app.py`, `sem_line_enhancer/`, and `requirements.txt` (with `streamlit`, `opencv-python-headless`, etc.).
- Create a new Hugging Face Space (Streamlit runtime) and point it at this repo.
- Set environment variable `PIPELINE_PRESET` only if you need to toggle experimental configs; otherwise defaults apply.

See `docs/LOCAL_TESTING.md` for preprocessing/grid-search details that inform the app.
