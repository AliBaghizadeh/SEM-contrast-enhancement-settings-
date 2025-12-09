import io
from pathlib import Path

import numpy as np
import streamlit as st
import tifffile
import cv2

from sem_line_enhancer.pipeline import SEMPreprocessor
from sem_line_enhancer.presets import PIPELINE_PRESETS, PREPROCESSOR_PRESETS, DEFAULT_PRESET

BASE_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"


def synthetic_grains_and_lines(shape: int = 512) -> np.ndarray:
    x = np.linspace(0, np.pi * 4, shape)
    y = np.linspace(0, np.pi * 4, shape)
    xx, yy = np.meshgrid(x, y)
    lines = 0.5 + 0.5 * np.sin(xx * 2 + yy)
    grains = np.clip(cv2.GaussianBlur(lines, (0, 0), 3) + 0.1 * np.random.randn(shape, shape), 0, 1)
    return grains.astype(np.float32)


def synthetic_grain_boundaries(shape: int = 512) -> np.ndarray:
    grid = np.indices((shape, shape)).sum(axis=0) % 40
    boundaries = np.where(grid < 2, 0.2, 0.8).astype(np.float32)
    boundaries = cv2.GaussianBlur(boundaries, (0, 0), 1)
    return boundaries


SAMPLE_FILES = {}
if EXAMPLES_DIR.exists():
    for path in sorted(EXAMPLES_DIR.glob("*.png")):
        SAMPLE_FILES[f"Sample: {path.stem}"] = path

SYNTHETIC_SAMPLES = {
    "Sample: synthetic grains + lines": synthetic_grains_and_lines,
    "Sample: synthetic grain boundaries": synthetic_grain_boundaries,
}

# Config the UI
st.set_page_config(page_title="SEM Line Enhancer", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fafdff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <h1 style="text-align:center;">SEM Image Contrast Enhancement</h1>
    <p style="text-align:center;">
        Upload a SEM micrograph or pick a sample to preview the preprocessing presets.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center;">
        <strong>Preset cheat-sheet</strong><br/>
        <em>Lines (ferroelastic / needle-like features):</em> multi-scale Frangi + DoG to emphasize elongated ridges, aggressive dirt removal, bright-line fusion. Useful when you care about domain walls or long conductive paths.<br/>
        <em>Boundaries (grain-only micrographs):</em> lighter Frangi weights, stronger CLAHE + smoothing to highlight polygonal grain edges. Ideal when the image mostly contains grain interiors and boundaries with little line texture.<br/>
        Select the preset that best matches your sample, then compare the <code>Lines</code>, <code>Base</code>, and <code>Fused</code> views to understand what will be fed downstream (e.g., SAM/MatSAM).
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_preprocessor(preset: str) -> SEMPreprocessor:
    params = PREPROCESSOR_PRESETS[preset]
    return SEMPreprocessor(**params)


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def load_sem_bytes(data: bytes, suffix: str) -> np.ndarray:
    if suffix in [".tif", ".tiff"]:
        arr = tifffile.imread(io.BytesIO(data))
    else:
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        arr = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError("Unable to decode image. Please upload TIFF/PNG/JPG.")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return _normalize_image(arr)


def load_sem_image(uploaded_file) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return load_sem_bytes(data, suffix)


def load_sem_from_path(path: Path) -> np.ndarray:
    data = path.read_bytes()
    return load_sem_bytes(data, path.suffix.lower())



col1, col2 = st.columns([2, 3])
with col1:
    st.markdown(
        "<h4 style='text-align:left;'>Select preset</h4>",
        unsafe_allow_html=True,
    )
    preset = st.selectbox(
        "",
        list(PIPELINE_PRESETS.keys()),
        index=list(PIPELINE_PRESETS.keys()).index(DEFAULT_PRESET),
    )

with col2:
    st.markdown(
        "<h4 style='text-align:left;'>Use sample or upload</h4>",
        unsafe_allow_html=True,
    )
    all_sample_choices = ["Upload your own"]
    if SAMPLE_FILES:
        all_sample_choices += list(SAMPLE_FILES.keys())
    if SYNTHETIC_SAMPLES:
        all_sample_choices += list(SYNTHETIC_SAMPLES.keys())

    sample_choice = st.selectbox("", all_sample_choices)

uploaded = None
image = None
filename = "uploaded_image"

if sample_choice != "Upload your own":
    if sample_choice in SAMPLE_FILES:
        sample_path = SAMPLE_FILES[sample_choice]
        try:
            image = load_sem_from_path(sample_path)
        except Exception as exc:
            st.error(f"Failed to load sample: {exc}")
            st.stop()
        filename = sample_path.stem
    else:
        generator = SYNTHETIC_SAMPLES[sample_choice]
        image = generator()
        filename = sample_choice.replace("Sample: ", "").replace(" ", "_")
else:
    uploaded = st.file_uploader(
        "Upload a SEM image (TIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"]
    )
    if uploaded is None:
        st.info("Upload an image or pick a sample to begin.")
        st.stop()
    filename = Path(uploaded.name).stem
    try:
        image = load_sem_image(uploaded)
    except Exception as exc:
        st.error(f"Failed to read image: {exc}")
        st.stop()

preprocessor = get_preprocessor(preset)

with st.spinner("Running preprocessing pipeline..."):
    i_lines, i_base, i_fused, _ = preprocessor.preprocess_dual(
        image, **PIPELINE_PRESETS[preset]
    )


def show_image(col, title: str, arr: np.ndarray):
    col.subheader(title)
    col.image(arr, clamp=True, use_column_width=True)


cols = st.columns(4)
show_image(cols[0], "Original", image)
show_image(cols[1], "Lines", i_lines)
show_image(cols[2], "Base", i_base)
show_image(cols[3], "Fused", i_fused)


def download_button(label: str, arr: np.ndarray, suffix: str):
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    st.download_button(
        label=label,
        data=buffer,
        file_name=f"{filename}_{suffix}.npy",
        mime="application/octet-stream",
    )


st.markdown("### Download arrays")
download_cols = st.columns(3)
download_button("Download `lines.npy`", i_lines, "lines")
download_button("Download `base.npy`", i_base, "base")
download_button("Download `fused.npy`", i_fused, "fused")

st.markdown("### Pipeline parameters")
st.json(
    {
        "preset": preset,
        "pipeline": PIPELINE_PRESETS[preset],
        "preprocessor": PREPROCESSOR_PRESETS[preset],
    }
)
