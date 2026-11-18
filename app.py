import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import glob
import datetime
import pandas as pd

st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
.metric-card {
    background: #1c1f26;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #4f8ef7;
}
.result-normal  { border-left-color: #2ecc71; }
.result-pneumo  { border-left-color: #e74c3c; }
.section-label  { font-size: 0.75rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; }
.big-label      { font-size: 1.8rem; font-weight: 700; margin: 0; }
h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

BASE = "C:/Users/sahim/OneDrive/Documents/NNL/chest_xray"
NORMAL_DIR   = f"{BASE}/chest_xray/test/NORMAL"
PNEUMO_DIR   = f"{BASE}/chest_xray/test/PNEUMONIA"
MODEL_PATH   = f"{BASE}/pneumonia_model.h5"

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_data
def dataset_counts():
    train_n = len(glob.glob(f"{BASE}/chest_xray/train/NORMAL/*.jpeg"))
    train_p = len(glob.glob(f"{BASE}/chest_xray/train/PNEUMONIA/*.jpeg"))
    test_n  = len(glob.glob(f"{NORMAL_DIR}/*.jpeg"))
    test_p  = len(glob.glob(f"{PNEUMO_DIR}/*.jpeg"))
    return train_n, train_p, test_n, test_p

with st.spinner("Loading model…"):
    model = get_model()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image):
    img_r = img.resize((64, 64))
    arr   = np.array(img_r) / 255.0
    return np.expand_dims(arr, axis=0)

def gradcam(model, img_array, layer_name=None):
    """Return a Grad-CAM overlay. Builds a fresh functional model from the Sequential layers
    so we never touch model.output / model.inputs (broken on Keras 3 Sequential)."""
    # Find last Conv2D layer
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
    if target_layer is None:
        return None

    # Build a new functional model by threading a fresh Input through the same layers
    inp = tf.keras.Input(shape=(64, 64, 3))
    x = inp
    conv_out_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer is target_layer:
            conv_out_tensor = x

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out_tensor, x])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads    = tape.gradient(loss, conv_out)[0]
    weights  = tf.reduce_mean(grads, axis=(0, 1))
    cam      = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy()
    cam      = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    # resize to 224 for display
    cam_uint8 = np.uint8(255 * cam)
    heatmap   = cv2.applyColorMap(cv2.resize(cam_uint8, (224, 224)), cv2.COLORMAP_JET)
    orig      = np.array(img_array[0] * 255, dtype=np.uint8)
    orig_224  = cv2.resize(orig, (224, 224))
    overlay   = cv2.addWeighted(orig_224, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def run_prediction(image: Image.Image, source_name: str):
    arr   = preprocess(image)
    score = float(model.predict(arr, verbose=0)[0][0])
    label = "Pneumonia" if score > 0.5 else "Normal"
    conf  = score if score > 0.5 else 1 - score
    hmap  = gradcam(model, arr)
    st.session_state.history.append({
        "Time":       datetime.datetime.now().strftime("%H:%M:%S"),
        "Source":     source_name,
        "Result":     label,
        "Confidence": f"{conf*100:.1f}%",
        "Raw Score":  f"{score:.4f}",
    })
    return score, label, conf, hmap

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🫁 Pneumonia Detector")
st.sidebar.divider()
st.sidebar.markdown("### Model")
st.sidebar.markdown("""
| Property | Value |
|---|---|
| Architecture | CNN |
| Layers | 2× Conv + MaxPool |
| Input | 64 × 64 RGB |
| Output | Binary sigmoid |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Epochs | 10 |
""")
st.sidebar.divider()
st.sidebar.markdown("### Dataset")
tn, tp, en, ep = dataset_counts()
st.sidebar.markdown(f"""
| Split | Normal | Pneumonia |
|---|---|---|
| Train | {tn} | {tp} |
| Test  | {en} | {ep} |
""")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_diag, tab_hist, tab_about = st.tabs(["🔬 Diagnose", "📋 History", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("### Chest X-Ray Analysis")

    input_mode = st.radio(
        "Image source",
        ["Upload your own", "Pick a sample (Normal)", "Pick a sample (Pneumonia)"],
        horizontal=True,
    )

    image = None
    source_name = ""

    if input_mode == "Upload your own":
        uploaded = st.file_uploader(
            "Drop a JPEG / PNG X-ray here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            source_name = uploaded.name

    elif input_mode == "Pick a sample (Normal)":
        samples = sorted(glob.glob(f"{NORMAL_DIR}/*.jpeg"))[:10]
        names   = [os.path.basename(p) for p in samples]
        choice  = st.selectbox("Select a Normal X-ray", names)
        image   = Image.open(samples[names.index(choice)]).convert("RGB")
        source_name = choice

    else:
        samples = sorted(glob.glob(f"{PNEUMO_DIR}/*.jpeg"))[:10]
        names   = [os.path.basename(p) for p in samples]
        choice  = st.selectbox("Select a Pneumonia X-ray", names)
        image   = Image.open(samples[names.index(choice)]).convert("RGB")
        source_name = choice

    if image:
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Input image**")
            st.image(image, use_container_width=True)

        with col_res:
            st.markdown("**Result**")
            run = st.button("▶ Run Detection", use_container_width=True, type="primary")

            if run:
                with st.spinner("Running model…"):
                    score, label, conf, hmap = run_prediction(image, source_name)

                pneumo_conf = score
                normal_conf = 1 - score
                is_pneumo   = label == "Pneumonia"

                card_cls = "result-pneumo" if is_pneumo else "result-normal"
                icon     = "⚠️" if is_pneumo else "✅"
                color    = "#e74c3c" if is_pneumo else "#2ecc71"

                st.markdown(f"""
                <div class="metric-card {card_cls}">
                    <div class="section-label">Diagnosis</div>
                    <div class="big-label" style="color:{color}">{icon} {label.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{conf*100:.1f}%")
                c2.metric("Raw Score", f"{score:.4f}")

                st.markdown("**Class probabilities**")
                prob_df = pd.DataFrame({
                    "Class": ["Normal", "Pneumonia"],
                    "Probability": [normal_conf, pneumo_conf],
                })
                st.bar_chart(prob_df.set_index("Class"))

                if hmap:
                    st.markdown("**Grad-CAM — model attention**")
                    st.image(hmap, caption="Regions the model focused on", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### Session Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet. Run a detection in the Diagnose tab.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        counts = df["Result"].value_counts()
        st.markdown("**Summary**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total scans", len(df))
        sc2.metric("Normal",    counts.get("Normal",    0))
        sc3.metric("Pneumonia", counts.get("Pneumonia", 0))

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About this project")
    st.markdown("""
This tool is a **binary image classifier** trained on the
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset from Kaggle.

#### How it works
1. A chest X-ray image is resized to **64 × 64 px** and normalized to [0, 1].
2. It passes through two **Conv2D + MaxPooling** blocks that extract spatial features.
3. The flattened features feed into a **Dense(128) → Dense(1, sigmoid)** head.
4. A score **> 0.5** is classified as *Pneumonia*; ≤ 0.5 as *Normal*.

#### Grad-CAM
The heatmap uses **Gradient-weighted Class Activation Mapping** to highlight
image regions that most influenced the model's decision — useful for
understanding what the network "looks at".

#### Limitations
- Input resolution is low (64 × 64), which limits spatial detail.
- The model was trained for 10 epochs — further fine-tuning could improve accuracy.
- This is a **research/learning project**, not a clinical diagnostic tool.

#### Stack
`TensorFlow / Keras` · `Streamlit` · `OpenCV` · `Pillow` · `pandas`
""")
