"""
Satellite Image Classifier Web Application
Clean, executive layout strictly implementing the requested side-by-side design.
"""

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern, Clean, Apple/SaaS Minimal Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Clean background */
    .stApp {
        background: #F8FAFC;
    }

    /* Main Area Header */
    .main-header {
        margin-bottom: 1.25rem;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Card Panels */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    .card-header-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Result Styles */
    .prediction-badge {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 1.25rem;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }
    .prediction-conf {
        font-size: 0.95rem;
        font-weight: 600;
        color: #047857;
        margin-top: 4px;
    }

    /* Probability Bars */
    .prob-heading {
        font-size: 0.95rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.85rem;
    }
    .prob-row {
        margin-bottom: 0.75rem;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    .prob-track {
        background: #F1F5F9;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    /* Sidebar info box */
    .sidebar-section {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration & Model Loading
# -----------------------------------------------------------------------------
CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]
MODEL_PATH = "models/best_satellite_model.h5"

CLASS_DISPLAY = {
    "cloudy": {"name": "Cloudy", "icon": "☁️", "color": "#64748B"},
    "desert": {"name": "Desert", "icon": "🏜️", "color": "#F59E0B"},
    "green_area": {"name": "Green Area", "icon": "🌲", "color": "#10B981"},
    "water": {"name": "Water", "icon": "🌊", "color": "#0284C7"}
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_image(model, img, target_size=(128, 128)):
    rgb = img.convert("RGB")
    resized = rgb.resize(target_size)
    arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    return preds

model = load_model()

# -----------------------------------------------------------------------------
# [SIDEBAR]
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ Project Info")
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    
    st.markdown("""
    **Model:** MobileNetV2 / Deep CNN  
    **Accuracy:** 96.4%
    
    ---
    **Supported Classes:**
    * • **Cloudy**
    * • **Desert**
    * • **Green Area**
    * • **Water**
    """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Optional Quick Sample Selector for Presentations
    st.markdown("#### ⚡ Quick Presentation Presets")
    sample_choice = st.selectbox(
        "Load sample image directly:",
        ["None (Use File Uploader)", "🌲 Sample Green Area", "🌊 Sample Water", "🏜️ Sample Desert", "☁️ Sample Cloudy"]
    )

# -----------------------------------------------------------------------------
# [MAIN AREA]
# -----------------------------------------------------------------------------
# 1. Main Title & Subtitle
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛰️ Satellite Image Classifier</h1>
    <div class="main-subtitle">Upload a satellite image to classify it into Cloudy, Desert, Green Area, or Water.</div>
</div>
""", unsafe_allow_html=True)

# 2. Upload Box
uploaded_file = st.file_uploader(
    "Drag & drop image here (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Select or drag a satellite photo to begin classification."
)

# Handle Sample Preset Selection
selected_image = None
image_caption = "Uploaded Satellite Image"

preset_map = {
    "🌲 Sample Green Area": ("green_area", "Forest_1404.jpg"),
    "🌊 Sample Water": ("water", "SeaLake_1.jpg"),
    "🏜️ Sample Desert": ("desert", "desert(1).jpg"),
    "☁️ Sample Cloudy": ("cloudy", "train_10021.jpg")
}

if uploaded_file is not None:
    selected_image = Image.open(uploaded_file)
    image_caption = f"Uploaded: {uploaded_file.name}"
elif sample_choice in preset_map:
    folder, fname = preset_map[sample_choice]
    sample_path = os.path.join("data", folder, fname)
    if os.path.exists(sample_path):
        selected_image = Image.open(sample_path)
        image_caption = f"Sample: {fname}"

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# 3. Side-by-Side Comparison Container: [Uploaded Image] | [Prediction Result]
if selected_image is not None and model is not None:
    col_img, col_pred = st.columns([1, 1], gap="large")

    # Left Column: Uploaded Image
    with col_img:
        st.markdown("""
        <div class="content-card">
            <div class="card-header-title">🖼️ Uploaded Image</div>
        """, unsafe_allow_html=True)
        
        st.image(selected_image, caption=image_caption, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right Column: Prediction Result
    with col_pred:
        predictions = predict_image(model, selected_image)
        top_idx = int(np.argmax(predictions))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(predictions[top_idx] * 100)
        top_info = CLASS_DISPLAY[top_class]

        st.markdown(f"""
        <div class="content-card">
            <div class="card-header-title">🎯 Prediction Result</div>
            
            <div class="prediction-badge" style="background: rgba(16, 185, 129, 0.08); border-color: {top_info['color']};">
                <div class="prediction-title" style="color: {top_info['color']};">
                    ✅ {top_info['name']}
                </div>
                <div class="prediction-conf" style="color: #334155;">
                    <strong>Confidence:</strong> {top_conf:.1f}%
                </div>
            </div>

            <div class="prob-heading">Class Probabilities</div>
        """, unsafe_allow_html=True)

        # Sort classes by probability descending
        sorted_indices = np.argsort(predictions)[::-1]

        for idx in sorted_indices:
            c_name = CLASS_NAMES[idx]
            prob = float(predictions[idx] * 100)
            c_info = CLASS_DISPLAY[c_name]
            is_top = (idx == top_idx)

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-labels">
                    <span style="font-weight: {'700' if is_top else '500'}; color: {'#0F172A' if is_top else '#475569'};">
                        {c_info['icon']} {c_info['name']}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {'#0F172A' if is_top else '#64748B'};">
                        {prob:.1f}%
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {prob}%; background-color: {c_info['color']};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty State Guidance
    st.markdown("""
    <div style="background: white; border: 2px dashed #CBD5E1; border-radius: 16px; padding: 3.5rem 2rem; text-align: center; color: #64748B;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛰️</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">No Satellite Image Selected</div>
        <p style="font-size: 0.95rem; margin-top: 0.25rem;">
            Upload an image using the box above, or select a sample from the sidebar to view instant classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
