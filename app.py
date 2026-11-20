"""
Module 5: Deployment & Executive Web Dashboard
Streamlit interactive web app for real-time satellite land cover image classification.
Features: Glassmorphism UI, Live Inference, Model Performance Analytics, Architecture Explorer.
"""

import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="TerraVision AI | Satellite Imagery Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling (Glassmorphism & Gradient Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background and containers */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(243, 246, 255, 0.95) 0%, rgba(238, 242, 250, 0.95) 90%);
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 35px -10px rgba(15, 23, 42, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(147, 197, 253, 0.3);
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #93C5FD;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.6rem;
        max-width: 750px;
        line-height: 1.5;
    }

    /* Stat Cards */
    .stat-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .stat-num {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
    }

    /* Glass Panels */
    .glass-panel {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.75rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.5rem;
    }

    /* Prediction Banner */
    .prediction-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 1.75rem;
        border-radius: 18px;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.2);
        margin-bottom: 1.5rem;
    }

    .conf-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .conf-high {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .conf-med {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }

    /* Progress bar custom styling */
    .prob-bar-container {
        margin-bottom: 1rem;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.92rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.35rem;
    }
    .prob-bar-bg {
        background: #F1F5F9;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #0F172A !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Model Cache
# ---------------------------------------------------------
CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]
MODEL_PATH = "models/best_satellite_model.h5"

CLASS_INFO = {
    "cloudy": {
        "icon": "☁️",
        "title": "Cloud Cover & Atmosphere",
        "color": "#64748B",
        "bar_color": "linear-gradient(90deg, #94A3B8, #64748B)",
        "desc": "Dense cloud cover, atmospheric vapor, or high-altitude haze obscuring the earth's surface.",
        "application": "Weather forecasting, atmospheric correction, solar irradiance calculation."
    },
    "desert": {
        "icon": "🏜️",
        "title": "Arid & Desert Terrain",
        "color": "#D97706",
        "bar_color": "linear-gradient(90deg, #F59E0B, #D97706)",
        "desc": "Arid barren landscapes, sandy dunes, exposed bedrock, and xerophytic dry zones.",
        "application": "Desertification tracking, solar farm planning, geological and mineral exploration."
    },
    "green_area": {
        "icon": "🌲",
        "title": "Dense Vegetation & Forest",
        "color": "#059669",
        "bar_color": "linear-gradient(90deg, #10B981, #059669)",
        "desc": "Lush vegetation, agricultural crops, grasslands, and dense temperate/tropical canopy.",
        "application": "Deforestation monitoring, carbon credit evaluation, crop health surveillance."
    },
    "water": {
        "icon": "🌊",
        "title": "Water Bodies & Hydrology",
        "color": "#0284C7",
        "bar_color": "linear-gradient(90deg, #38BDF8, #0284C7)",
        "desc": "Open oceans, inland lakes, flowing rivers, wetlands, or coastal reservoirs.",
        "application": "Drought monitoring, flood risk assessment, aquatic ecosystem preservation."
    }
}

@st.cache_resource
def load_classification_model():
    """Loads and caches the pre-trained deep learning model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(pil_img, target_size=(128, 128)):
    """Prepares image for model inference."""
    rgb_img = pil_img.convert("RGB")
    resized_img = rgb_img.resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor, rgb_img

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🛰️</div>
        <h2 style="margin: 0; color: #0F172A; font-weight: 800;">TerraVision AI</h2>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0.2rem;">Satellite Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Model Information")
    st.markdown("""
    * **Model:** Deep Convolutional Neural Network
    * **Test Accuracy:** **90.61%**
    * **Macro F1-Score:** **90.95%**
    * **Input Dimensions:** $128 \\times 128 \\times 3$
    * **Weights Checkpoint:** `models/best_satellite_model.h5`
    """)
    
    st.markdown("---")
    st.markdown("### 🌐 Target Land Classes")
    for c_name, c_data in CLASS_INFO.items():
        st.markdown(f"**{c_data['icon']} {c_name.replace('_', ' ').title()}**  \n<span style='font-size: 0.8rem; color: #64748B;'>{c_data['desc'][:65]}...</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with TensorFlow 2.14, Keras & Streamlit.")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI Powered Earth Observation</div>
    <h1 class="hero-title">Satellite Land Cover Classification System</h1>
    <p class="hero-subtitle">
        Automated multi-spectral terrain analysis powered by Deep Convolutional Neural Networks.
        Accurately detects cloud cover, arid desert lands, dense vegetation, and open water bodies with 90.61% benchmark accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Key Performance Metric Cards
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-num" style="color: #2563EB;">90.61%</div>
        <div class="stat-label">Test Set Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-num" style="color: #059669;">90.95%</div>
        <div class="stat-label">Macro F1-Score</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-num" style="color: #7C3AED;">5,631</div>
        <div class="stat-label">Dataset Images</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-num" style="color: #D97706;">~15 ms</div>
        <div class="stat-label">Inference Latency</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_live, tab_metrics, tab_arch, tab_team = st.tabs([
    "🛰️ Live Classifier",
    "📊 Performance & Confusion Matrix",
    "🧠 CNN Architecture & Pipeline",
    "📖 Project Overview"
])

model = load_classification_model()

# =========================================================
# TAB 1: LIVE CLASSIFIER
# =========================================================
with tab_live:
    if model is None:
        st.error("⚠️ Model file `models/best_satellite_model.h5` not found. Please verify the model path.")
    else:
        col_input, col_result = st.columns([1.05, 1.25], gap="large")

        with col_input:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("### 📤 1. Select or Upload Image")
            
            input_source = st.radio(
                "Choose input source:",
                ["📁 Curated Sample Gallery", "⬆️ Upload Custom Satellite Photo"],
                horizontal=True
            )

            selected_image = None
            sample_source_tag = ""

            if input_source == "📁 Curated Sample Gallery":
                sample_col1, sample_col2 = st.columns(2)
                with sample_col1:
                    chosen_class = st.selectbox("Category:", CLASS_NAMES, format_func=lambda x: f"{CLASS_INFO[x]['icon']} {x.replace('_', ' ').title()}")
                with sample_col2:
                    sample_dir = os.path.join("data", chosen_class)
                    sample_files = []
                    if os.path.exists(sample_dir):
                        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))][:20]
                    chosen_file = st.selectbox("Select Sample:", sample_files)
                
                if sample_files and chosen_file:
                    sample_path = os.path.join(sample_dir, chosen_file)
                    selected_image = Image.open(sample_path)
                    sample_source_tag = f"Sample: {chosen_file} (Ground Truth: {chosen_class.upper()})"

            else:
                uploaded_file = st.file_uploader(
                    "Drop an aerial or satellite photo (.jpg, .jpeg, .png)",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    sample_source_tag = f"Uploaded File: {uploaded_file.name}"

            # Image Preview Card
            if selected_image is not None:
                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
                st.image(selected_image, caption=sample_source_tag, use_column_width=True)
                
                # Image metadata badges
                st.markdown(f"""
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span style="background: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Dimensions: {selected_image.size[0]}×{selected_image.size[1]} px</span>
                    <span style="background: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Channels: {len(selected_image.getbands())} ({selected_image.mode})</span>
                    <span style="background: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Status: Ready for Inference</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("### 🎯 2. Deep Learning Analysis")

            if selected_image is not None:
                with st.spinner("Processing image through Convolutional Neural Network..."):
                    input_tensor, _ = preprocess_image(selected_image)
                    raw_predictions = model.predict(input_tensor, verbose=0)[0]
                    top_idx = int(np.argmax(raw_predictions))
                    pred_class = CLASS_NAMES[top_idx]
                    confidence = float(raw_predictions[top_idx] * 100)

                target_info = CLASS_INFO[pred_class]
                conf_class = "conf-high" if confidence >= 85 else "conf-med"

                # Prediction Banner
                st.markdown(f"""
                <div class="prediction-card" style="border-left-color: {target_info['color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 700;">Predicted Land Category</span>
                        <span class="conf-badge {conf_class}">{confidence:.2f}% Confidence</span>
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
                        <span>{target_info['icon']}</span>
                        <span>{pred_class.replace('_', ' ').upper()}</span>
                    </div>
                    <p style="color: #CBD5E1; font-size: 0.95rem; margin-bottom: 0; line-height: 1.5;">
                        {target_info['desc']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Class Probabilities Breakdown (Custom Animated Progress Bars)
                st.markdown("#### 📊 Confidence Score Distribution")
                for i, c_name in enumerate(CLASS_NAMES):
                    prob = float(raw_predictions[i] * 100)
                    c_meta = CLASS_INFO[c_name]
                    is_winner = (i == top_idx)
                    weight = "800" if is_winner else "600"
                    
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div class="prob-bar-label">
                            <span style="font-weight: {weight}; display: flex; align-items: center; gap: 6px;">
                                {c_meta['icon']} {c_name.replace('_', ' ').title()}
                                {'<span style="background: #3B82F6; color: white; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem; margin-left: 4px;">TOP</span>' if is_winner else ''}
                            </span>
                            <span style="font-weight: 700; color: #0F172A;">{prob:.2f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: {prob}%; background: {c_meta['bar_color']};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Real-World Domain Application
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem 1.25rem; border-radius: 12px;">
                    <div style="font-weight: 700; color: #1E293B; font-size: 0.9rem; margin-bottom: 0.3rem;">🌍 Practical Domain Application</div>
                    <div style="color: #475569; font-size: 0.85rem; line-height: 1.4;">{target_info['application']}</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="text-align: center; padding: 4rem 2rem; color: #94A3B8;">
                    <div style="font-size: 3.5rem; margin-bottom: 1rem;">🛰️</div>
                    <h3 style="color: #475569; font-weight: 700;">No Image Selected</h3>
                    <p style="font-size: 0.95rem; max-width: 380px; margin: 0 auto;">
                        Choose a sample image from the gallery or upload an aerial photo on the left to run instant neural network inference.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2: PERFORMANCE & CONFUSION MATRIX
# =========================================================
with tab_metrics:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📈 Evaluation on Unseen Test Dataset (607 Images)")
    st.caption("All metrics were computed strictly on the holdout test set to ensure realistic, unbiased generalization.")
    
    # Metrics table
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Cloudy F1-Score", "93.48%", "Precision: 93.48%")
    with m2:
        st.metric("Desert F1-Score", "93.78%", "Precision: 93.39%")
    with m3:
        st.metric("Green Area F1-Score", "89.60%", "Precision: 84.00%")
    with m4:
        st.metric("Water F1-Score", "86.96%", "Precision: 94.59%")

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2, gap="large")

    with img_col1:
        st.markdown("#### 🎯 Confusion Matrix Heatmap")
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", use_column_width=True)
        else:
            st.info("Confusion matrix plot not found in `reports/`.")

    with img_col2:
        st.markdown("#### 📉 Training & Validation Learning Curves")
        if os.path.exists("reports/training_curves.png"):
            st.image("reports/training_curves.png", use_column_width=True)
        else:
            st.info("Training curves plot not found in `reports/`.")

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    
    st.markdown("#### 🔍 Error Analysis (Borderline Misclassified Samples)")
    st.caption("Visualizing cases where the model disagreed with the ground truth reveals that errors primarily happen on complex transitional boundaries (e.g. thin cloud veils over water or dark coastal shorelines).")
    if os.path.exists("reports/error_analysis_samples.png"):
        st.image("reports/error_analysis_samples.png", use_column_width=True)
    else:
        st.info("Error analysis plot not found in `reports/`.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3: CNN ARCHITECTURE & PIPELINE
# =========================================================
with tab_arch:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Convolutional Neural Network Architecture")
    
    st.markdown("""
    The system utilizes a specialized **3-Stage Convolutional Neural Network** designed to learn multi-scale spatial and spectral features from satellite sensors while keeping parameter counts lightweight for real-time edge execution.
    """)

    arch_c1, arch_c2 = st.columns([1.2, 1], gap="large")

    with arch_c1:
        st.markdown("#### Layer Hierarchy & Feature Extraction")
        layers_data = [
            ("Input & Rescaling", "(None, 128, 128, 3)", "Pixel normalization [0, 255] → [0.0, 1.0]"),
            ("Conv Block 1", "(None, 64, 64, 32)", "32 Filters (3×3), BatchNorm, MaxPool(2×2), Dropout(0.15)"),
            ("Conv Block 2", "(None, 32, 32, 64)", "64 Filters (3×3), BatchNorm, MaxPool(2×2), Dropout(0.30)"),
            ("Conv Block 3", "(None, 16, 16, 128)", "128 Filters (3×3), BatchNorm, MaxPool(2×2), Dropout(0.30)"),
            ("Global Avg Pool", "(None, 128)", "Spatial dimension reduction to eliminate overfitting"),
            ("Dense Classifier", "(None, 128)", "Fully connected ReLU layer with BatchNorm & Dropout(0.45)"),
            ("Softmax Output", "(None, 4)", "Probability distribution over [cloudy, desert, green_area, water]")
        ]
        
        for l_name, l_shape, l_detail in layers_data:
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; color: #1E293B; font-size: 0.9rem;">{l_name}</span>
                    <div style="color: #64748B; font-size: 0.78rem;">{l_detail}</div>
                </div>
                <span style="background: #E2E8F0; color: #334155; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-family: monospace;">{l_shape}</span>
            </div>
            """, unsafe_allow_html=True)

    with arch_c2:
        st.markdown("#### Model Specification")
        st.markdown("""
        * **Total Parameters:** **111,684** (436 KB)
        * **Trainable Parameters:** **110,980**
        * **Optimizer:** Adam (`lr = 0.001`, with `ReduceLROnPlateau` scheduler)
        * **Loss Function:** Sparse Categorical Crossentropy
        * **Data Augmentations:**
          * Random Horizontal & Vertical Flips (orientation invariance)
          * Random Rotation ($\pm 15^\circ$)
          * Random Zoom ($\pm 10\%$)
          * Random Contrast adjustment
        """)
        
        st.markdown("#### Transfer Learning Benchmark")
        st.info("MobileNetV2 feature extractor was also evaluated as an alternative backbone for comparison, validating the custom CNN's superior parameter efficiency.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 4: PROJECT OVERVIEW
# =========================================================
with tab_team:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📋 Project Overview & Specifications")
    st.markdown("""
    This project satisfies all requirements of the **Deep Learning Image Classification System** specification:
    
    1. **Module 1 (Data Acquisition & Preparation):** Cleaned 5,631 satellite images, handled CMYK conversions, augmented dataset, and created stratified 80/10/10 train-val-test splits.
    2. **Module 2 (Model Architecture & Design):** Built deep CNN with Batch Normalization, Max Pooling, and Dropout regularization.
    3. **Module 3 (Model Training & Optimization):** Configured Adam optimizer, early stopping, learning rate annealing, and weight checkpoints.
    4. **Module 4 (Evaluation & Validation):** Computed accuracy, precision, recall, F1-scores, confusion matrix, and error analysis.
    5. **Module 5 (Deployment & UI):** Deployed full executive Streamlit dashboard for real-time inference and reporting.
    """)
    
    st.markdown("---")
    st.markdown("#### 🔗 Repository & Team Collaboration")
    st.markdown("""
    * **GitHub Repository:** [https://github.com/poojagotmare/Satellite-Image-Classification](https://github.com/poojagotmare/Satellite-Image-Classification)
    * **Branch:** `main`
    * **How to Run Web App:** `streamlit run app.py`
    * **How to Run Pipeline:** `python run_pipeline.py`
    """)
    st.markdown('</div>', unsafe_allow_html=True)
