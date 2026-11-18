"""
Module 5: Deployment & Web Application
Streamlit interactive web app for real-time satellite land cover image classification.
"""

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Streamlit Page Setup
st.set_page_config(
    page_title="Satellite Image Classifier | Deep Learning",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]
MODEL_PATH = "models/best_satellite_model.h5"

CLASS_DESCRIPTIONS = {
    "cloudy": "High cloud cover, atmospheric haze, or dense cloud formations obscuring the terrain.",
    "desert": "Arid terrain, sand dunes, barren rocky expanses, or dry land with minimal vegetation.",
    "green_area": "Dense forests, agricultural fields, grasslands, or lush vegetative land cover.",
    "water": "Oceans, seas, lakes, rivers, or open coastal water bodies."
}

CLASS_ICONS = {
    "cloudy": "☁️",
    "desert": "🏜️",
    "green_area": "🌲",
    "water": "🌊"
}

@st.cache_resource
def load_classification_model():
    """Loads and caches the trained deep learning model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(pil_img, target_size=(128, 128)):
    """Prepares an uploaded PIL image for model inference."""
    rgb_img = pil_img.convert("RGB")
    resized_img = rgb_img.resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32)
    # Model includes Rescaling layer, so raw [0, 255] float array is expected
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor, rgb_img

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/satellite.png", width=120)
    st.title("Project Info")
    st.markdown("""
    **Deep Learning Image Classification**  
    *Land Cover & Remote Sensing Analysis*
    
    ---
    **Supported Classes:**
    - ☁️ `cloudy`
    - 🏜️ `desert`
    - 🌲 `green_area`
    - 🌊 `water`
    
    ---
    **Architecture**: Convolutional Neural Network (CNN)
    - Input Resolution: $128 \\times 128 \\times 3$
    - Normalization: Batch Normalization
    - Regularization: Dropout (0.15 - 0.45)
    """)

# Main Header
st.markdown('<div class="main-title">🛰️ Satellite Land Cover Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated satellite image analysis powered by Deep Convolutional Neural Networks.</div>', unsafe_allow_html=True)

model = load_classification_model()

if model is None:
    st.warning("⚠️ Trained model file `models/best_satellite_model.keras` was not found. Please train the model first by running `python run_pipeline.py`.")
else:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Satellite Image")
        upload_mode = st.radio("Choose Input Method:", ["Upload File", "Select Sample from Dataset"], horizontal=True)

        selected_image = None

        if upload_mode == "Upload File":
            uploaded_file = st.file_uploader("Upload an aerial or satellite photo (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
        else:
            sample_class = st.selectbox("Select Class:", CLASS_NAMES)
            sample_folder = os.path.join("data", sample_class)
            if os.path.exists(sample_folder):
                sample_files = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.png'))][:15]
                selected_file_name = st.selectbox("Pick an image:", sample_files)
                if selected_file_name:
                    selected_image = Image.open(os.path.join(sample_folder, selected_file_name))
            else:
                st.info("Sample directory not found.")

        if selected_image is not None:
            st.image(selected_image, caption="Input Satellite Imagery", use_column_width=True)

    with col2:
        st.subheader("2. Real-Time Classification Result")
        if selected_image is not None:
            with st.spinner("Analyzing spectral patterns & spatial features..."):
                input_tensor, display_img = preprocess_image(selected_image)
                predictions = model.predict(input_tensor, verbose=0)[0]
                
                top_idx = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[top_idx]
                confidence_pct = float(predictions[top_idx] * 100)

            # Display Top Result Card
            icon = CLASS_ICONS.get(predicted_class, "🎯")
            st.success(f"### {icon} Predicted: **{predicted_class.upper()}**")
            st.metric(label="Model Confidence Score", value=f"{confidence_pct:.2f}%")
            
            st.info(f"**Interpretation:** {CLASS_DESCRIPTIONS[predicted_class]}")

            # Confidence Breakdown Bar Chart
            st.markdown("#### Probability Distribution Across Classes")
            prob_dict = {
                f"{CLASS_ICONS[CLASS_NAMES[i]]} {CLASS_NAMES[i]}": float(predictions[i])
                for i in range(len(CLASS_NAMES))
            }
            st.bar_chart(prob_dict)
        else:
            st.info("👈 Upload an image or select a sample on the left to view the classification inference.")
