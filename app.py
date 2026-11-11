"""
Streamlit Web Application: Indian Cricket Team Member Face Identification
Interactive computer vision application with face detection, multi-model predictions,
confidence distributions, cricketer player cards, evaluation analytics, and webcam support.
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.detector import FaceDetector
from src.features import FeatureExtractor
from src.models import load_artifacts
from src.player_info import PLAYERS_INFO, get_player_info


# Streamlit Page Configuration
st.set_page_config(
    page_title="Indian Cricket Team Face ID",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian Cricket theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0b3c68 0%, #1a5b8c 50%, #f26522 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0f2fe;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 4px solid #0b3c68;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .player-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Loads model artifacts and feature extractors once."""
    try:
        model, scaler, class_mapping = load_artifacts("artifacts")
        detector = FaceDetector()
        extractor = FeatureExtractor(hog_size=(64, 64))
        return model, scaler, class_mapping, detector, extractor, None
    except Exception as e:
        return None, None, None, None, None, str(e)


def run_prediction_pipeline(image_bgr, model, scaler, class_mapping, detector, extractor):
    """Processes a BGR image and returns complete classification data."""
    crop, bbox, method = detector.crop_primary_face(
        image_bgr, target_size=(64, 64), margin=0.12, fallback_center=True
    )

    if crop is None:
        return None

    features = extractor.extract_features(crop, mode="combined")
    features_scaled = scaler.transform(features.reshape(1, -1))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features_scaled)[0]
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
    else:
        probs = np.zeros(len(class_mapping))
        p = model.predict(features_scaled)[0]
        probs[p] = 1.0

    top_idx = int(np.argmax(probs))
    best_key = class_mapping[top_idx]
    confidence = float(probs[top_idx])

    player_info = get_player_info(best_key)
    display_name = player_info["display_name"]

    annotated = detector.draw_prediction_overlay(
        image_bgr, bbox=bbox, player_name=display_name, confidence=confidence
    )

    df_probs = pd.DataFrame({
        "Player": [class_mapping[i].replace("_", " ").title() for i in range(len(class_mapping))],
        "Confidence (%)": [float(p * 100) for p in probs]
    }).sort_values(by="Confidence (%)", ascending=True)

    return {
        "display_name": display_name,
        "player_key": best_key,
        "confidence": confidence,
        "crop": crop,
        "bbox": bbox,
        "annotated": annotated,
        "df_probs": df_probs,
        "player_info": player_info,
        "method": method
    }


def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>🏏 Indian Cricket Team Member Face Identification</h1>
        <p>Computer Vision & Machine Learning Facial Biometric Recognition System</p>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, class_mapping, detector, extractor, load_err = load_classifier()

    if load_err:
        st.error(f"⚠️ Model artifacts not found: {load_err}")
        st.info("Please train the model first by running `python train.py` in your terminal.")
        return

    # Sidebar
    st.sidebar.title("🇮🇳 Team Squad")
    st.sidebar.markdown("**Selected Indian Cricket Stars:**")
    for key, info in PLAYERS_INFO.items():
        st.sidebar.markdown(f"• **{info['display_name']}** `#{info['jersey_number']}` ({info['nickname']})")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pipeline Specifications")
    st.sidebar.markdown("""
    - **Face Detector:** OpenCV Haar Cascade
    - **Feature Descriptors:**
      - HOG (1764-d)
      - 2D Haar Wavelets (1024-d)
      - LBP Texture (256-d)
      - Raw Normalized (1024-d)
    - **Total Feature Size:** 4,068 dimensions
    - **Champion Classifier:** SVM (Linear Hyperplane)
    - **Validation:** 5-Fold Stratified CV
    """)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Face Identification",
        "📸 Live Webcam Test",
        "📊 Model Evaluation & Metrics",
        "📁 Kaggle & Google Dataset Hub"
    ])

    # ---------------- TAB 1: IDENTIFICATION ----------------
    with tab1:
        st.subheader("Upload an Image or Select a Sample Cricketer")

        col_input_type, col_sample = st.columns([1, 2])
        with col_input_type:
            input_mode = st.radio("Choose Input Mode:", ["Pick from Test Gallery", "Upload Image File"], horizontal=True)

        selected_image_bgr = None
        test_samples_dir = "data/test_samples"

        if input_mode == "Pick from Test Gallery":
            if os.path.exists(test_samples_dir):
                sample_files = [f for f in os.listdir(test_samples_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                if sample_files:
                    selected_file = st.selectbox(
                        "Select sample test image:",
                        sample_files,
                        format_func=lambda f: f.replace("_", " ").replace(".jpg", "").title()
                    )
                    file_path = os.path.join(test_samples_dir, selected_file)
                    selected_image_bgr = cv2.imread(file_path)
            else:
                st.warning("Sample test directory not found. Please upload an image.")

        else:
            uploaded_file = st.file_uploader("Upload Cricketer Face Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                selected_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if selected_image_bgr is not None:
            res = run_prediction_pipeline(selected_image_bgr, model, scaler, class_mapping, detector, extractor)

            if res is None:
                st.error("Could not detect any face in the uploaded image. Please try another portrait.")
            else:
                st.markdown("---")
                col_left, col_right = st.columns([1.2, 1.8])

                with col_left:
                    st.markdown("### 🎯 Face Detection & Bounding Box")
                    annotated_rgb = cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption=f"Prediction: {res['display_name']} ({res['confidence']*100:.1f}%)", use_container_width=True)

                    # Cropped Face Inspection
                    st.markdown("#### Isolated Face Chip (Normalized 64x64)")
                    crop_rgb = cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, width=130)

                with col_right:
                    # Player Card
                    pinfo = res["player_info"]
                    st.markdown(f"""
                    <div class="player-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h2 style="margin: 0; color: #0b3c68;">{pinfo['display_name']}</h2>
                            <span style="font-size: 1.4rem; font-weight: 700; color: #f26522;">#{pinfo['jersey_number']}</span>
                        </div>
                        <p style="color: #64748b; font-style: italic; margin-top: 2px;">"{pinfo['nickname']}"</p>
                        <p><strong>🏏 Role:</strong> {pinfo['role']}</p>
                        <p><strong>🎯 Facial Signatures:</strong> {pinfo['facial_features_note']}</p>
                        <p><strong>🏆 Key Achievements:</strong></p>
                        <ul>
                            {"".join([f"<li>{a}</li>" for a in pinfo['notable_achievements']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📈 Prediction Confidence Distribution")
                    fig = px.bar(
                        res["df_probs"],
                        x="Confidence (%)",
                        y="Player",
                        orientation="h",
                        color="Confidence (%)",
                        color_continuous_scale=["#cbd5e1", "#3b82f6", "#f26522"],
                        range_x=[0, 100]
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2: LIVE WEBCAM ----------------
    with tab2:
        st.subheader("Live Camera Snapshot Recognition")
        st.write("Take a snapshot with your webcam to test face detection and recognition live.")

        cam_pic = st.camera_input("Take a photo")
        if cam_pic is not None:
            bytes_data = cam_pic.getvalue()
            img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            res = run_prediction_pipeline(img_bgr, model, scaler, class_mapping, detector, extractor)
            if res:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    annotated_rgb = cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption=f"Identified as: {res['display_name']}", use_container_width=True)
                with col_c2:
                    st.success(f"### Closest Match: {res['display_name']} ({res['confidence']*100:.1f}%)")
                    st.write(f"**Role:** {res['player_info']['role']}")
                    st.write(f"**Jersey:** #{res['player_info']['jersey_number']}")
                    st.plotly_chart(
                        px.bar(res["df_probs"], x="Confidence (%)", y="Player", orientation="h", range_x=[0, 100]),
                        use_container_width=True
                    )
            else:
                st.error("No face detected in webcam snapshot. Adjust your lighting or face position.")

    # ---------------- TAB 3: EVALUATION & METRICS ----------------
    with tab3:
        st.subheader("Model Benchmark & Evaluation Results")

        # Load metrics.json
        metrics_path = "artifacts/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

            summary_rows = []
            for m_name, vals in metrics_data.items():
                summary_rows.append({
                    "Algorithm": m_name,
                    "5-Fold CV Accuracy": f"{vals['cv_mean_acc']*100:.2f}% (±{vals['cv_std_acc']*100:.2f}%)",
                    "Test Accuracy": f"{vals['test_acc']*100:.2f}%",
                    "Precision": f"{vals['precision']*100:.2f}%",
                    "Recall": f"{vals['recall']*100:.2f}%",
                    "F1-Score": f"{vals['f1_score']*100:.2f}%"
                })

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        col_cm, col_bench = st.columns(2)
        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            cm_img_path = "artifacts/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Run `python train.py` to generate the confusion matrix plot.")

        with col_bench:
            st.markdown("#### 🏆 Benchmark Comparison Across Algorithms")
            bench_img_path = "artifacts/model_comparison.png"
            if os.path.exists(bench_img_path):
                st.image(bench_img_path, use_container_width=True)
            else:
                st.info("Run `python train.py` to generate the benchmark comparison chart.")

        st.markdown("---")
        st.markdown("### 🔬 Biometric Feature Analysis & Eigenfaces")
        col_decomp, col_eigen = st.columns(2)
        with col_decomp:
            st.markdown("#### 🧩 Multi-Modal Feature Decomposition Pipeline")
            st.caption("How the system breaks down a facial portrait: Detection -> CLAHE -> 2D Haar Wavelets -> LBP Texture -> HOG Gradients")
            decomp_path = "artifacts/feature_decomposition.png"
            if os.path.exists(decomp_path):
                st.image(decomp_path, use_container_width=True)

        with col_eigen:
            st.markdown("#### 👤 Indian Cricket Squad Eigenfaces (PCA Bases)")
            st.caption("Principal component facial variance patterns discovered across the training dataset")
            eigen_path = "artifacts/eigenfaces.png"
            if os.path.exists(eigen_path):
                st.image(eigen_path, use_container_width=True)

    # ---------------- TAB 4: KAGGLE DATASET HUB ----------------
    with tab4:
        st.subheader("📁 Kaggle & Google Dataset Search Integration")
        st.markdown("""
        You can expand the model's training dataset using images from **Kaggle** or **Google Dataset Search**.
        The pipeline is fully compatible with any image dataset structured by player name.
        """)

        st.markdown("### 🌐 Recommended Kaggle Datasets")
        st.markdown("""
        1. **[Indian Cricketers Image Dataset (Shilpee Prasad)](https://www.kaggle.com/datasets/shilpeeprasad/indian-cricketers-image-dataset)**
           - ~1,400 images of Indian cricketers partitioned into player subfolders (~50 images each).
           - Ideal for expanding Virat Kohli, Rohit Sharma, MS Dhoni, Sachin Tendulkar, etc.
        2. **[Indian Cricketers Dataset (Aminul Hossain)](https://www.kaggle.com/datasets/aminulhossain01/indian-cricketers-dataset)**
           - Photos of prominent Indian men and women cricketers.
        3. **[Cricket Players Faces (Pranav Surte)](https://www.kaggle.com/datasets/pranavsurte/cricket-players-faces)**
           - Cropped facial dataset formatted for computer vision classifiers.
        """)

        st.markdown("### 📥 How to Add Kaggle Images to This Project")
        st.code("""
# Method 1: Kaggle CLI (if kaggle.json is set up)
python download_kaggle_dataset.py --download shilpeeprasad/indian-cricketers-image-dataset

# Method 2: Unzip a manually downloaded dataset
python download_kaggle_dataset.py --extract C:/path/to/archive.zip

# Method 3: Manual Folder Drop-in
# Place raw player photos inside:
# data/raw/virat_kohli/
# data/raw/rohit_sharma/
# data/raw/ms_dhoni/
# ...

# Then retrain the entire pipeline:
python train.py
        """, language="bash")


if __name__ == "__main__":
    main()

