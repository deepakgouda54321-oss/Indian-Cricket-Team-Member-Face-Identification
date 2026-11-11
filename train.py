"""
End-to-End Training & Evaluation Pipeline for Indian Cricket Team Face Identification.
- Loads cropped face images from data/processed/
- Extracts multi-modal features (HOG + 2D Haar Wavelet + LBP + Raw pixels)
- Standardizes feature distributions
- Trains and benchmarks SVM (RBF & Linear), Random Forest, Logistic Regression, KNN
- Selects the champion model and exports all deployment artifacts
"""

import os
import sys
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.features import FeatureExtractor
from src.models import evaluate_models, save_artifacts
from src.evaluate import plot_confusion_matrix, plot_model_comparison, print_detailed_report
from src.preprocess import DatasetPreprocessor
from build_starter_dataset import generate_starter_dataset


def load_dataset(processed_dir: str = "data/processed", feature_mode: str = "combined"):
    """
    Loads all cropped faces from processed_dir, extracts features, and builds X, y matrices.
    """
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print("[!] Processed data missing. Running preprocessing...")
        if not os.path.exists("data/raw") or not os.listdir("data/raw"):
            print("[!] Raw data missing. Generating starter dataset...")
            generate_starter_dataset()
        preprocessor = DatasetPreprocessor()
        preprocessor.process_all(augment=True)

    extractor = FeatureExtractor(hog_size=(64, 64))
    classes = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    class_mapping = {idx: name for idx, name in enumerate(classes)}

    print(f"[*] Loading face images for classes: {classes}")

    X_list = []
    y_list = []
    valid_exts = {".png", ".jpg", ".jpeg"}

    for class_id, class_name in class_mapping.items():
        class_folder = os.path.join(processed_dir, class_name)
        img_files = [f for f in os.listdir(class_folder) if os.path.splitext(f.lower())[1] in valid_exts]

        for img_file in img_files:
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extractor.extract_features(img, mode=feature_mode)
            X_list.append(features)
            y_list.append(class_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"[+] Total samples loaded: {len(X)} with {X.shape[1]} features each across {len(classes)} classes.")
    return X, y, class_mapping


def run_training_pipeline(feature_mode: str = "combined"):
    """Executes the full training, hyperparameter evaluation, and reporting process."""
    print("=" * 70)
    print("  INDIAN CRICKET TEAM MEMBER FACE IDENTIFICATION - MODEL TRAINING")
    print("=" * 70)

    # 1. Load Data & Features
    X, y, class_mapping = load_dataset(feature_mode=feature_mode)

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[*] Dataset Split: {len(X_train)} Training samples | {len(X_test)} Test samples")

    # 3. Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train & Benchmark Candidate Models
    results, best_name, champion_model = evaluate_models(
        X_train_scaled, y_train, X_test_scaled, y_test, cv_folds=5
    )

    # 5. Predictions on Test Set with Champion Model
    y_pred_champion = champion_model.predict(X_test_scaled)
    class_names = [class_mapping[i].replace("_", " ").title() for i in range(len(class_mapping))]

    # 6. Generate Metrics & Confusion Matrix Visualizations
    plot_confusion_matrix(
        y_test,
        y_pred_champion,
        class_names=class_names,
        output_path="artifacts/confusion_matrix.png",
        model_name=best_name
    )

    plot_model_comparison(results, output_path="artifacts/model_comparison.png")

    print_detailed_report(y_test, y_pred_champion, class_names=class_names)

    # 7. Persist Artifacts
    save_artifacts(
        model=champion_model,
        scaler=scaler,
        class_mapping=class_mapping,
        metrics=results,
        artifacts_dir="artifacts"
    )

    print("=" * 70)
    print(f"[+] TRAINING PIPELINE COMPLETE! Champion Model: {best_name}")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()

