"""
Inference & Prediction Engine for Indian Cricket Team Face Identification.
- Accepts an image path (or runs against test samples)
- Localizes facial region with OpenCV Haar Cascade
- Extracts multi-modal features (HOG + Wavelet + LBP + Raw)
- Computes class probabilities and top-ranked player matches
- Saves annotated images with bounding boxes and confidence banners
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.detector import FaceDetector
from src.features import FeatureExtractor
from src.models import load_artifacts
from src.player_info import get_player_info


class CricketFaceClassifier:
    """End-to-end classifier for Indian Cricket Team Member Face Identification."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model, self.scaler, self.class_mapping = load_artifacts(artifacts_dir)
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor(hog_size=(64, 64))

    def predict_image(
        self,
        image_or_path: Any,
        save_annotated_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classifies the primary face in an image.
        Returns:
            - 'predicted_name': string
            - 'player_key': raw class key
            - 'confidence': float (0.0 to 1.0)
            - 'top_predictions': list of (name, probability)
            - 'bbox': (x, y, w, h)
            - 'annotated_image': BGR numpy array
            - 'player_info': dictionary of career details
        """
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image file not found: {image_or_path}")
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_or_path}")
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path.copy()
        else:
            raise TypeError("Expected image file path or numpy.ndarray")

        # 1. Detect and crop primary face
        crop, bbox, method = self.detector.crop_primary_face(
            image,
            target_size=(64, 64),
            margin=0.12,
            fallback_center=True
        )

        if crop is None:
            raise RuntimeError("Unable to isolate a facial region from the input image.")

        # 2. Extract multi-modal features
        features = self.extractor.extract_features(crop, mode="combined")
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # 3. Model Inference & Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_scaled)[0]
        elif hasattr(self.model, "decision_function"):
            # Calibrate decision scores with softmax
            scores = self.model.decision_function(features_scaled)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
        else:
            probs = np.zeros(len(self.class_mapping))
            pred_idx = self.model.predict(features_scaled)[0]
            probs[pred_idx] = 1.0

        top_indices = np.argsort(probs)[::-1]
        best_idx = int(top_indices[0])
        best_key = self.class_mapping[best_idx]
        confidence = float(probs[best_idx])

        # Top matches list
        top_matches = [
            (self.class_mapping[idx].replace("_", " ").title(), float(probs[idx]))
            for idx in top_indices
        ]

        player_info = get_player_info(best_key)
        display_name = player_info["display_name"]

        # 4. Generate annotated visualization
        annotated_img = self.detector.draw_prediction_overlay(
            image,
            bbox=bbox,
            player_name=display_name,
            confidence=confidence
        )

        if save_annotated_path:
            os.makedirs(os.path.dirname(save_annotated_path), exist_ok=True)
            cv2.imwrite(save_annotated_path, annotated_img)
            print(f"[+] Saved annotated prediction to: {save_annotated_path}")

        return {
            "predicted_name": display_name,
            "player_key": best_key,
            "confidence": confidence,
            "top_predictions": top_matches,
            "bbox": bbox,
            "detection_method": method,
            "annotated_image": annotated_img,
            "cropped_face": crop,
            "player_info": player_info
        }


def main():
    parser = argparse.ArgumentParser(description="Indian Cricket Team Face Identification Inference CLI")
    parser.add_argument("--image", type=str, help="Path to input image file to classify")
    parser.add_argument("--test-all", action="store_true", help="Evaluate classifier on all sample test images")
    parser.add_argument("--output-dir", type=str, default="artifacts/predictions", help="Directory to save output annotated images")

    args = parser.parse_args()
    classifier = CricketFaceClassifier()

    if args.test_all or not args.image:
        test_dir = "data/test_samples"
        if not os.path.exists(test_dir):
            print(f"[!] Test samples directory '{test_dir}' not found.")
            return

        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print("=" * 70)
        print("     BENCHMARKING CLASSIFIER ON DEDICATED TEST SAMPLES")
        print("=" * 70)
        print(f"{'Image Filename':<28} | {'Predicted Player':<20} | {'Confidence':<10}")
        print("-" * 70)

        correct = 0
        for f in test_files:
            file_path = os.path.join(test_dir, f)
            out_file = os.path.join(args.output_dir, f"annotated_{f}")
            res = classifier.predict_image(file_path, save_annotated_path=out_file)

            # Ground truth check from filename
            expected_key = "_".join(f.split("_")[:2])
            is_match = expected_key.lower() == res["player_key"].lower()
            if is_match:
                correct += 1
            mark = "[OK]" if is_match else "[X]"

            print(f"{f:<28} | {res['predicted_name']:<20} | {res['confidence']*100:>6.1f}% {mark}")

        print("-" * 70)
        acc = (correct / len(test_files)) * 100 if test_files else 0
        print(f"[+] Overall Test Set Accuracy: {correct}/{len(test_files)} ({acc:.1f}%)\n")

    elif args.image:
        print(f"[*] Analyzing image: {args.image}")
        filename = os.path.basename(args.image)
        out_file = os.path.join(args.output_dir, f"annotated_{filename}")
        res = classifier.predict_image(args.image, save_annotated_path=out_file)

        print("\n" + "=" * 55)
        print(f"  IDENTIFIED PLAYER: {res['predicted_name'].upper()}")
        print(f"  Confidence:        {res['confidence'] * 100:.2f}%")
        print("=" * 55)
        info = res["player_info"]
        print(f"  Role:              {info['role']}")
        print(f"  Jersey Number:     #{info['jersey_number']}")
        print(f"  Nickname:          {info['nickname']}")
        print("\n  Top Ranked Probabilities:")
        for name, prob in res["top_predictions"]:
            bar = "#" * int(prob * 25)
            print(f"    - {name:<18} : {prob*100:>5.1f}% | {bar}")
        print("=" * 55)


if __name__ == "__main__":
    main()

