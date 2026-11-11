"""
Preprocessing and Dataset Preparation Pipeline:
- Traverses raw player image directories
- Performs Haar-cascade face detection & cropping with boundary padding
- Applies data augmentation (rotations, flips, illumination shifts, noise)
- Saves standardized face chips to data/processed/<player_name>/
"""

import os
import sys
import shutil
from typing import Dict, List, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.detector import FaceDetector
except ImportError:
    from detector import FaceDetector


def augment_face_image(face_img: np.ndarray) -> List[np.ndarray]:
    """Generates synthetic variations (flips, rotations, illumination) from a single face crop."""
    augmented = []
    h, w = face_img.shape[:2]

    # 1. Original
    augmented.append(face_img)

    # 2. Horizontal Flip (Facial symmetry)
    flipped = cv2.flip(face_img, 1)
    augmented.append(flipped)

    # 3. Slight Rotations (-8, +8 degrees)
    center = (w // 2, h // 2)
    for angle in [-8.0, 8.0]:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # 4. Illumination variations (darker and brighter)
    for factor in [0.85, 1.15]:
        val = np.clip(face_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented.append(val)

    # 5. Subtle blur (simulating stadium camera motion)
    blurred = cv2.GaussianBlur(face_img, (3, 3), 0.5)
    augmented.append(blurred)

    return augmented


class DatasetPreprocessor:
    """Preprocesses raw cricket team member image datasets into standardized face crops."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", target_size=(64, 64)):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.target_size = target_size
        self.detector = FaceDetector()

    def process_all(self, augment: bool = True, clear_existing: bool = True) -> Dict[str, int]:
        """
        Processes all cricketer folders in raw_dir and outputs face crops to processed_dir.
        Returns a dictionary mapping player name to number of processed face images.
        """
        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f"Raw data directory '{self.raw_dir}' does not exist.")

        if clear_existing and os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)

        summary = {}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        player_dirs = [d for d in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, d))]
        if not player_dirs:
            print(f"[!] No player folders found inside '{self.raw_dir}'.")
            return {}

        print(f"[*] Found {len(player_dirs)} player classes: {player_dirs}")

        for player_name in player_dirs:
            player_raw_path = os.path.join(self.raw_dir, player_name)
            player_proc_path = os.path.join(self.processed_dir, player_name)
            os.makedirs(player_proc_path, exist_ok=True)

            img_files = [
                f for f in os.listdir(player_raw_path)
                if os.path.splitext(f.lower())[1] in valid_extensions
            ]

            saved_count = 0
            for idx, img_name in enumerate(img_files):
                img_path = os.path.join(player_raw_path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                crop, bbox, method = self.detector.crop_primary_face(
                    image,
                    target_size=self.target_size,
                    margin=0.12,
                    fallback_center=True
                )

                if crop is not None:
                    crops_to_save = augment_face_image(crop) if augment else [crop]
                    for aug_idx, c in enumerate(crops_to_save):
                        out_name = f"{player_name}_{idx:04d}_aug{aug_idx}.png"
                        out_path = os.path.join(player_proc_path, out_name)
                        cv2.imwrite(out_path, c)
                        saved_count += 1

            summary[player_name] = saved_count
            print(f"  [+] {player_name}: Processed {len(img_files)} raw images -> {saved_count} face samples.")

        return summary


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.process_all(augment=True)
