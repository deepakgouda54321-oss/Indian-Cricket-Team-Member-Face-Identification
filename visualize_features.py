"""
Feature & Biometric Visualization Module:
- Decomposes a face through all stages of the Computer Vision pipeline:
  (Original -> Detected Face -> CLAHE -> 2D Haar Wavelet -> LBP Texture -> HOG Gradients)
- Computes and visualizes the Eigenfaces (PCA Principal Components) across the squad
- Generates a squad gallery poster with jersey numbers and roles
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.detector import FaceDetector
from src.features import haar_wavelet_2d
from src.player_info import PLAYERS_INFO


def visualize_pipeline_stages(
    sample_image_path: str = "data/test_samples/virat_kohli_test1.jpg",
    output_path: str = "artifacts/feature_decomposition.png"
):
    """Visualizes step-by-step transformation of a player's face across the CV pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    detector = FaceDetector()

    img = cv2.imread(sample_image_path)
    if img is None:
        print(f"[!] Could not load {sample_image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop, bbox, _ = detector.crop_primary_face(img, target_size=(64, 64), margin=0.12)

    # 1. Bounding box overlay
    annotated = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 2. Cropped Face
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 3. Grayscale CLAHE
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 4. 2D Haar Wavelet Detail Decomposition
    norm = gray_clahe.astype(np.float32) / 255.0
    decomp = haar_wavelet_2d(norm, level=2)
    # Zero out low-frequency illumination to show high-frequency texture map
    h_d, w_d = decomp.shape
    decomp[: h_d // 4, : w_d // 4] = 0.0
    wavelet_vis = np.abs(decomp)
    wavelet_vis = ((wavelet_vis - wavelet_vis.min()) / (wavelet_vis.max() - wavelet_vis.min() + 1e-8) * 255).astype(np.uint8)

    # 5. Local Binary Patterns (LBP)
    h_c, w_c = gray_clahe.shape
    lbp = np.zeros((h_c - 2, w_c - 2), dtype=np.uint8)
    center = gray_clahe[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit_idx, (dy, dx) in enumerate(offsets):
        neighbor = gray_clahe[1 + dy : h_c - 1 + dy, 1 + dx : w_c - 1 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit_idx)

    # 6. Sobel Gradient Magnitude (HOG proxy visualization)
    grad_x = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5), dpi=200)

    axes[0].imshow(annotated_rgb)
    axes[0].set_title("1. Haar Face Detection", fontsize=10, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(crop_rgb)
    axes[1].set_title("2. Cropped Face (64x64)", fontsize=10, weight="bold")
    axes[1].axis("off")

    axes[2].imshow(gray_clahe, cmap="gray")
    axes[2].set_title("3. CLAHE Normalization", fontsize=10, weight="bold")
    axes[2].axis("off")

    axes[3].imshow(wavelet_vis, cmap="inferno")
    axes[3].set_title("4. 2D Haar Wavelet Details", fontsize=10, weight="bold")
    axes[3].axis("off")

    axes[4].imshow(lbp, cmap="cividis")
    axes[4].set_title("5. LBP Texture Micro-map", fontsize=10, weight="bold")
    axes[4].axis("off")

    axes[5].imshow(mag_vis, cmap="magma")
    axes[5].set_title("6. HOG Edge Gradients", fontsize=10, weight="bold")
    axes[5].axis("off")

    plt.suptitle("Computer Vision Feature Decomposition Pipeline — Indian Cricket Star", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Feature decomposition saved to: {output_path}")


def compute_eigenfaces(
    processed_dir: str = "data/processed",
    output_path: str = "artifacts/eigenfaces.png",
    n_components: int = 12
):
    """Computes and plots the top Eigenfaces (Principal Components) of the cricket squad."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(processed_dir):
        print(f"[!] {processed_dir} does not exist.")
        return

    images = []
    valid_exts = {".png", ".jpg", ".jpeg"}
    for player in os.listdir(processed_dir):
        p_dir = os.path.join(processed_dir, player)
        if not os.path.isdir(p_dir):
            continue
        for f in os.listdir(p_dir):
            if os.path.splitext(f.lower())[1] in valid_exts:
                img = cv2.imread(os.path.join(p_dir, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_res = cv2.resize(img, (64, 64))
                    images.append(img_res.flatten())

    if len(images) < n_components:
        print("[!] Not enough images for PCA.")
        return

    X = np.array(images, dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    # Mean face + Eigenfaces
    mean_face = pca.mean_.reshape(64, 64)
    eigenfaces = pca.components_.reshape((n_components, 64, 64))

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), dpi=200)
    for i, ax in enumerate(axes.flat):
        if i == 0:
            ax.imshow(mean_face, cmap="bone")
            ax.set_title("Squad Mean Face", fontsize=9, weight="bold")
        else:
            comp_idx = i - 1
            ax.imshow(eigenfaces[comp_idx], cmap="coolwarm")
            var = pca.explained_variance_ratio_[comp_idx] * 100
            ax.set_title(f"Eigenface #{i}\n({var:.1f}% var)", fontsize=8, weight="bold")
        ax.axis("off")

    plt.suptitle("Indian Cricket Team Eigenfaces — Principal Component Face Bases", fontsize=12, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Eigenfaces visualization saved to: {output_path}")


def generate_squad_poster(
    test_dir: str = "data/test_samples",
    output_path: str = "artifacts/squad_gallery.png"
):
    """Renders a 6-player squad gallery poster with roles and jersey numbers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    players = ["virat_kohli", "rohit_sharma", "ms_dhoni", "sachin_tendulkar", "hardik_pandya", "jasprit_bumrah"]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4), dpi=200)

    for i, p_key in enumerate(players):
        img_file = os.path.join(test_dir, f"{p_key}_test1.jpg")
        ax = axes[i]
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        info = PLAYERS_INFO.get(p_key, {})
        ax.set_title(f"{info.get('display_name', p_key)}\n#{info.get('jersey_number', '')} • {info.get('nickname', '')}", fontsize=9, weight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("🇮🇳 Indian Cricket Team — Selected Face Identification Squad", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Squad gallery poster saved to: {output_path}")


if __name__ == "__main__":
    visualize_pipeline_stages()
    compute_eigenfaces()
    generate_squad_poster()

