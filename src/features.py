"""
Feature extraction pipeline for facial representation:
- 2D Haar Wavelet Transform (pure NumPy illumination-invariant texture isolation)
- Histogram of Oriented Gradients (HOG) via OpenCV
- Local Binary Patterns (LBP) texture descriptors
- Multi-modal stacked feature representations
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def haar_wavelet_2d(img: np.ndarray, level: int = 1) -> np.ndarray:
    """
    Performs 2D discrete Haar Wavelet decomposition on a grayscale image.
    Decomposes into approximation and high-frequency (horizontal, vertical, diagonal) detail subbands.
    Pure NumPy implementation without requiring external pywt.
    """
    arr = img.astype(np.float32)
    h, w = arr.shape
    for _ in range(level):
        if h < 2 or w < 2:
            break
        # Row transform
        half_w = w // 2
        low_row = (arr[:h, 0:w:2] + arr[:h, 1:w:2]) / np.sqrt(2.0)
        high_row = (arr[:h, 0:w:2] - arr[:h, 1:w:2]) / np.sqrt(2.0)
        arr[:h, :half_w] = low_row
        arr[:h, half_w:w] = high_row

        # Column transform
        half_h = h // 2
        low_col = (arr[0:h:2, :w] + arr[1:h:2, :w]) / np.sqrt(2.0)
        high_col = (arr[0:h:2, :w] - arr[1:h:2, :w]) / np.sqrt(2.0)
        arr[:half_h, :w] = low_col
        arr[half_h:h, :w] = high_col

        h, w = half_h, half_w

    return arr


def extract_wavelet_features(face_gray: np.ndarray, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Extracts high-frequency wavelet detail features.
    Filters out global illumination variations and highlights edges, eyes, nose, and facial hair.
    """
    resized = cv2.resize(face_gray, (32, 32), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    decomp = haar_wavelet_2d(norm, level=2)

    # Zero out top-left approximation (illumination bias) to focus strictly on structural details
    h, w = decomp.shape
    decomp[: h // 4, : w // 4] = 0.0

    return decomp.flatten()


def extract_lbp_histogram(face_gray: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """
    Computes Local Binary Pattern (LBP) histogram for micro-texture representation.
    Compares 8 neighboring pixels to the center pixel.
    """
    resized = cv2.resize(face_gray, (32, 32), interpolation=cv2.INTER_AREA)
    h, w = resized.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    center = resized[1:-1, 1:-1]
    # 8 neighbors
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1)
    ]
    for bit_idx, (dy, dx) in enumerate(offsets):
        neighbor = resized[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit_idx)

    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, 256), density=True)
    return hist.astype(np.float32)


class FeatureExtractor:
    """Multi-modal feature extractor for face images."""

    def __init__(self, hog_size: Tuple[int, int] = (64, 64)):
        self.hog_size = hog_size
        # Standard HOG configuration for face features
        self.hog = cv2.HOGDescriptor(
            _winSize=hog_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

    def extract_hog(self, face_gray: np.ndarray) -> np.ndarray:
        """Computes 1764-dimensional HOG feature vector."""
        if face_gray.shape != self.hog_size:
            face_gray = cv2.resize(face_gray, self.hog_size, interpolation=cv2.INTER_AREA)
        features = self.hog.compute(face_gray)
        return features.flatten() if features is not None else np.zeros(1764, dtype=np.float32)

    def extract_features(
        self,
        face_img: np.ndarray,
        mode: str = "combined"
    ) -> np.ndarray:
        """
        Extracts facial features from a cropped face image.
        Modes:
            - 'hog': HOG gradient orientation descriptor (1764-d)
            - 'wavelet': 2D Haar Wavelet detail features (1024-d)
            - 'lbp': Local Binary Pattern micro-texture histogram (256-d)
            - 'raw': Normalized downscaled pixel values (1024-d)
            - 'combined': Stacked HOG + Wavelet + LBP + Raw features
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Input face image is empty or None")

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Normalize contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)

        if mode == "hog":
            return self.extract_hog(norm_gray)
        elif mode == "wavelet":
            return extract_wavelet_features(norm_gray)
        elif mode == "lbp":
            return extract_lbp_histogram(norm_gray)
        elif mode == "raw":
            resized = cv2.resize(norm_gray, (32, 32), interpolation=cv2.INTER_AREA)
            return (resized.astype(np.float32) / 255.0).flatten()
        elif mode == "combined":
            feat_hog = self.extract_hog(norm_gray)
            feat_wavelet = extract_wavelet_features(norm_gray)
            feat_lbp = extract_lbp_histogram(norm_gray)
            resized = cv2.resize(norm_gray, (32, 32), interpolation=cv2.INTER_AREA)
            feat_raw = (resized.astype(np.float32) / 255.0).flatten()
            return np.hstack([feat_hog, feat_wavelet, feat_lbp, feat_raw])
        else:
            raise ValueError(f"Unknown feature extraction mode: {mode}")

