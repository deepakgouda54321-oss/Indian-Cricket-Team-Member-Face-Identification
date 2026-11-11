"""
Unit and Integration Test Suite for Indian Cricket Team Face Identification.
Run with: pytest tests/test_pipeline.py
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import FaceDetector
from src.features import FeatureExtractor, haar_wavelet_2d, extract_lbp_histogram
from src.player_info import get_player_info, PLAYERS_INFO
from predict import CricketFaceClassifier


@pytest.fixture
def sample_face_image():
    """Generates a synthetic portrait array for testing."""
    img = np.ones((120, 100, 3), dtype=np.uint8) * 180
    # Draw oval face
    cv2.ellipse(img, (50, 60), (30, 40), 0, 0, 360, (140, 170, 210), -1)
    # Eyes
    cv2.circle(img, (38, 52), 4, (20, 20, 20), -1)
    cv2.circle(img, (62, 52), 4, (20, 20, 20), -1)
    # Mouth
    cv2.ellipse(img, (50, 78), (10, 4), 0, 0, 360, (80, 70, 140), -1)
    return img


def test_face_detector_crop(sample_face_image):
    detector = FaceDetector()
    crop, bbox, method = detector.crop_primary_face(
        sample_face_image,
        target_size=(64, 64),
        margin=0.12,
        fallback_center=True
    )
    assert crop is not None
    assert crop.shape == (64, 64, 3)
    assert len(bbox) == 4
    assert bbox[2] > 0 and bbox[3] > 0
    assert method in ["haar_cascade", "center_crop"]


def test_face_detector_empty_input():
    detector = FaceDetector()
    crop, bbox, method = detector.crop_primary_face(None)
    assert crop is None
    assert bbox is None
    assert method == "none"

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    crop, bbox, method = detector.crop_primary_face(empty_img)
    assert crop is None
    assert bbox is None


def test_feature_extractor_dimensions():
    extractor = FeatureExtractor(hog_size=(64, 64))
    test_face = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    feat_hog = extractor.extract_features(test_face, mode="hog")
    assert feat_hog.shape == (1764,)

    feat_wavelet = extractor.extract_features(test_face, mode="wavelet")
    assert feat_wavelet.shape == (1024,)

    feat_lbp = extractor.extract_features(test_face, mode="lbp")
    assert feat_lbp.shape == (256,)

    feat_raw = extractor.extract_features(test_face, mode="raw")
    assert feat_raw.shape == (1024,)

    feat_combined = extractor.extract_features(test_face, mode="combined")
    assert feat_combined.shape == (1764 + 1024 + 256 + 1024,)  # 4068


def test_haar_wavelet_math():
    arr = np.random.rand(64, 64).astype(np.float32)
    decomp = haar_wavelet_2d(arr, level=2)
    assert decomp.shape == (64, 64)
    assert not np.isnan(decomp).any()


def test_lbp_histogram():
    gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    hist = extract_lbp_histogram(gray, num_bins=256)
    assert hist.shape == (256,)
    assert np.isclose(np.sum(hist), 1.0, atol=1e-4)


def test_player_info_known_and_fallback():
    kohli = get_player_info("virat_kohli")
    assert kohli["display_name"] == "Virat Kohli"
    assert kohli["jersey_number"] == 18

    dhoni = get_player_info("ms_dhoni")
    assert dhoni["display_name"] == "MS Dhoni"
    assert dhoni["jersey_number"] == 7

    unknown = get_player_info("unknown_cricketer")
    assert "Unknown Cricketer" in unknown["display_name"]
    assert unknown["jersey_number"] == "N/A"


def test_model_inference_on_sample():
    test_img_path = "data/test_samples/virat_kohli_test1.jpg"
    if not os.path.exists(test_img_path):
        pytest.skip("Test sample image not found.")

    classifier = CricketFaceClassifier(artifacts_dir="artifacts")
    res = classifier.predict_image(test_img_path)

    assert "predicted_name" in res
    assert res["predicted_name"] == "Virat Kohli"
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_predictions"]) == 6
    assert res["cropped_face"].shape == (64, 64, 3)
    assert res["annotated_image"].shape == cv2.imread(test_img_path).shape
