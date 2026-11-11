"""
Face and landmark detection module using OpenCV Cascades.
Provides robust face localization, cropping, alignment margins, and visual annotations.
"""

import os
from typing import List, Dict, Optional, Tuple, Union
import cv2
import numpy as np


class FaceDetector:
    """Detects faces and eyes using OpenCV Haar Cascades."""

    def __init__(self):
        cascade_dir = cv2.data.haarcascades
        self.face_cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        self.eye_cascade_path = os.path.join(cascade_dir, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {self.face_cascade_path}")

    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 4,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Dict]:
        """
        Detects all frontal faces in the input BGR image.
        Returns a list of dicts:
            - 'bbox': (x, y, w, h)
            - 'confidence_eyes': int (number of eyes found)
            - 'roi_color': cropped BGR face
            - 'roi_gray': cropped Grayscale face
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Contrast optimization for detection
        gray_eq = cv2.equalizeHist(gray)

        # Primary pass
        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        # Secondary relaxed pass if no face found
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=min_size
            )

        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = image[y : y + h, x : x + w] if len(image.shape) == 3 else gray[y : y + h, x : x + w]

            # Eye verification in upper 65% of the face ROI
            eye_roi = roi_gray[0 : int(h * 0.65), :]
            eyes = self.eye_cascade.detectMultiScale(eye_roi, scaleFactor=1.1, minNeighbors=2)

            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "num_eyes": len(eyes),
                "has_two_eyes": len(eyes) >= 2,
                "roi_color": roi_color,
                "roi_gray": roi_gray,
                "method": "haar_cascade"
            })

        # Sort by area descending (largest face first)
        results.sort(key=lambda item: item["bbox"][2] * item["bbox"][3], reverse=True)
        return results

    def crop_primary_face(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (64, 64),
        margin: float = 0.12,
        fallback_center: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str]:
        """
        Extracts the primary (largest) face with a padding margin.
        Returns:
            - cropped_face_resized: np.ndarray (target_size)
            - bbox: (x, y, w, h) in original coordinates
            - method: 'haar_cascade' or 'center_crop' or 'none'
        """
        if image is None or image.size == 0:
            return None, None, "none"

        detections = self.detect_faces(image)
        h_img, w_img = image.shape[:2]

        if detections:
            x, y, w, h = detections[0]["bbox"]
            # Apply padding margin
            pad_x = int(w * margin)
            pad_y = int(h * margin)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                return resized, (x, y, w, h), "haar_cascade"

        if fallback_center:
            # Fallback to centered square crop if no cascade detection
            side = min(h_img, w_img)
            cx, cy = w_img // 2, h_img // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w_img, x1 + side)
            y2 = min(h_img, y1 + side)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                return resized, (x1, y1, x2 - x1, y2 - y1), "center_crop"

        return None, None, "none"

    @staticmethod
    def draw_prediction_overlay(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        player_name: str,
        confidence: float,
        color: Tuple[int, int, int] = (0, 165, 255)  # Indian Team Blue/Orange in BGR
    ) -> np.ndarray:
        """Draws a stylish bounded box with cricketer name and confidence percentage."""
        annotated = image.copy()
        x, y, w, h = bbox

        # Main bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Corner accents for a modern tech overlay
        corner_len = max(8, int(min(w, h) * 0.15))
        accent_color = (255, 200, 0)  # Gold/Cyan highlight
        t = 3
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), accent_color, t)
        cv2.line(annotated, (x, y), (x, y + corner_len), accent_color, t)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), accent_color, t)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), accent_color, t)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), accent_color, t)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), accent_color, t)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), accent_color, t)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), accent_color, t)

        # Label Banner
        text = f"{player_name} ({confidence * 100:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.7, w / 220.0))
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        banner_y1 = max(0, y - th - 12)
        banner_y2 = y

        # Banner background
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 14, banner_y2), (20, 20, 25), -1)
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 14, banner_y2), accent_color, 1)

        # Text inside banner
        cv2.putText(annotated, text, (x + 7, y - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

