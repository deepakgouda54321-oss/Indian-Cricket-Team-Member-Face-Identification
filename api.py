"""
FastAPI REST API for Indian Cricket Team Face Identification.
Provides HTTP endpoints for image classification, player roster metadata, and performance metrics.
Run with: uvicorn api:app --port 8000 --reload
"""

import os
import sys
import base64
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from predict import CricketFaceClassifier
from src.player_info import PLAYERS_INFO

app = FastAPI(
    title="Indian Cricket Team Face Identification API",
    description="REST API for facial biometric recognition and player identification of Indian cricket stars.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None


@app.on_event("startup")
def startup_event():
    global classifier
    try:
        classifier = CricketFaceClassifier(artifacts_dir="artifacts")
        print("[+] REST API: Classifier and artifacts successfully loaded.")
    except Exception as e:
        print(f"[!] REST API startup warning: {e}")


class Base64ImageRequest(BaseModel):
    image_base64: str


class PredictionResponse(BaseModel):
    status: str
    predicted_player: str
    player_key: str
    confidence_percentage: float
    bounding_box: Dict[str, int]
    top_ranked_predictions: List[Dict[str, Any]]
    player_metadata: Dict[str, Any]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "indian-cricket-face-id-api", "version": "1.0.0"}


@app.get("/players")
def list_squad_members():
    """Returns the full roster of supported Indian Cricket Team members."""
    return {"squad_count": len(PLAYERS_INFO), "players": PLAYERS_INFO}


@app.get("/metrics")
def get_evaluation_metrics():
    """Returns benchmark accuracy, precision, recall, and F1-scores across all algorithms."""
    metrics_path = "artifacts/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics artifact not found. Please run train.py first.")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return {"status": "success", "metrics": data}


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_from_file_upload(file: UploadFile = File(...)):
    """Classifies a cricketer face from an uploaded image file (JPEG, PNG, WEBP)."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier model is not initialized.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image format.")

    try:
        res = classifier.predict_image(image)
        x, y, w, h = res["bbox"]

        top_list = [
            {"player": name, "probability": prob, "percentage": round(prob * 100, 2)}
            for name, prob in res["top_predictions"]
        ]

        return PredictionResponse(
            status="success",
            predicted_player=res["predicted_name"],
            player_key=res["player_key"],
            confidence_percentage=round(res["confidence"] * 100, 2),
            bounding_box={"x": x, "y": y, "width": w, "height": h},
            top_ranked_predictions=top_list,
            player_metadata=res["player_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference error: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
def predict_from_base64(payload: Base64ImageRequest):
    """Classifies a cricketer face from a base64-encoded image string."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier model is not initialized.")

    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        decoded = base64.b64decode(raw_b64)
        nparr = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Decoded image is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    try:
        res = classifier.predict_image(image)
        x, y, w, h = res["bbox"]
        top_list = [
            {"player": name, "probability": prob, "percentage": round(prob * 100, 2)}
            for name, prob in res["top_predictions"]
        ]

        return PredictionResponse(
            status="success",
            predicted_player=res["predicted_name"],
            player_key=res["player_key"],
            confidence_percentage=round(res["confidence"] * 100, 2),
            bounding_box={"x": x, "y": y, "width": w, "height": h},
            top_ranked_predictions=top_list,
            player_metadata=res["player_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
