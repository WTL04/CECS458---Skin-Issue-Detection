# backend/main.py

import io
import os
import base64
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from openai import OpenAI

# ---------- Config ----------

# Uses OPENAI_API_KEY from your environment by default
#   export OPENAI_API_KEY="sk-..."
client = OpenAI()

# YOLO model path 
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "runs/notebook_train14/weights/best.pt")

# Default model for skincare routine text
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")


# ---------- App setup ----------

app = FastAPI(title="Skin Issue Detection API")

# CORS so your frontend (http://localhost:5500, 8000, etc.) can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict to your real frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Load YOLO model once ----------

try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"✅ Loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO model from {MODEL_PATH}: {e}")
    yolo_model = None


# ---------- Pydantic models ----------

class Detection(BaseModel):
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    detections: List[Detection]
    gpt_routine: str
    annotated_image: str  # base64-encoded PNG of the annotated image


# ---------- Helper functions ----------

def run_yolo_on_image(pil_image: Image.Image) -> Tuple[List[Detection], str]:
    """
    Run YOLO on the image and return:
      - list of Detection objects
      - base64 PNG string of the annotated image
    """
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    # Run YOLO
    results = yolo_model(pil_image)
    detections: List[Detection] = []

    if not results:
        # No results from the model: just return original image as "annotated"
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return detections, annotated_b64

    result = results[0]
    boxes = result.boxes
    names = yolo_model.names  # class index -> label string

    # Extract detections
    if boxes is not None and boxes.data is not None:
        for box in boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            label = names.get(cls_idx, f"class_{cls_idx}")
            detections.append(Detection(label=label, confidence=conf))

    # Create annotated image using YOLO's built-in plotting
    plotted = result.plot()  # NumPy array (BGR)

    if isinstance(plotted, np.ndarray):
        # Convert BGR -> RGB then to PIL
        annotated_pil = Image.fromarray(plotted[..., ::-1])
    else:
        # Fallback: if something weird happens, just use original image
        annotated_pil = pil_image

    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return detections, annotated_b64


def call_gpt_for_routine(detections: List[Detection]) -> str:
    """
    Turn YOLO detections into a high-level text description and
    ask the OpenAI model for a gentle, non-diagnostic skincare routine.
    """

    if not detections:
        detected_text = (
            "The model did not detect any specific skin issues with high confidence."
        )
    else:
        issues = ", ".join(
            f"{d.label} ({d.confidence:.2f})" for d in detections
        )
        detected_text = f"The model detected the following possible skin issues: {issues}."

    system_prompt = (
        "You are a cautious skincare educator. "
        "You can talk about general skincare routines and over-the-counter product categories. "
        "Do NOT diagnose health conditions or mention prescription drugs. "
        "Always remind the user to consult a dermatologist or other healthcare professional for diagnosis and treatment."
    )

    user_prompt = (
        f"{detected_text}\n\n"
        "Based on these possible skin issues, suggest a simple morning and night routine "
        "using only general, over-the-counter skincare product categories "
        "(like gentle cleanser, non-comedogenic moisturizer, sunscreen, etc.). "
        "For each step, briefly explain what it does. "
        "End with a clear disclaimer that this is not a diagnosis and they should consult a professional."
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Recommended convenience property for plain-text output
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback in case output_text is missing/empty
    try:
        first = response.output[0].content[0].text
        value = getattr(first, "value", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass

    return (
        "Sorry, I couldn't generate a skincare routine right now. "
        "Please try again later or consult a dermatologist for personalized advice."
    )


# ---------- FastAPI endpoint ----------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image (from webcam snapshot), runs YOLO,
    then asks GPT for a routine and returns everything.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Read and open the image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # YOLO detections + annotated image
    try:
        detections, annotated_b64 = run_yolo_on_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference error: {e}")

    # GPT routine
    try:
        routine = call_gpt_for_routine(detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return AnalyzeResponse(
        detections=detections,
        gpt_routine=routine,
        annotated_image=annotated_b64,
    )
