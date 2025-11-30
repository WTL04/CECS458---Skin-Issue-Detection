# backend/main.py

import io
import os
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
from ultralytics import YOLO

from openai import OpenAI

# ---------- Config ----------
# Make sure OPENAI_API_KEY is set in your environment
# e.g., export OPENAI_API_KEY="sk-..."
client = OpenAI()

MODEL_PATH = "skin_yolo.pt"  # update if needed

# ---------- App setup ----------
app = FastAPI(title="Skin Issue Detection API")

# CORS so your frontend (http://localhost:8000 or 5500 etc.) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load YOLO model once ----------
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    yolo_model = None


class Detection(BaseModel):
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    detections: List[Detection]
    gpt_routine: str


def run_yolo_on_image(pil_image: Image.Image) -> List[Detection]:
    """
    Run YOLO on the image and return a list of detection objects.
    """
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    results = yolo_model(pil_image, verbose=False)
    detections: List[Detection] = []

    if not results:
        return detections

    result = results[0]
    boxes = result.boxes

    if boxes is None or boxes.data is None:
        return detections

    names = yolo_model.names  # class index -> label string

    for box in boxes:
        cls_idx = int(box.cls.item())
        conf = float(box.conf.item())
        label = names.get(cls_idx, f"class_{cls_idx}")
        detections.append(Detection(label=label, confidence=conf))

    return detections


def call_gpt_for_routine(detections: List[Detection]) -> str:
    """
    Turn detections into a high-level text description and ask GPT
    for an educational skincare routine (with disclaimers).
    """

    if not detections:
        detected_text = "The model did not detect any specific skin issues."
    else:
        # e.g. "acne (0.87), redness (0.75)"
        issues = ", ".join(
            f"{d.label} (confidence {d.confidence:.2f})" for d in detections
        )
        detected_text = f"The model detected these visible skin issues: {issues}."

    system_prompt = (
        "You are a cautious skincare educator. "
        "You can talk about general skincare routines and over-the-counter product categories. "
        "Do NOT diagnose diseases or suggest prescription-only treatments. "
        "Always include a clear disclaimer that this information is not medical advice "
        "and that the user should see a dermatologist or healthcare professional for diagnosis or treatment."
    )

    user_prompt = (
        f"{detected_text}\n\n"
        "Using only general skincare knowledge and non-prescription product categories, "
        "suggest a simple morning and night skincare routine, plus a short explanation "
        "of what each step does. Assume the user is an adult with no known medical conditions. "
        "Keep it concise and easy to understand."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # or another model your class allows
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )

    return completion.choices[0].message.content.strip()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image (from webcam snapshot), runs YOLO, then asks GPT for a routine.
    """

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # YOLO detections
    try:
        detections = run_yolo_on_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference error: {e}")

    # GPT routine
    try:
        routine = call_gpt_for_routine(detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return AnalyzeResponse(detections=detections, gpt_routine=routine)
