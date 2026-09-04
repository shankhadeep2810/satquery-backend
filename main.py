import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import io
import numpy as np


# Load .env from the same folder as main.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found!")

client = genai.Client(api_key=api_key)

app = FastAPI(
    title="SatQuery AI Backend",
    description="AI-powered satellite image analysis system"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "SatQuery AI Backend is running successfully!"
    }


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    # Read uploaded image
    image_data = await file.read()

    # Open image
    image = Image.open(
        io.BytesIO(image_data)
    ).convert("RGB")

    # Resize large images for faster AI analysis

    MAX_SIZE = 768

    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))

    # Create AI prompt
    prompt = f"""
You are SatQuery, an AI assistant specialized in satellite imagery analysis.

Analyze this satellite image carefully and answer the user's question.

User question: {question}

Give a clear, useful answer based only on what can reasonably be observed in the image.
"""

    # Get AI response
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[prompt, image]
    )

    ai_answer = response.text

    # Convert image into numerical array
    image_array = np.array(image)

    # Get image information
    width, height = image.size

    # Calculate basic statistics
    average_brightness = float(np.mean(image_array))
    minimum_pixel = int(np.min(image_array))
    maximum_pixel = int(np.max(image_array))

    # Return result
    return {
        "ai_answer": ai_answer,
        "filename": file.filename,
        "user_question": question,
        "image_width": width,
        "image_height": height,
        "average_brightness": round(average_brightness, 2),
        "minimum_pixel_value": minimum_pixel,
        "maximum_pixel_value": maximum_pixel,
        "message": "Satellite image processed successfully!",
        "status": "analysis_complete"
    }