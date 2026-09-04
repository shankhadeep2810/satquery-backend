import os
import io
import traceback

import numpy as np

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

from google import genai
from google.genai import types


# Load .env file
load_dotenv()


# Get Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")


# Create FastAPI application
app = FastAPI(
    title="SatQuery AI Backend",
    description="AI-powered Satellite Image Change Detection API"
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home route
@app.get("/")
def home():

    return {
        "message": "SatQuery AI Backend is running successfully!",
        "api_key_found": bool(API_KEY)
    }


# Analyze satellite images
@app.post("/analyze-image")
async def analyze_image(

    before_file: UploadFile = File(...),

    after_file: UploadFile = File(...),

    question: str = Form(...)

):

    try:

        # Check API key
        if not API_KEY:

            return {
                "status": "error",
                "error": "GEMINI_API_KEY was not found in your .env file"
            }


        # Read BEFORE image
        before_data = await before_file.read()


        # Read AFTER image
        after_data = await after_file.read()


        # Check images are not empty
        if not before_data:

            return {
                "status": "error",
                "error": "BEFORE image is empty"
            }


        if not after_data:

            return {
                "status": "error",
                "error": "AFTER image is empty"
            }


        # Open BEFORE image for validation
        before_image = Image.open(
            io.BytesIO(before_data)
        ).convert("RGB")


        # Open AFTER image for validation
        after_image = Image.open(
            io.BytesIO(after_data)
        ).convert("RGB")


        # Resize images for faster processing
        MAX_SIZE = 1024


        if max(before_image.size) > MAX_SIZE:

            before_image.thumbnail(
                (MAX_SIZE, MAX_SIZE)
            )


        if max(after_image.size) > MAX_SIZE:

            after_image.thumbnail(
                (MAX_SIZE, MAX_SIZE)
            )


        # Save resized BEFORE image to memory
        before_buffer = io.BytesIO()

        before_image.save(
            before_buffer,
            format="JPEG",
            quality=90
        )

        before_image_bytes = before_buffer.getvalue()


        # Save resized AFTER image to memory
        after_buffer = io.BytesIO()

        after_image.save(
            after_buffer,
            format="JPEG",
            quality=90
        )

        after_image_bytes = after_buffer.getvalue()


        # Create Gemini client
        client = genai.Client(
            api_key=API_KEY
        )


        # Create AI prompt
        prompt = f"""
You are SatQuery, an AI assistant specialized in satellite imagery analysis and change detection.

You are given TWO satellite images of the same geographical area.

The FIRST image is the BEFORE image.

The SECOND image is the AFTER image.

Compare the two images carefully.

User question:

{question}

Identify only meaningful changes that can actually be observed.

Possible changes include:

1. Urban Expansion

New buildings, construction, settlements, or developed areas.

2. Vegetation Changes

Increase or decrease in vegetation or green areas.

3. Agricultural Changes

Changes in farmland, crop areas, or agricultural patterns.

4. Land Use Changes

Changes in how land appears to be used.

5. Water Changes

Changes in rivers, lakes, ponds, coastlines, or water bodies.

6. Infrastructure Changes

New roads, transport infrastructure, or major construction.

7. Environmental Changes

Other clearly visible environmental changes.

IMPORTANT RULES:

Only describe changes that can reasonably be observed.

Do not invent information.

If no clear change can be confidently identified, clearly say so.

Return plain text only.

Do not use Markdown.

Do not use hashtags.

Do not use asterisks.

Do not use bold formatting.

Use numbered sections.

After every numbered section title, leave exactly one blank line before the explanation.

Leave exactly one blank line between sections.

Make the response clear and easy to read in a web application.
"""


        # Create BEFORE image part
        before_part = types.Part.from_bytes(
            data=before_image_bytes,
            mime_type="image/jpeg"
        )


        # Create AFTER image part
        after_part = types.Part.from_bytes(
            data=after_image_bytes,
            mime_type="image/jpeg"
        )


        # Send prompt and BOTH images to Gemini
        response = client.models.generate_content(

            model="gemini-3.6-flash",

            contents=[
                prompt,
                before_part,
                after_part
            ]

        )


        # Get AI response safely
        ai_answer = response.text


        if not ai_answer:

            ai_answer = "The AI did not return a text response."


        # Convert images to arrays
        before_array = np.array(before_image)

        after_array = np.array(after_image)


        # Get image dimensions
        before_width, before_height = before_image.size

        after_width, after_height = after_image.size


        # Calculate brightness
        before_brightness = float(
            np.mean(before_array)
        )

        after_brightness = float(
            np.mean(after_array)
        )


        # Return successful result
        return {

            "status": "analysis_complete",

            "message": "Satellite image analysis completed successfully!",

            "ai_answer": ai_answer,

            "before_filename": before_file.filename,

            "after_filename": after_file.filename,

            "user_question": question,

            "before_image_width": before_width,

            "before_image_height": before_height,

            "after_image_width": after_width,

            "after_image_height": after_height,

            "before_average_brightness": round(
                before_brightness,
                2
            ),

            "after_average_brightness": round(
                after_brightness,
                2
            )

        }


    except Exception as e:

        error_message = str(e)

        error_type = type(e).__name__

        full_error = traceback.format_exc()


        print("\n========== SATQUERY ERROR ==========")

        print(full_error)

        print("====================================\n")


        return {

            "status": "error",

            "error_type": error_type,

            "error_message": error_message,

            "details": full_error

        }