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

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SatQuery AI Backend",
    description="AI-powered Satellite Image Change Detection API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "SatQuery AI Backend is running successfully!",
        "api_key_found": bool(API_KEY),
        "rasterio_available": RASTERIO_AVAILABLE
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "SatQuery AI Backend",
        "gemini_configured": bool(API_KEY),
        "rasterio_available": RASTERIO_AVAILABLE
    }


# ============================================================
# IMAGE HELPERS
# ============================================================

MAX_PREVIEW_SIZE = 1024


def is_tiff(filename: str) -> bool:

    filename = filename.lower()

    return filename.endswith(".tif") or filename.endswith(".tiff")


async def read_upload(upload: UploadFile):

    data = await upload.read()

    if not data:

        raise ValueError(
            f"{upload.filename} is empty."
        )

    return data


# ============================================================
# RASTERIO TIFF READER
# ============================================================

def read_tiff(data: bytes):

    if not RASTERIO_AVAILABLE:

        raise RuntimeError(
            "Rasterio is required for TIFF/GeoTIFF files. "
            "Run: pip install rasterio"
        )

    with rasterio.MemoryFile(data) as memfile:

        with memfile.open() as dataset:

            array = dataset.read(
                out_dtype="float32"
            )

            width = dataset.width
            height = dataset.height

            crs = str(dataset.crs) if dataset.crs else None

            transform = str(dataset.transform)

            bounds = {
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top
            }

            nodata = dataset.nodata

    return {
        "array": array,
        "width": width,
        "height": height,
        "bands": array.shape[0],
        "crs": crs,
        "transform": transform,
        "bounds": bounds,
        "nodata": nodata
    }


# ============================================================
# NORMAL IMAGE READER
# ============================================================

def read_normal_image(data: bytes):

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    width, height = image.size

    if max(width, height) > MAX_PREVIEW_SIZE:

        image.thumbnail(
            (MAX_PREVIEW_SIZE, MAX_PREVIEW_SIZE)
        )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    return {
        "image": image,
        "array": array,
        "width": image.width,
        "height": image.height
    }


# ============================================================
# TIFF → PREVIEW JPEG
# ============================================================

def tiff_to_preview(raster):

    array = raster["array"]

    # Use first band for a simple preview.
    band = array[0]

    valid = np.isfinite(band)

    if not np.any(valid):

        raise ValueError(
            "TIFF contains no valid numerical pixels."
        )

    values = band[valid]

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low:

        high = low + 1

    normalized = (
        (band - low) /
        (high - low)
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    normalized = (
        normalized * 255
    ).astype(np.uint8)

    image = Image.fromarray(
        normalized,
        mode="L"
    )

    if max(image.size) > MAX_PREVIEW_SIZE:

        image.thumbnail(
            (MAX_PREVIEW_SIZE, MAX_PREVIEW_SIZE)
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return buffer.getvalue()


# ============================================================
# NORMAL IMAGE → JPEG
# ============================================================

def image_to_preview(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return buffer.getvalue()


# ============================================================
# NUMERICAL CHANGE DETECTION
# ============================================================

def calculate_change(before, after):

    before_array = before
    after_array = after

    # Convert multi-band raster to mean intensity
    if before_array.ndim == 3:

        before_array = np.nanmean(
            before_array,
            axis=0
        )

    if after_array.ndim == 3:

        after_array = np.nanmean(
            after_array,
            axis=0
        )

    # Resize to common dimensions if necessary
    common_height = min(
        before_array.shape[0],
        after_array.shape[0]
    )

    common_width = min(
        before_array.shape[1],
        after_array.shape[1]
    )

    before_array = before_array[
        :common_height,
        :common_width
    ]

    after_array = after_array[
        :common_height,
        :common_width
    ]

    valid = (
        np.isfinite(before_array)
        &
        np.isfinite(after_array)
    )

    before_valid = before_array[valid]
    after_valid = after_array[valid]

    if before_valid.size == 0:

        raise ValueError(
            "No valid overlapping pixels were found."
        )

    difference = np.abs(
        after_valid - before_valid
    )

    mean_difference = float(
        np.mean(difference)
    )

    maximum_difference = float(
        np.max(difference)
    )

    before_mean = float(
        np.mean(before_valid)
    )

    after_mean = float(
        np.mean(after_valid)
    )

    mean_change = after_mean - before_mean

    # Adaptive threshold based on observed differences
    threshold = float(
        np.mean(difference)
        +
        2 * np.std(difference)
    )

    changed_pixels = int(
        np.sum(
            difference > threshold
        )
    )

    total_pixels = int(
        difference.size
    )

    change_percentage = (
        changed_pixels /
        total_pixels
    ) * 100

    return {

        "changed_pixels": changed_pixels,

        "total_pixels": total_pixels,

        "change_percentage": round(
            float(change_percentage),
            2
        ),

        "mean_absolute_change": round(
            mean_difference,
            4
        ),

        "maximum_change": round(
            maximum_difference,
            4
        ),

        "before_mean": round(
            before_mean,
            4
        ),

        "after_mean": round(
            after_mean,
            4
        ),

        "mean_change": round(
            mean_change,
            4
        ),

        "change_threshold": round(
            threshold,
            4
        )
    }


# ============================================================
# CONFIDENCE ESTIMATION
# ============================================================

def calculate_confidence(
    change_percentage,
    valid_pixels
):

    if valid_pixels < 100:

        return 30.0

    if change_percentage < 1:

        return 60.0

    if change_percentage < 5:

        return 70.0

    if change_percentage < 20:

        return 80.0

    return 90.0


# ============================================================
# MAIN ANALYSIS
# ============================================================

@app.post("/analyze-image")
async def analyze_image(

    before_file: UploadFile = File(...),

    after_file: UploadFile = File(...),

    question: str = Form(...)

):

    try:

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        if not API_KEY:

            return {

                "status": "error",

                "error":
                    "GEMINI_API_KEY was not found "
                    "in your .env file"

            }


        # ----------------------------------------------------
        # READ FILES
        # ----------------------------------------------------

        before_data = await read_upload(
            before_file
        )

        after_data = await read_upload(
            after_file
        )


        # ----------------------------------------------------
        # DETERMINE FILE TYPE
        # ----------------------------------------------------

        before_is_tiff = is_tiff(
            before_file.filename
        )

        after_is_tiff = is_tiff(
            after_file.filename
        )


        # ----------------------------------------------------
        # READ TIFF / NORMAL IMAGE
        # ----------------------------------------------------

        before_raster = None
        after_raster = None


        if before_is_tiff:

            before_raster = read_tiff(
                before_data
            )

            before_array = (
                before_raster["array"]
            )

            before_preview = tiff_to_preview(
                before_raster
            )

            before_width = (
                before_raster["width"]
            )

            before_height = (
                before_raster["height"]
            )

        else:

            before_image_data = (
                read_normal_image(
                    before_data
                )
            )

            before_array = (
                before_image_data["array"]
            )

            before_preview = (
                image_to_preview(
                    before_image_data["image"]
                )
            )

            before_width = (
                before_image_data["width"]
            )

            before_height = (
                before_image_data["height"]
            )


        if after_is_tiff:

            after_raster = read_tiff(
                after_data
            )

            after_array = (
                after_raster["array"]
            )

            after_preview = tiff_to_preview(
                after_raster
            )

            after_width = (
                after_raster["width"]
            )

            after_height = (
                after_raster["height"]
            )

        else:

            after_image_data = (
                read_normal_image(
                    after_data
                )
            )

            after_array = (
                after_image_data["array"]
            )

            after_preview = (
                image_to_preview(
                    after_image_data["image"]
                )
            )

            after_width = (
                after_image_data["width"]
            )

            after_height = (
                after_image_data["height"]
            )


        # ----------------------------------------------------
        # CALCULATE NUMERICAL CHANGE
        # ----------------------------------------------------

        change_metrics = calculate_change(
            before_array,
            after_array
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = calculate_confidence(

            change_metrics[
                "change_percentage"
            ],

            change_metrics[
                "total_pixels"
            ]

        )


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        client = genai.Client(
            api_key=API_KEY
        )


        # ----------------------------------------------------
        # DETAILED AI ANALYSIS PROMPT
        # ----------------------------------------------------

        prompt = f"""

You are SatQuery AI, an advanced satellite-image analysis assistant.

The FIRST image is the BEFORE image.
The SECOND image is the AFTER image.

User question:
{question}

Backend numerical change measurements:

Changed pixels:
{change_metrics["changed_pixels"]}

Total pixels:
{change_metrics["total_pixels"]}

Change percentage:
{change_metrics["change_percentage"]}%

Mean absolute change:
{change_metrics["mean_absolute_change"]}

Maximum change:
{change_metrics["maximum_change"]}

Before mean:
{change_metrics["before_mean"]}

After mean:
{change_metrics["after_mean"]}

Mean change:
{change_metrics["mean_change"]}

Change threshold:
{change_metrics["change_threshold"]}

Analyze BOTH images carefully and answer the user's question using visual evidence together with the backend measurements.

Your answer MUST contain exactly 7 large, detailed paragraphs.

Paragraph 1 — Overall finding:
Start with a clear overall explanation of what the images show and directly address the user's question. Give the reader a useful high-level understanding before going into details.

Paragraph 2 — Detailed visual evidence:
Describe the important visible features in the images. Discuss observable patterns involving land cover, vegetation, water, buildings, roads, infrastructure, agriculture, terrain, brightness, texture, or other relevant features. Only describe things that can actually be supported by the images.

Paragraph 3 — Location and spatial distribution:
Explain where the important differences appear within the image. You may describe regions using terms such as north, south, east, west, center, upper portion, lower portion, or other clearly visible spatial relationships. Do NOT invent coordinates, place names, addresses, or exact geographic locations.

Paragraph 4 — Before versus after comparison:
Make a detailed comparison between the BEFORE and AFTER images. Explain what appears to have changed, what appears to have remained similar, and how the visual evidence supports the comparison. If the images do not clearly establish a change, say so instead of guessing.

Paragraph 5 — Quantitative evidence:
Use the backend numerical measurements provided above to strengthen the analysis. Explain what the changed-pixel count, change percentage, mean absolute change, maximum change, before mean, after mean, mean change, and threshold indicate when relevant. Make it clear that these are image-level computational measurements. Do NOT convert pixels into hectares, square kilometres, acres, coordinates, or real-world area unless such information is explicitly provided by the backend.

Paragraph 6 — Interpretation:
Explain the most reasonable interpretation of the observed differences. Clearly separate what is directly visible from what is only a possible explanation. For example, changes in brightness or texture may have multiple causes such as imaging conditions, vegetation conditions, water conditions, construction, or other environmental factors. Never claim a specific cause with certainty unless the images provide sufficient evidence.

Paragraph 7 — Conclusion and limitations:
Give a strong concluding summary of the main finding. State what the visual evidence and numerical measurements collectively suggest. Also mention important limitations, including image quality, resolution, alignment, lighting or acquisition conditions, and the fact that the backend measurements are computational image-level evidence rather than independently validated scientific ground truth.

IMPORTANT RULES:

- Give exactly 7 substantial paragraphs.
- Each paragraph should be detailed and informative, not one or two sentences.
- Do not use bullet points.
- Do not use numbered lists.
- Do not use Markdown headings.
- Do not give short answers.
- Do not invent numbers.
- Do not invent hectares or square-kilometre measurements.
- Do not invent coordinates.
- Do not invent dates.
- Do not invent locations or place names.
- Do not invent roads, buildings, vegetation, water bodies, or other objects.
- Do not claim that a change is scientifically proven.
- Do not claim that the confidence value represents scientifically validated accuracy.
- Treat the backend numerical measurements as computational evidence.
- Clearly distinguish observation from interpretation.
- If something cannot be determined from the images, explicitly say that it cannot be determined reliably.
- Focus on evidence that a judge can understand from the uploaded images and backend measurements.

"""


        # ----------------------------------------------------
        # PREPARE BEFORE IMAGE FOR GEMINI
        # ----------------------------------------------------

        before_part = types.Part.from_bytes(

            data=before_preview,

            mime_type="image/jpeg"

        )


        # ----------------------------------------------------
        # PREPARE AFTER IMAGE FOR GEMINI
        # ----------------------------------------------------

        after_part = types.Part.from_bytes(

            data=after_preview,

            mime_type="image/jpeg"

        )


        # ----------------------------------------------------
        # GEMINI GENERATION
        # ----------------------------------------------------

                # ----------------------------------------------------
        # CALL GEMINI WITH RETRY FOR TEMPORARY 503 ERRORS
        # ----------------------------------------------------

        response = None

        max_retries = 3

        for attempt in range(max_retries):

            try:

                response = client.models.generate_content(

                    model="gemini-3.6-flash",

                    contents=[

                        prompt,

                        before_part,

                        after_part

                    ]

                )

                break

            except Exception as gemini_error:

                error_text = str(gemini_error)

                is_temporary_error = (

                    "503" in error_text

                    or

                    "UNAVAILABLE" in error_text

                    or

                    "high demand" in error_text

                )

                if not is_temporary_error:

                    raise

                if attempt == max_retries - 1:

                    raise

                wait_seconds = 5 * (2 ** attempt)

                print(

                    f"Gemini temporarily unavailable. "

                    f"Retrying in {wait_seconds} seconds..."

                )

                time.sleep(wait_seconds)


        # ----------------------------------------------------
        # GET AI ANSWER
        # ----------------------------------------------------

        ai_answer = response.text

        if not ai_answer:

            ai_answer = (
                "The AI did not return "
                "a text response."
            )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        result = {

            "status":
                "analysis_complete",

            "message":
                "Satellite image analysis "
                "completed successfully!",

            "ai_answer":
                ai_answer,

            "before_filename":
                before_file.filename,

            "after_filename":
                after_file.filename,

            "user_question":
                question,

            "before_image_width":
                before_width,

            "before_image_height":
                before_height,

            "after_image_width":
                after_width,

            "after_image_height":
                after_height,

            "change_metrics":
                change_metrics,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "before_format":
                "GeoTIFF"
                if before_is_tiff
                else "Image",

            "after_format":
                "GeoTIFF"
                if after_is_tiff
                else "Image"

        }


        # ----------------------------------------------------
        # ADD BEFORE GEOSPATIAL INFORMATION
        # ----------------------------------------------------

        if before_raster:

            result[
                "before_geospatial"
            ] = {

                "crs":
                    before_raster["crs"],

                "bounds":
                    before_raster["bounds"],

                "bands":
                    before_raster["bands"]

            }


        # ----------------------------------------------------
        # ADD AFTER GEOSPATIAL INFORMATION
        # ----------------------------------------------------

        if after_raster:

            result[
                "after_geospatial"
            ] = {

                "crs":
                    after_raster["crs"],

                "bounds":
                    after_raster["bounds"],

                "bands":
                    after_raster["bands"]

            }


        # ----------------------------------------------------
        # RETURN SUCCESS
        # ----------------------------------------------------

        return result


    except Exception as e:

        error_message = str(e)

        error_type = type(e).__name__

        full_error = traceback.format_exc()


        print(
            "\n========== SATQUERY ERROR =========="
        )

        print(full_error)

        print(
            "====================================\n"
        )


        return {

            "status": "error",

            "error_type":
                error_type,

            "error_message":
                error_message,

            "details":
                full_error

        }