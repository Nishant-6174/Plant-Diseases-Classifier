import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field

from src.constants import CLASS_NAMES_PATH, DISEASE_INFO_PATH
from src.exception import PlantDiseaseException
from src.logger import logger
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.common import decode_base64_image, parse_class_metadata, read_json
from src.utils.image_validator import validate_image
from src.utils.gradcam import (
    generate_gradcam_heatmap,
    heatmap_to_base64,
    overlay_heatmap_on_image,
    pil_to_base64,
)

# Global pipeline instance
_pipeline: Optional[PredictionPipeline] = None
START_TIME = time.time()


def get_pipeline() -> PredictionPipeline:
    """
    Lazy loader for PredictionPipeline ensuring thread-safe access.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing PredictionPipeline instance...")
        _pipeline = PredictionPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown event handling.
    """
    logger.info("Plant Disease Classifier Application starting up...")
    try:
        _ = get_pipeline()
        logger.info("PredictionPipeline warmed up and ready for traffic.")
    except Exception as e:
        logger.error(f"Lifespan initialization error: {e}")
    yield
    logger.info("Plant Disease Classifier Application shutting down.")


# Initialize FastAPI App
app = FastAPI(
    title="Plant Disease Classifier API",
    description="Production-grade Deep Learning API for detecting 86 categories of crop diseases using EfficientNetB0.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount Static Files & Jinja2 Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Pydantic Request Schemas
class Base64PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string (with or without data URI header)")
    top_k: Optional[int] = Field(5, description="Number of top predictions to return (1-10)")


# ============================================================================
# Web UI Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def index_view(request: Request):
    """
    Renders the modern Plant Disease Classifier web interface.
    """
    classes_list = read_json(CLASS_NAMES_PATH)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "total_classes": len(classes_list),
            "version": "1.0.0"
        }
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    System health check endpoint verifying model status and server uptime.
    """
    uptime_seconds = int(time.time() - START_TIME)
    pipe = get_pipeline()
    is_ready = pipe is not None and pipe.predictor.model is not None

    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "Plant Disease Classifier API",
        "model_loaded": is_ready,
        "total_classes": len(pipe.predictor.classes) if is_ready else 86,
        "uptime_seconds": uptime_seconds,
        "framework": "TensorFlow / Keras (EfficientNetB0)"
    }


@app.post("/predict", tags=["Inference"])
async def predict_file(
    file: UploadFile = File(..., description="Plant leaf image file (JPEG, PNG, WEBP)"),
    top_k: int = 5
):
    """
    Classify a plant leaf image by uploading a multipart file.
    Returns disease diagnosis, probability distribution, and comprehensive treatment remedy.
    """
    # Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg", "application/octet-stream"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a valid JPEG, PNG, or WEBP image."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── Input validation (conservative heuristic) ─────────────────────
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_valid, reason = validate_image(pil_image)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_image",
                    "message": "Invalid image. Please upload a clear image of a plant leaf.",
                },
            )

        pipe = get_pipeline()
        result = pipe.predict_image(contents, top_k=top_k)
        return JSONResponse(content=result, status_code=200)

    except PlantDiseaseException as pde:
        logger.error(f"Prediction exception: {pde}")
        raise HTTPException(status_code=500, detail=str(pde))
    except Exception as e:
        logger.error(f"Unexpected prediction failure: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-base64", tags=["Inference"])
async def predict_base64_endpoint(payload: Base64PredictionRequest):
    """
    Classify an image provided as a Base64 string (useful for webcams or mobile clients).
    """
    try:
        pipe = get_pipeline()

        # ── Decode Base64 manually to retain the original PIL image for Grad-CAM ──
        original_pil = decode_base64_image(payload.image)

        # ── Input validation (conservative heuristic) ─────────────────────
        is_valid, reason = validate_image(original_pil)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_image",
                    "message": "Invalid image. Please upload a clear image of a plant leaf.",
                },
            )

        # ── Standard prediction (must not fail due to Grad-CAM) ──────────
        prediction_result = pipe.predict_image(original_pil, top_k=payload.top_k or 5)

        # ── Grad-CAM computation ─────────────────────────────────────────
        gradcam_payload = {
            "success": False,
            "target_layer": "top_activation",
            "target_class": None,
            "original_image": None,
            "heatmap": None,
            "overlay": None,
            "error": None
        }

        try:
            # Preprocess exactly as the predictor does
            predictor = pipe.predictor
            img_tensor = predictor.preprocess_image(original_pil)   # shape (1, 224, 224, 3)

            # Determine predicted class index from prediction result
            pred_class_name = prediction_result["prediction"]["class_name"]
            predicted_class_idx = predictor.classes.index(pred_class_name)

            # Generate Grad-CAM heatmap
            heatmap, used_class_idx = generate_gradcam_heatmap(
                model=predictor.model,
                img_tensor=img_tensor,
                class_idx=predicted_class_idx
            )

            # Encode original image as base64
            original_b64 = pil_to_base64(original_pil)

            # Encode coloured heatmap as base64
            heatmap_b64 = heatmap_to_base64(
                heatmap=heatmap,
                target_size=original_pil.size   # (width, height)
            )

            # Encode blended overlay as base64
            overlay_b64 = overlay_heatmap_on_image(
                original_pil=original_pil,
                heatmap=heatmap,
                alpha=0.45
            )

            gradcam_payload.update({
                "success": True,
                "target_class": used_class_idx,
                "original_image": original_b64,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64,
            })
            logger.info(f"Grad-CAM generated successfully for class_idx={used_class_idx}")

        except Exception as gcam_err:
            # Grad-CAM failure must NOT kill the prediction response
            logger.error(f"Grad-CAM generation failed (prediction still returned): {gcam_err}")
            gradcam_payload["error"] = str(gcam_err)

        prediction_result["gradcam"] = gradcam_payload
        return JSONResponse(content=prediction_result, status_code=200)

    except PlantDiseaseException as pde:
        logger.error(f"Base64 prediction exception: {pde}")
        raise HTTPException(status_code=400, detail=str(pde))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-with-gradcam", tags=["Inference"])
async def predict_with_gradcam(
    file: UploadFile = File(..., description="Plant leaf image (JPEG, PNG, WEBP)"),
    top_k: int = 5
):
    """
    Classify a plant leaf image AND generate Grad-CAM explainability maps.

    Returns the standard prediction result PLUS:
    - gradcam.original_image  — base64 encoded original image
    - gradcam.heatmap         — base64 Jet-coloured Grad-CAM heatmap
    - gradcam.overlay         — base64 Grad-CAM heatmap blended onto original
    - gradcam.target_layer    — name of the convolutional layer used
    - gradcam.target_class    — predicted class index used for gradients
    - gradcam.success         — whether Grad-CAM succeeded
    - gradcam.error           — error message if Grad-CAM failed (prediction still returned)
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg", "application/octet-stream"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload JPEG, PNG, or WEBP."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── Input validation (conservative heuristic) ─────────────────────
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_valid, reason = validate_image(pil_image)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_image",
                    "message": "Invalid image. Please upload a clear image of a plant leaf.",
                },
            )

        # ── Standard prediction (must not fail due to Grad-CAM) ──────────
        pipe = get_pipeline()
        prediction_result = pipe.predict_image(contents, top_k=top_k)

        # ── Grad-CAM computation ─────────────────────────────────────────
        gradcam_payload = {
            "success": False,
            "target_layer": "top_activation",
            "target_class": None,
            "original_image": None,
            "heatmap": None,
            "overlay": None,
            "error": None
        }

        try:
            # Reconstruct PIL image from raw bytes (original dimensions)
            original_pil = Image.open(io.BytesIO(contents)).convert("RGB")

            # Preprocess exactly as the predictor does
            predictor = pipe.predictor
            img_tensor = predictor.preprocess_image(contents)   # shape (1, 224, 224, 3)

            # Determine predicted class index from prediction result
            pred_class_name = prediction_result["prediction"]["class_name"]
            predicted_class_idx = predictor.classes.index(pred_class_name)

            # Generate Grad-CAM heatmap
            heatmap, used_class_idx = generate_gradcam_heatmap(
                model=predictor.model,
                img_tensor=img_tensor,
                class_idx=predicted_class_idx
            )

            # Encode original image as base64
            original_b64 = pil_to_base64(original_pil)

            # Encode coloured heatmap as base64
            heatmap_b64 = heatmap_to_base64(
                heatmap=heatmap,
                target_size=original_pil.size   # (width, height)
            )

            # Encode blended overlay as base64
            overlay_b64 = overlay_heatmap_on_image(
                original_pil=original_pil,
                heatmap=heatmap,
                alpha=0.45
            )

            gradcam_payload.update({
                "success": True,
                "target_class": used_class_idx,
                "original_image": original_b64,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64,
            })
            logger.info(f"Grad-CAM generated successfully for class_idx={used_class_idx}")

        except Exception as gcam_err:
            # Grad-CAM failure must NOT kill the prediction response
            logger.error(f"Grad-CAM generation failed (prediction still returned): {gcam_err}")
            gradcam_payload["error"] = str(gcam_err)

        prediction_result["gradcam"] = gradcam_payload
        return JSONResponse(content=prediction_result, status_code=200)

    except PlantDiseaseException as pde:
        logger.error(f"Grad-CAM endpoint exception: {pde}")
        raise HTTPException(status_code=500, detail=str(pde))
    except Exception as e:
        logger.error(f"Unexpected error in /predict-with-gradcam: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", tags=["Inference"])
async def batch_predict_endpoint(
    files: List[UploadFile] = File(..., description="List of plant leaf images"),
    top_k: int = 3
):
    """
    Run batch inference on multiple plant leaf images concurrently.
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum batch size is 20 images.")

    try:
        image_bytes_list = []
        for f in files:
            image_bytes_list.append(await f.read())

        pipe = get_pipeline()
        results = pipe.predict_batch(image_bytes_list, top_k=top_k)
        return JSONResponse(content={"total_images": len(files), "results": results}, status_code=200)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classes", tags=["Encyclopedia"])
async def get_all_classes():
    """
    Returns list of all 86 supported plant disease categories with parsed metadata.
    """
    classes_list = read_json(CLASS_NAMES_PATH)
    formatted = [parse_class_metadata(cls) for cls in classes_list]
    return {
        "count": len(formatted),
        "classes": formatted
    }


@app.get("/api/class-info/{class_name}", tags=["Encyclopedia"])
async def get_class_info(class_name: str):
    """
    Returns pathology, symptom breakdown, and treatment guidelines for a specific disease class.
    """
    info_db = read_json(DISEASE_INFO_PATH)
    if class_name in info_db:
        return info_db[class_name]
    else:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found.")


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching server on http://0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
