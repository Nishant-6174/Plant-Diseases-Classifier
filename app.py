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
from src.utils.common import (
    decode_base64_image,
    parse_class_metadata,
    read_json,
)
from src.utils.image_validator import validate_image
from src.utils.gradcam import (
    generate_gradcam_heatmap,
    heatmap_to_base64,
    overlay_heatmap_on_image,
    pil_to_base64,
)


# ============================================================================
# GLOBALS
# ============================================================================

_pipeline: Optional[PredictionPipeline] = None
START_TIME = time.time()


def get_pipeline() -> PredictionPipeline:
    """
    Lazy loader for PredictionPipeline.
    """
    global _pipeline

    if _pipeline is None:
        logger.info("Initializing PredictionPipeline instance...")
        _pipeline = PredictionPipeline()

    return _pipeline


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI startup and shutdown lifecycle.
    """

    logger.info("Plant Disease Classifier Application starting up...")

    try:
        _ = get_pipeline()
        logger.info("PredictionPipeline warmed up and ready for traffic.")

    except Exception as e:
        logger.error(f"Lifespan initialization error: {e}")

    yield

    logger.info("Plant Disease Classifier Application shutting down.")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Plant Disease Classifier API",
    description=(
        "Production-grade Deep Learning API for detecting "
        "86 categories of crop diseases using EfficientNetB0."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class Base64PredictionRequest(BaseModel):
    image: str = Field(
        ...,
        description=(
            "Base64 encoded image string "
            "(with or without data URI header)"
        ),
    )

    top_k: Optional[int] = Field(
        5,
        description="Number of top predictions to return (1-10)",
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

ALLOWED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
    "application/octet-stream",
]


def validate_uploaded_plant_image(
    pil_image: Image.Image,
):
    """
    Run image validation.

    CLIP plant/non-plant validation is optional and controlled
    by the ENABLE_CLIP_GATE environment variable.

    Local development:
        ENABLE_CLIP_GATE=true

    Render deployment:
        ENABLE_CLIP_GATE=false

    The existing image validator always runs.

    Returns:
        (True, None) if valid.

    Otherwise:
        (False, JSONResponse)
    """

    # ------------------------------------------------------------------------
    # STEP 1: OPTIONAL CLIP PLANT GATE
    # ------------------------------------------------------------------------

    enable_clip_gate = os.getenv(
        "ENABLE_CLIP_GATE",
        "false",
    ).lower() == "true"

    if enable_clip_gate:

        try:
            # Lazy import:
            # torch and transformers are loaded only when
            # CLIP validation is actually enabled.
            from src.utils.clip_plant_gate import (
                validate_plant_image
            )

            is_plant, plant_confidence = (
                validate_plant_image(
                    pil_image
                )
            )

        except Exception as exc:

            logger.error(
                f"CLIP plant validation failed: {exc}"
            )

            return False, JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_type": "plant_validation_error",
                    "message": (
                        "Unable to validate whether the "
                        "image contains a plant."
                    ),
                },
            )

        if not is_plant:

            logger.warning(
                "Non-plant image rejected by CLIP gate. "
                f"Confidence={plant_confidence:.4f}"
            )

            return False, JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_image",
                    "message": (
                        "This image does not appear to contain "
                        "a plant. Please upload a clear plant "
                        "leaf image."
                    ),
                    "plant_confidence": round(
                        plant_confidence,
                        4,
                    ),
                },
            )

    else:

        logger.info(
            "CLIP plant gate disabled. "
            "Using existing image validator."
        )

    # ------------------------------------------------------------------------
    # STEP 2: EXISTING IMAGE VALIDATOR
    # ------------------------------------------------------------------------

    is_valid, reason = validate_image(
        pil_image
    )

    if not is_valid:

        logger.warning(
            f"Image rejected by image validator: {reason}"
        )

        return False, JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "invalid_image",
                "message": (
                    "Invalid image. Please upload a "
                    "clear image of a plant leaf."
                ),
                "reason": reason,
            },
        )

    return True, None


# ============================================================================
# WEB UI
# ============================================================================

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["Web Interface"],
)
async def index_view(request: Request):
    """
    Render the Plant Disease Classifier web interface.
    """

    classes_list = read_json(
        CLASS_NAMES_PATH
    )

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "total_classes": len(classes_list),
            "version": "1.0.0",
        },
    )


# ============================================================================
# HEALTH
# ============================================================================

@app.get(
    "/health",
    tags=["System"],
)
async def health_check():
    """
    System health check.
    """

    uptime_seconds = int(
        time.time() - START_TIME
    )

    pipe = get_pipeline()

    is_ready = (
        pipe is not None
        and pipe.predictor.model is not None
    )

    return {
        "status": (
            "healthy"
            if is_ready
            else "degraded"
        ),
        "service": "Plant Disease Classifier API",
        "model_loaded": is_ready,
        "total_classes": (
            len(pipe.predictor.classes)
            if is_ready
            else 86
        ),
        "uptime_seconds": uptime_seconds,
        "framework": (
            "TensorFlow / Keras (EfficientNetB0)"
        ),
    }


# ============================================================================
# /PREDICT
# ============================================================================

@app.post(
    "/predict",
    tags=["Inference"],
)
async def predict_file(
    file: UploadFile = File(
        ...,
        description=(
            "Plant leaf image file "
            "(JPEG, PNG, WEBP)"
        ),
    ),
    top_k: int = 5,
):
    """
    Classify a plant leaf image.

    CLIP plant gate runs when ENABLE_CLIP_GATE=true.
    Existing image validation always runs.
    """

    # ------------------------------------------------------------------------
    # FILE TYPE
    # ------------------------------------------------------------------------

    if (
        file.content_type
        and file.content_type.lower()
        not in ALLOWED_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type: "
                f"{file.content_type}. "
                "Please upload a valid JPEG, PNG, "
                "or WEBP image."
            ),
        )

    try:

        # --------------------------------------------------------------------
        # READ FILE
        # --------------------------------------------------------------------

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # --------------------------------------------------------------------
        # OPEN IMAGE
        # --------------------------------------------------------------------

        try:
            pil_image = (
                Image.open(
                    io.BytesIO(contents)
                ).convert("RGB")
            )

        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_image",
                    "message": (
                        "The uploaded file is not "
                        "a valid image."
                    ),
                },
            )

        # --------------------------------------------------------------------
        # CLIP + IMAGE VALIDATION
        # --------------------------------------------------------------------

        is_valid, error_response = (
            validate_uploaded_plant_image(
                pil_image
            )
        )

        if not is_valid:
            return error_response

        # --------------------------------------------------------------------
        # PREDICTION
        # --------------------------------------------------------------------

        pipe = get_pipeline()

        result = pipe.predict_image(
            contents,
            top_k=top_k,
        )

        return JSONResponse(
            content=result,
            status_code=200,
        )

    except PlantDiseaseException as pde:

        logger.error(
            f"Prediction exception: {pde}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(pde),
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"Unexpected prediction failure: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ============================================================================
# /PREDICT-BASE64
# ============================================================================

@app.post(
    "/predict-base64",
    tags=["Inference"],
)
async def predict_base64_endpoint(
    payload: Base64PredictionRequest,
):
    """
    Classify a Base64 encoded image.

    CLIP plant gate runs when ENABLE_CLIP_GATE=true.
    Grad-CAM is generated after successful prediction.
    """

    try:

        # --------------------------------------------------------------------
        # DECODE IMAGE
        # --------------------------------------------------------------------

        original_pil = decode_base64_image(
            payload.image
        )

        # --------------------------------------------------------------------
        # CLIP + IMAGE VALIDATION
        # --------------------------------------------------------------------

        is_valid, error_response = (
            validate_uploaded_plant_image(
                original_pil
            )
        )

        if not is_valid:
            return error_response

        # --------------------------------------------------------------------
        # PIPELINE
        # --------------------------------------------------------------------

        pipe = get_pipeline()

        prediction_result = pipe.predict_image(
            original_pil,
            top_k=payload.top_k or 5,
        )

        # --------------------------------------------------------------------
        # GRAD-CAM
        # --------------------------------------------------------------------

        gradcam_payload = {
            "success": False,
            "target_layer": "top_activation",
            "target_class": None,
            "original_image": None,
            "heatmap": None,
            "overlay": None,
            "error": None,
        }

        try:

            predictor = pipe.predictor

            img_tensor = (
                predictor.preprocess_image(
                    original_pil
                )
            )

            pred_class_name = (
                prediction_result[
                    "prediction"
                ]["class_name"]
            )

            predicted_class_idx = (
                predictor.classes.index(
                    pred_class_name
                )
            )

            heatmap, used_class_idx = (
                generate_gradcam_heatmap(
                    model=predictor.model,
                    img_tensor=img_tensor,
                    class_idx=predicted_class_idx,
                )
            )

            original_b64 = (
                pil_to_base64(
                    original_pil
                )
            )

            heatmap_b64 = (
                heatmap_to_base64(
                    heatmap=heatmap,
                    target_size=original_pil.size,
                )
            )

            overlay_b64 = (
                overlay_heatmap_on_image(
                    original_pil=original_pil,
                    heatmap=heatmap,
                    alpha=0.45,
                )
            )

            gradcam_payload.update(
                {
                    "success": True,
                    "target_class": used_class_idx,
                    "original_image": original_b64,
                    "heatmap": heatmap_b64,
                    "overlay": overlay_b64,
                }
            )

            logger.info(
                "Grad-CAM generated successfully "
                f"for class_idx={used_class_idx}"
            )

        except Exception as gcam_err:

            logger.error(
                "Grad-CAM generation failed "
                f"(prediction still returned): "
                f"{gcam_err}"
            )

            gradcam_payload["error"] = str(
                gcam_err
            )

        prediction_result["gradcam"] = (
            gradcam_payload
        )

        return JSONResponse(
            content=prediction_result,
            status_code=200,
        )

    except PlantDiseaseException as pde:

        logger.error(
            f"Base64 prediction exception: {pde}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(pde),
        )

    except Exception as e:

        logger.error(
            f"Unexpected error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================================
# /PREDICT-WITH-GRADCAM
# ============================================================================

@app.post(
    "/predict-with-gradcam",
    tags=["Inference"],
)
async def predict_with_gradcam(
    file: UploadFile = File(
        ...,
        description=(
            "Plant leaf image "
            "(JPEG, PNG, WEBP)"
        ),
    ),
    top_k: int = 5,
):
    """
    Classify a plant leaf and generate Grad-CAM.

    CLIP plant gate runs when ENABLE_CLIP_GATE=true.
    Existing image validation always runs.
    """

    if (
        file.content_type
        and file.content_type.lower()
        not in ALLOWED_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type: "
                f"{file.content_type}. "
                "Please upload JPEG, PNG, or WEBP."
            ),
        )

    try:

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # --------------------------------------------------------------------
        # OPEN IMAGE
        # --------------------------------------------------------------------

        original_pil = (
            Image.open(
                io.BytesIO(contents)
            ).convert("RGB")
        )

        # --------------------------------------------------------------------
        # CLIP + IMAGE VALIDATION
        # --------------------------------------------------------------------

        is_valid, error_response = (
            validate_uploaded_plant_image(
                original_pil
            )
        )

        if not is_valid:
            return error_response

        # --------------------------------------------------------------------
        # PREDICTION
        # --------------------------------------------------------------------

        pipe = get_pipeline()

        prediction_result = pipe.predict_image(
            contents,
            top_k=top_k,
        )

        # --------------------------------------------------------------------
        # GRAD-CAM
        # --------------------------------------------------------------------

        gradcam_payload = {
            "success": False,
            "target_layer": "top_activation",
            "target_class": None,
            "original_image": None,
            "heatmap": None,
            "overlay": None,
            "error": None,
        }

        try:

            predictor = pipe.predictor

            img_tensor = (
                predictor.preprocess_image(
                    contents
                )
            )

            pred_class_name = (
                prediction_result[
                    "prediction"
                ]["class_name"]
            )

            predicted_class_idx = (
                predictor.classes.index(
                    pred_class_name
                )
            )

            heatmap, used_class_idx = (
                generate_gradcam_heatmap(
                    model=predictor.model,
                    img_tensor=img_tensor,
                    class_idx=predicted_class_idx,
                )
            )

            original_b64 = (
                pil_to_base64(
                    original_pil
                )
            )

            heatmap_b64 = (
                heatmap_to_base64(
                    heatmap=heatmap,
                    target_size=original_pil.size,
                )
            )

            overlay_b64 = (
                overlay_heatmap_on_image(
                    original_pil=original_pil,
                    heatmap=heatmap,
                    alpha=0.45,
                )
            )

            gradcam_payload.update(
                {
                    "success": True,
                    "target_class": used_class_idx,
                    "original_image": original_b64,
                    "heatmap": heatmap_b64,
                    "overlay": overlay_b64,
                }
            )

            logger.info(
                "Grad-CAM generated successfully "
                f"for class_idx={used_class_idx}"
            )

        except Exception as gcam_err:

            logger.error(
                "Grad-CAM generation failed "
                f"(prediction still returned): "
                f"{gcam_err}"
            )

            gradcam_payload["error"] = str(
                gcam_err
            )

        prediction_result["gradcam"] = (
            gradcam_payload
        )

        return JSONResponse(
            content=prediction_result,
            status_code=200,
        )

    except PlantDiseaseException as pde:

        logger.error(
            f"Grad-CAM endpoint exception: {pde}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(pde),
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"Unexpected error in "
            f"/predict-with-gradcam: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {str(e)}"
            ),
        )


# ============================================================================
# /BATCH-PREDICT
# ============================================================================

@app.post(
    "/batch-predict",
    tags=["Inference"],
)
async def batch_predict_endpoint(
    files: List[UploadFile] = File(
        ...,
        description="List of plant leaf images",
    ),
    top_k: int = 3,
):
    """
    Run batch prediction.

    Every image is checked by the optional CLIP plant gate
    before being passed to the disease classifier.
    """

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum batch size is 20 images."
            ),
        )

    try:

        image_bytes_list = []

        for f in files:

            if (
                f.content_type
                and f.content_type.lower()
                not in ALLOWED_IMAGE_TYPES
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid file type for "
                        f"{f.filename}."
                    ),
                )

            contents = await f.read()

            if len(contents) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File {f.filename} is empty."
                    ),
                )

            try:

                pil_image = (
                    Image.open(
                        io.BytesIO(contents)
                    ).convert("RGB")
                )

            except Exception:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"File {f.filename} is not "
                        "a valid image."
                    ),
                )

            # ---------------------------------------------------------------
            # OPTIONAL CLIP + EXISTING VALIDATION
            # ---------------------------------------------------------------

            is_valid, error_response = (
                validate_uploaded_plant_image(
                    pil_image
                )
            )

            if not is_valid:
                return error_response

            image_bytes_list.append(
                contents
            )

        # --------------------------------------------------------------------
        # BATCH PREDICTION
        # --------------------------------------------------------------------

        pipe = get_pipeline()

        results = pipe.predict_batch(
            image_bytes_list,
            top_k=top_k,
        )

        return JSONResponse(
            content={
                "total_images": len(files),
                "results": results,
            },
            status_code=200,
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"Batch prediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================================
# CLASSES
# ============================================================================

@app.get(
    "/api/classes",
    tags=["Encyclopedia"],
)
async def get_all_classes():
    """
    Return all 86 supported disease categories.
    """

    classes_list = read_json(
        CLASS_NAMES_PATH
    )

    formatted = [
        parse_class_metadata(cls)
        for cls in classes_list
    ]

    return {
        "count": len(formatted),
        "classes": formatted,
    }


# ============================================================================
# CLASS INFORMATION
# ============================================================================

@app.get(
    "/api/class-info/{class_name}",
    tags=["Encyclopedia"],
)
async def get_class_info(
    class_name: str,
):
    """
    Return pathology, symptoms and treatment
    information for a disease class.
    """

    info_db = read_json(
        DISEASE_INFO_PATH
    )

    if class_name in info_db:

        return info_db[class_name]

    raise HTTPException(
        status_code=404,
        detail=(
            f"Class '{class_name}' not found."
        ),
    )


# ============================================================================
# RUN DIRECTLY
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000",
        )
    )

    logger.info(
        f"Launching server on "
        f"http://0.0.0.0:{port}"
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )