import io
import sys
from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image
import tensorflow as tf

from src.constants import CLASS_NAMES_PATH, DISEASE_INFO_PATH, IMAGE_SIZE, TOP_K
from src.components.model_loader import get_model
from src.exception import PlantDiseaseException
from src.logger import logger
from src.utils.common import parse_class_metadata, read_json


class PlantDiseasePredictor:
    """
    Image preprocessing, prediction, Top-K probability ranking,
    and pathology/remedy information enrichment component.
    """

    def __init__(self, model_path: str = None):
        try:
            self.model = get_model(model_path) if model_path else get_model()
            self.classes: List[str] = read_json(CLASS_NAMES_PATH)
            self.disease_info: Dict[str, Any] = read_json(DISEASE_INFO_PATH)
            logger.info(f"Predictor initialized with {len(self.classes)} classes.")
        except Exception as e:
            logger.error(f"Error initializing PlantDiseasePredictor: {e}")
            raise PlantDiseaseException(e, sys)

    def preprocess_image(self, image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> np.ndarray:
        """
        Preprocesses raw image into normalized float32 tensor matching EfficientNetB0 specifications:
        - Format: RGB
        - Dimensions: 224 x 224
        - Tensor Shape: (1, 224, 224, 3)
        """
        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            elif isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, io.BytesIO):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32)

            # Add batch dimension -> (1, 224, 224, 3)
            img_tensor = np.expand_dims(img_array, axis=0)
            return img_tensor

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise PlantDiseaseException(e, sys)

    def predict(self, image_input: Union[str, bytes, io.BytesIO, Image.Image], top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Runs model inference on preprocessed image and returns formatted diagnosis.
        """
        try:
            input_tensor = self.preprocess_image(image_input)
            raw_preds = self.model.predict(input_tensor, verbose=0)[0]

            # In case model output is logits, apply softmax
            if not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2):
                probabilities = tf.nn.softmax(raw_preds).numpy()
            else:
                probabilities = raw_preds

            # Top-K predictions indices
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            # Primary top-1 prediction
            top_idx = int(top_indices[0])
            top_class = self.classes[top_idx]
            top_confidence = float(probabilities[top_idx])

            meta = parse_class_metadata(top_class)
            info = self.disease_info.get(top_class, {
                "plant": meta["plant"],
                "disease_name": meta["condition"],
                "status": "Healthy" if meta["is_healthy"] else "Infected",
                "severity": "None" if meta["is_healthy"] else "Medium",
                "cause": "Unknown Pathogen",
                "symptoms": "Detailed symptoms not available.",
                "organic_treatment": "Consult agricultural extension specialist.",
                "chemical_treatment": "Standard protective broad-spectrum fungicide.",
                "prevention": "Ensure clean field sanitation and crop rotation."
            })

            # Format top-K predictions list
            top_k_list = []
            for rank, idx in enumerate(top_indices, 1):
                cls_name = self.classes[int(idx)]
                cls_meta = parse_class_metadata(cls_name)
                conf = float(probabilities[int(idx)])
                top_k_list.append({
                    "rank": rank,
                    "class_name": cls_name,
                    "display_name": cls_meta["display_name"],
                    "plant": cls_meta["plant"],
                    "condition": cls_meta["condition"],
                    "is_healthy": cls_meta["is_healthy"],
                    "confidence": round(conf * 100, 2),
                    "probability": round(conf, 4)
                })

            result = {
                "success": True,
                "prediction": {
                    "class_name": top_class,
                    "display_name": meta["display_name"],
                    "plant": meta["plant"],
                    "condition": meta["condition"],
                    "is_healthy": meta["is_healthy"],
                    "confidence": round(top_confidence * 100, 2),
                    "probability": round(top_confidence, 4),
                    "status": info.get("status", "Healthy" if meta["is_healthy"] else "Infected"),
                    "severity": info.get("severity", "None" if meta["is_healthy"] else "Medium"),
                    "cause": info.get("cause", "N/A"),
                    "symptoms": info.get("symptoms", ""),
                    "organic_treatment": info.get("organic_treatment", ""),
                    "chemical_treatment": info.get("chemical_treatment", ""),
                    "prevention": info.get("prevention", "")
                },
                "top_k_predictions": top_k_list,
                "model_info": {
                    "architecture": "EfficientNetB0",
                    "total_classes": len(self.classes),
                    "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
                }
            }

            logger.info(f"Diagnosis completed: {meta['display_name']} with confidence {result['prediction']['confidence']}%")
            return result

        except Exception as e:
            logger.error(f"Inference error in PlantDiseasePredictor: {e}")
            raise PlantDiseaseException(e, sys)
