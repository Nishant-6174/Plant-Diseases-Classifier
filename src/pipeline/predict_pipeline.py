import sys
from typing import Any, Dict, List, Union
from PIL import Image

from src.components.predictor import PlantDiseasePredictor
from src.exception import PlantDiseaseException
from src.logger import logger
from src.utils.common import decode_base64_image


class PredictionPipeline:
    """
    End-to-End Prediction Pipeline orchestrating requests from API endpoints or client scripts.
    """

    def __init__(self, model_path: str = None):
        try:
            self.predictor = PlantDiseasePredictor(model_path=model_path)
            logger.info("PredictionPipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing PredictionPipeline: {e}")
            raise PlantDiseaseException(e, sys)

    def predict_image(self, image_data: Union[bytes, Image.Image, str], top_k: int = 5) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes, PIL image, or image path.
        """
        try:
            return self.predictor.predict(image_data, top_k=top_k)
        except Exception as e:
            logger.error(f"Error in predict_image pipeline: {e}")
            raise PlantDiseaseException(e, sys)

    def predict_base64(self, base64_str: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Decodes base64 string and runs inference.
        """
        try:
            image = decode_base64_image(base64_str)
            return self.predictor.predict(image, top_k=top_k)
        except Exception as e:
            logger.error(f"Error in predict_base64 pipeline: {e}")
            raise PlantDiseaseException(e, sys)

    def predict_batch(self, image_list: List[Union[bytes, Image.Image, str]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Runs batch prediction over a list of image inputs.
        """
        try:
            results = []
            for idx, img in enumerate(image_list):
                try:
                    res = self.predictor.predict(img, top_k=top_k)
                    results.append({"index": idx, "success": True, "result": res})
                except Exception as item_err:
                    results.append({"index": idx, "success": False, "error": str(item_err)})
            return results
        except Exception as e:
            logger.error(f"Error in predict_batch pipeline: {e}")
            raise PlantDiseaseException(e, sys)
