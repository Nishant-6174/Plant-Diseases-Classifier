import os
import sys
import threading
from typing import Optional

import numpy as np
import tensorflow as tf

from src.constants import MODEL_PATH, IMAGE_SIZE
from src.exception import PlantDiseaseException
from src.logger import logger


class CompatibleDense(tf.keras.layers.Dense):
    """
    Compatibility wrapper for Dense layers saved with a newer Keras version.

    Removes quantization_config from the serialized configuration when
    loading an older Keras model.
    """

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("quantization_config", None)
        return super().from_config(config)


class ModelLoader:
    """
    Thread-safe Singleton Model Loader for EfficientNetB0 plant disease classifier.
    """

    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._model = None
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self, model_path: str = MODEL_PATH):
        if self._initialized:
            return

        self.model_path = model_path
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:

        try:

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at path: {self.model_path}"
                )

            logger.info(
                f"Loading trained model from: {self.model_path} ..."
            )

            self._model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
                custom_objects={
                    "Dense": CompatibleDense
                }
            )

            logger.info(
                "Model loaded into memory successfully!"
            )

            dummy_input = np.zeros(
                (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                dtype=np.float32
            )

            _ = self._model.predict(
                dummy_input,
                verbose=0
            )

            logger.info(
                "Model warm-up inference completed successfully!"
            )

        except Exception as e:

            logger.error(
                f"Failed to load model from {self.model_path}: {e}"
            )

            raise PlantDiseaseException(e, sys)

    @property
    def model(self) -> tf.keras.Model:

        if self._model is None:
            self._load_model()

        return self._model


def get_model(
    model_path: str = MODEL_PATH
) -> tf.keras.Model:

    loader = ModelLoader(
        model_path=model_path
    )

    return loader.model