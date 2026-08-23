import os

# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# Artifacts & Models
MODEL_NAME = "plant_disease_efficientnetb0_compatible.keras"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
CLASS_NAMES_PATH = os.path.join(DATA_DIR, "disease_classes.json")
DISEASE_INFO_PATH = os.path.join(DATA_DIR, "disease_info.json")

# Model configuration
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
TOP_K = 5

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000