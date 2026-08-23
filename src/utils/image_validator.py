"""
image_validator.py
------------------
A conservative heuristic filter for the Plant Disease Classifier API.

PURPOSE:
    Reject images that are obviously inconsistent with plant-leaf inputs
    (e.g., solid blue/gray/black frames, completely white images).

    This is NOT a reliable leaf-vs-non-leaf classifier. It cannot reliably
    distinguish between a car photo and a heavily diseased leaf using colour
    and texture statistics alone.  Its sole goal is:

        "Reject images that are clearly NOT plant leaves while minimising
         false rejection of legitimate (healthy or diseased) leaf images."

    When in doubt, the validator PASSES the image to EfficientNet.

APPROACH (three independent checks, all must fail to trigger rejection):
    1. Organic-colour ratio  – the fraction of pixels whose HSV hue falls in
       the range associated with plants (greens, yellows, browns, rust reds).
    2. Laplacian variance    – a zero-variance image is a pixel-perfect solid
       block with no texture whatsoever.
    3. Canny edge density    – a featureless image has essentially no edges.

    Rejection only occurs when the image clearly and unambiguously fails
    an extreme threshold (e.g. 0 % organic pixels AND 0 edge pixels).
    Any image that is even slightly ambiguous is passed through.
"""
import cv2
import numpy as np
from PIL import Image

from src.logger import logger


# ---------------------------------------------------------------------------
# Thresholds – deliberately extreme / permissive
# ---------------------------------------------------------------------------

# Minimum fraction of pixels with "plant-like" colour.
# Set to 0 so a completely monochromatic non-plant image (pure blue, pure
# gray) is caught ONLY when combined with zero texture + zero edges.
_MIN_PLANT_RATIO: float = 0.0

# A Laplacian variance of < 1.0 indicates a perfectly uniform block of
# colour (every pixel identical). Real photographs never reach this.
_MAX_FEATURELESS_VARIANCE: float = 1.0

# Canny edge density threshold: < this means essentially no edges at all.
_MAX_FEATURELESS_EDGE_DENSITY: float = 0.0005


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Determine whether an image is plausibly a plant-leaf photograph.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to validate (any mode; converted to RGB internally).

    Returns
    -------
    (is_valid, reason) : (bool, str)
        is_valid – True  → pass the image to the disease classifier.
                   False → reject; do not run disease prediction.
        reason   – human-readable explanation (logged and returned in the
                   API response when the image is rejected).
    """
    try:
        # ── Convert to numpy RGB array ──────────────────────────────────────
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if rgb.size == 0:
            return False, "Empty image."

        bgr = rgb[:, :, ::-1]  # OpenCV expects BGR

        total_pixels: int = rgb.shape[0] * rgb.shape[1]

        # ── 1. Texture (Laplacian variance) ─────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        variance: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ── 2. Edge density (Canny) ─────────────────────────────────────────
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density: float = float(cv2.countNonZero(edges)) / total_pixels

        # ── 3. Organic-colour ratio (HSV) ───────────────────────────────────
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Greens / yellows / browns: hue 10–100 °
        mask_gb = cv2.inRange(hsv, np.array([10, 15, 15]), np.array([100, 255, 255]))
        # Rust / brick reds (hue wraps): 0–10 °  and  165–180 °
        mask_r1 = cv2.inRange(hsv, np.array([0, 15, 15]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([165, 15, 15]), np.array([180, 255, 255]))
        plant_mask = cv2.bitwise_or(mask_gb, cv2.bitwise_or(mask_r1, mask_r2))
        plant_ratio: float = float(cv2.countNonZero(plant_mask)) / total_pixels

        logger.info(
            "Image validation — plant_ratio=%.4f  variance=%.2f  edge_density=%.5f",
            plant_ratio, variance, edge_density,
        )

        # ── Rejection rule ───────────────────────────────────────────────────
        # Only reject when ALL THREE signals indicate an obviously featureless,
        # non-organic image.  This keeps the filter extremely conservative.
        if (
            plant_ratio <= _MIN_PLANT_RATIO
            and variance < _MAX_FEATURELESS_VARIANCE
            and edge_density < _MAX_FEATURELESS_EDGE_DENSITY
        ):
            reason = (
                "Image does not appear to contain plant-like colours or texture. "
                "Please upload a clear photograph of a plant leaf."
            )
            logger.warning("Validation rejected image: %s", reason)
            return False, reason

        return True, "OK"

    except Exception as exc:
        # If something goes wrong in the validator itself, be conservative:
        # pass the image through rather than blocking a legitimate upload.
        logger.error("Image validation error (passing image through): %s", exc)
        return True, "OK"
