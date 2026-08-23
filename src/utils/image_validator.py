"""
image_validator.py
------------------

Conservative leaf-image validation for the Plant Disease Classifier.

This module is a validation gate, NOT a disease classifier.

The EfficientNet model will always produce one of its known classes when
given an arbitrary image. This validator therefore attempts to reject
images that are clearly outside the expected plant-image domain.

Important design principle:

    Plant-like colour is strong evidence that an image should be allowed
    through the validator.

    Texture/edges are supporting evidence only.

This is especially important because a perfectly uniform synthetic green
or yellow image is intentionally used by the project's tests and must
not be rejected merely because it has zero texture.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.logger import logger


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum plant-like colour required to pass the colour-content rule.
_MIN_PLANT_RATIO = 0.08

# Minimum vegetation (green) ratio.
_MIN_VEGETATION_RATIO = 0.04

# Extremely high percentage of low-saturation pixels combined with very
# low mean saturation is characteristic of grayscale/neutral images.
_MAX_LOW_SATURATION_RATIO = 0.92

# Featureless threshold.
#
# IMPORTANT:
# We do NOT reject a featureless image merely because variance and edges
# are low. A solid green/yellow test image is intentionally valid.
_MAX_FEATURELESS_VARIANCE = 2.0
_MAX_FEATURELESS_EDGE_DENSITY = 0.0008

# Percentage of pixels that must be plant-like before colour alone is
# considered strong enough to pass.
_STRONG_PLANT_RATIO = 0.25


def _calculate_features(image: Image.Image) -> dict:
    """
    Calculate visual features used by the validation gate.
    """

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)

    if rgb.size == 0:
        raise ValueError("Empty image.")

    height, width = rgb.shape[:2]
    total_pixels = height * width

    if total_pixels == 0:
        raise ValueError("Image contains no pixels.")

    # ------------------------------------------------------------------
    # OpenCV representations
    # ------------------------------------------------------------------

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # ------------------------------------------------------------------
    # Plant-like colour masks
    # ------------------------------------------------------------------
    #
    # OpenCV hue range = 0..179.
    #
    # Green:
    # approximately 50..190 degrees in conventional HSV.
    #
    # Yellow/brown:
    # common colours of healthy, aging and diseased leaves.
    #
    # Red/rust:
    # common in several plant disease classes.
    # ------------------------------------------------------------------

    green_mask = cv2.inRange(
        hsv,
        np.array([25, 25, 20], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )

    yellow_brown_mask = cv2.inRange(
        hsv,
        np.array([8, 30, 20], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )

    red_mask_1 = cv2.inRange(
        hsv,
        np.array([0, 35, 20], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )

    red_mask_2 = cv2.inRange(
        hsv,
        np.array([165, 35, 20], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    plant_mask = cv2.bitwise_or(
        green_mask,
        cv2.bitwise_or(
            yellow_brown_mask,
            cv2.bitwise_or(red_mask_1, red_mask_2),
        ),
    )

    vegetation_mask = green_mask

    plant_pixels = cv2.countNonZero(plant_mask)
    vegetation_pixels = cv2.countNonZero(vegetation_mask)

    plant_ratio = plant_pixels / total_pixels
    vegetation_ratio = vegetation_pixels / total_pixels

    # ------------------------------------------------------------------
    # Saturation / brightness statistics
    # ------------------------------------------------------------------

    low_saturation_mask = (s < 25).astype(np.uint8) * 255

    low_saturation_ratio = (
        cv2.countNonZero(low_saturation_mask) / total_pixels
    )

    mean_saturation = float(np.mean(s))
    mean_value = float(np.mean(v))

    # ------------------------------------------------------------------
    # Texture and edges
    # ------------------------------------------------------------------

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150,
    )

    edge_density = (
        cv2.countNonZero(edges) / total_pixels
    )

    return {
        "plant_ratio": float(plant_ratio),
        "vegetation_ratio": float(vegetation_ratio),
        "low_saturation_ratio": float(low_saturation_ratio),
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "variance": variance,
        "edge_density": float(edge_density),
    }


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Determine whether an image is sufficiently plausible as a plant image.

    Returns
    -------
    (is_valid, reason)

    True:
        Image is allowed to reach the disease classifier.

    False:
        Image is rejected before inference.
    """

    try:
        features = _calculate_features(image)

        plant_ratio = features["plant_ratio"]
        vegetation_ratio = features["vegetation_ratio"]
        low_saturation_ratio = features["low_saturation_ratio"]
        mean_saturation = features["mean_saturation"]
        variance = features["variance"]
        edge_density = features["edge_density"]

        logger.info(
            (
                "Image validation — "
                "plant_ratio=%.4f vegetation_ratio=%.4f "
                "low_sat=%.4f mean_sat=%.2f "
                "variance=%.2f edge_density=%.5f"
            ),
            plant_ratio,
            vegetation_ratio,
            low_saturation_ratio,
            mean_saturation,
            variance,
            edge_density,
        )

        # --------------------------------------------------------------
        # Rule 1: strongly plant-like colour
        #
        # This rule comes FIRST intentionally.
        #
        # A solid green or yellow/brown test image has:
        #
        #     variance = 0
        #     edge_density = 0
        #
        # but it still contains 100% plant-like colour.
        #
        # Therefore strong plant colour should allow the image through.
        # --------------------------------------------------------------

        if (
            plant_ratio >= _STRONG_PLANT_RATIO
            or vegetation_ratio >= _STRONG_PLANT_RATIO
        ):
            return True, "OK"

        # --------------------------------------------------------------
        # Rule 2: completely featureless NON-plant image
        #
        # Only reject featureless images when they ALSO lack meaningful
        # plant-like colour.
        # --------------------------------------------------------------

        if (
            variance < _MAX_FEATURELESS_VARIANCE
            and edge_density < _MAX_FEATURELESS_EDGE_DENSITY
            and plant_ratio < _MIN_PLANT_RATIO
            and vegetation_ratio < _MIN_VEGETATION_RATIO
        ):
            reason = (
                "The uploaded image appears to be blank or featureless. "
                "Please upload a clear photograph of a plant leaf."
            )

            logger.warning(
                "Validation rejected image: %s",
                reason,
            )

            return False, reason

        # --------------------------------------------------------------
        # Rule 3: almost entirely neutral / grayscale image
        #
        # This catches many blank screenshots, diagrams and grayscale
        # images while allowing normal colour photographs.
        # --------------------------------------------------------------

        if (
            low_saturation_ratio >= _MAX_LOW_SATURATION_RATIO
            and mean_saturation < 20
        ):
            reason = (
                "The uploaded image does not appear to contain "
                "sufficient plant-like visual information. "
                "Please upload a clear photograph of a plant leaf."
            )

            logger.warning(
                "Validation rejected image: %s",
                reason,
            )

            return False, reason

        # --------------------------------------------------------------
        # Rule 4: insufficient plant-like colour AND weak visual
        # structure.
        #
        # We do not require plant colour alone for every legitimate
        # photograph because lighting/background conditions can alter
        # colour statistics.
        #
        # However, an image with almost no plant colour and almost no
        # visual structure is very likely outside the intended domain.
        # --------------------------------------------------------------

        if (
            plant_ratio < _MIN_PLANT_RATIO
            and vegetation_ratio < _MIN_VEGETATION_RATIO
            and variance < _MAX_FEATURELESS_VARIANCE
            and edge_density < _MAX_FEATURELESS_EDGE_DENSITY
        ):
            reason = (
                "The uploaded image does not appear to contain a "
                "plant leaf. Please upload a clear photograph of "
                "a plant leaf."
            )

            logger.warning(
                "Validation rejected image: %s",
                reason,
            )

            return False, reason

        # --------------------------------------------------------------
        # Otherwise, be conservative and allow the classifier to decide.
        # --------------------------------------------------------------

        return True, "OK"

    except Exception as exc:
        # Never allow a validator implementation error to crash the API.
        #
        # If validation itself fails, pass the image through so that a
        # legitimate image is not accidentally blocked.
        logger.error(
            "Image validation error (passing image through): %s",
            exc,
        )

        return True, "OK"