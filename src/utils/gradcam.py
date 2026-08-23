"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNetB0.

Target layer: 'top_activation' inside the 'efficientnetb0' sub-model.
  - This is the final Swish activation after top_conv + top_bn.
  - At 224×224 input, its output is shape (1, 7, 7, 1280) — rich spatial feature maps.
  - Standard, well-established Grad-CAM target for EfficientNet variants.

Compatible with: TensorFlow 2.21.0 / Keras 3.15.0
"""

import base64
import io
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from src.exception import PlantDiseaseException
from src.logger import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRADCAM_LAYER_NAME = "top_activation"      # Target layer inside EfficientNetB0
BASE_MODEL_NAME = "efficientnetb0"         # Nested Functional sub-model name


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------

def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    class_idx: Optional[int] = None,
    layer_name: str = GRADCAM_LAYER_NAME
) -> Tuple[np.ndarray, int]:
    """
    Computes the Grad-CAM heatmap for the given image and class index.

    Pipeline:
      1. Build a sub-model that outputs (conv_features, final_logits) simultaneously.
      2. Record conv_features on the GradientTape.
      3. Forward through the classification head manually using model layers.
      4. Compute dLoss/d(conv_features) via backprop.
      5. Global-average-pool the gradients → weight vector per filter.
      6. Weighted sum of feature maps → raw CAM.
      7. ReLU + normalize to [0, 1].

    Args:
        model:       The full loaded Keras model.
        img_tensor:  Preprocessed image tensor, shape (1, 224, 224, 3), float32.
        class_idx:   Target class index. If None, uses argmax of predictions.
        layer_name:  Conv layer name inside EfficientNetB0 to target.

    Returns:
        (heatmap, class_idx) — heatmap is float32 array in [0, 1], shape (H, W).
    """
    try:
        # ── 1. Retrieve the base EfficientNetB0 sub-model ──────────────────
        base_model = model.get_layer(BASE_MODEL_NAME)

        # ── 2. Retrieve the target conv layer ──────────────────────────────
        try:
            conv_layer = base_model.get_layer(layer_name)
        except ValueError as ve:
            available = [l.name for l in base_model.layers
                         if isinstance(l, (tf.keras.layers.Conv2D,
                                           tf.keras.layers.Activation,
                                           tf.keras.layers.DepthwiseConv2D))]
            logger.error(
                f"Layer '{layer_name}' not found in EfficientNetB0. "
                f"Available conv/activation layers: {available}"
            )
            raise PlantDiseaseException(
                ValueError(f"Grad-CAM target layer '{layer_name}' not found."), sys
            )

        # ── 3. Build a sub-model: base_model_input → top_activation output ─
        conv_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=conv_layer.output,
            name="gradcam_conv_extractor"
        )

        img_tf = tf.cast(img_tensor, tf.float32)

        with tf.GradientTape() as tape:
            # Handle optional data augmentation layer (disabled at inference)
            if "data_augmentation" in [l.name for l in model.layers]:
                x = model.get_layer("data_augmentation")(img_tf, training=False)
            else:
                x = img_tf

            # Extract convolutional feature maps and watch them
            conv_features = conv_model(x, training=False)
            tape.watch(conv_features)

            # Forward through classification head layers
            h = model.get_layer("global_average_pooling2d_1")(conv_features)
            h = model.get_layer("batch_normalization_1")(h, training=False)
            h = model.get_layer("dropout_2")(h, training=False)
            h = model.get_layer("dense_2")(h)
            h = model.get_layer("dropout_3")(h, training=False)
            logits = model.get_layer("dense_3")(h)

            # Determine target class
            if class_idx is None:
                class_idx = int(tf.argmax(logits[0]))

            # Scalar loss: score for the target class
            loss = logits[:, class_idx]

        # ── 4. Gradients of class score w.r.t. feature maps ───────────────
        grads = tape.gradient(loss, conv_features)   # shape (1, H, W, C)

        if grads is None:
            raise ValueError(
                "GradientTape returned None gradients. "
                "Ensure the model graph is connected through the target layer."
            )

        # ── 5. Global average pooling of gradients → per-filter weights ────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)

        # ── 6. Weighted combination of feature maps ─────────────────────────
        # conv_features[0]: (H, W, C), pooled_grads: (C,)
        cam = tf.reduce_sum(conv_features[0] * pooled_grads, axis=-1)  # (H, W)

        # ── 7. ReLU + normalize ────────────────────────────────────────────
        cam = tf.maximum(cam, 0.0)
        max_val = tf.reduce_max(cam)
        if max_val > 0:
            cam = cam / max_val

        return cam.numpy(), class_idx

    except PlantDiseaseException:
        raise
    except Exception as e:
        logger.error(f"Grad-CAM computation failed: {e}")
        raise PlantDiseaseException(e, sys)


# ---------------------------------------------------------------------------
# Heatmap-only image (colourized, no blend)
# ---------------------------------------------------------------------------

def heatmap_to_base64(
    heatmap: np.ndarray,
    target_size: Tuple[int, int]
) -> str:
    """
    Converts a raw [0,1] heatmap array into a coloured JPEG image
    encoded as a base64 data URI.

    Args:
        heatmap:     2-D float32 array in [0, 1].
        target_size: (width, height) to resize the heatmap to.

    Returns:
        Base64 data URI string: "data:image/jpeg;base64,..."
    """
    try:
        # Resize heatmap to target image dimensions
        h_img = Image.fromarray(np.uint8(255 * heatmap))
        h_resized = h_img.resize(target_size, Image.Resampling.BICUBIC)
        h_arr = np.array(h_resized, dtype=np.float32) / 255.0

        # Jet colormap: Blue → Cyan → Green → Yellow → Red
        r = np.clip(1.5 - np.abs(2.0 * h_arr - 1.5) * 2.0, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(2.0 * h_arr - 1.0) * 2.0, 0.0, 1.0)
        b = np.clip(1.5 - np.abs(2.0 * h_arr - 0.5) * 2.0, 0.0, 1.0)

        coloured = np.stack([r, g, b], axis=-1)
        coloured_uint8 = np.uint8(coloured * 255)

        pil_img = Image.fromarray(coloured_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    except Exception as e:
        logger.error(f"heatmap_to_base64 failed: {e}")
        raise PlantDiseaseException(e, sys)


# ---------------------------------------------------------------------------
# Overlay: heatmap blended onto original image
# ---------------------------------------------------------------------------

def overlay_heatmap_on_image(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"          # kept for API compatibility; only jet used
) -> str:
    """
    Overlays the Grad-CAM heatmap on the original PIL image using Jet colormap.

    Args:
        original_pil: The original (un-resized) PIL image in RGB.
        heatmap:      2-D float32 array in [0, 1] (from generate_gradcam_heatmap).
        alpha:        Blending weight for the heatmap overlay (0 = original only).
        colormap:     Kept for compatibility; Jet is always used.

    Returns:
        Base64 data URI string: "data:image/jpeg;base64,..."
    """
    try:
        orig_rgb = original_pil.convert("RGB")
        target_size = orig_rgb.size  # (width, height)

        # Resize heatmap to match original image dimensions
        h_img = Image.fromarray(np.uint8(255 * heatmap))
        h_resized = h_img.resize(target_size, Image.Resampling.BICUBIC)
        h_arr = np.array(h_resized, dtype=np.float32) / 255.0

        # Jet colormap
        r = np.clip(1.5 - np.abs(2.0 * h_arr - 1.5) * 2.0, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(2.0 * h_arr - 1.0) * 2.0, 0.0, 1.0)
        b = np.clip(1.5 - np.abs(2.0 * h_arr - 0.5) * 2.0, 0.0, 1.0)
        coloured = np.stack([r, g, b], axis=-1) * 255.0   # [0, 255]

        # Blend with original
        orig_arr = np.array(orig_rgb, dtype=np.float32)
        blended = (1.0 - alpha) * orig_arr + alpha * coloured
        blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        blended_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        blended_pil.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    except Exception as e:
        logger.error(f"overlay_heatmap_on_image failed: {e}")
        raise PlantDiseaseException(e, sys)


# ---------------------------------------------------------------------------
# Original image → Base64 helper
# ---------------------------------------------------------------------------

def pil_to_base64(pil_image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """
    Encodes a PIL image as a base64 data URI.
    """
    try:
        rgb = pil_image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format=fmt, quality=quality)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
        return f"data:image/{mime};base64,{encoded}"
    except Exception as e:
        logger.error(f"pil_to_base64 failed: {e}")
        raise PlantDiseaseException(e, sys)
