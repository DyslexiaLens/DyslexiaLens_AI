"""
DyslexiaLens — Inference Engine
================================
Loads the trained .keras model and runs single-sample inference.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import Huber

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app.services.dyslexia.feature_extractor import FEATURE_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = (128, 128)
IMG_CHANNELS = 1
DECISION_THRESHOLD = 0.52
_SEV_MILD_MAX = 0.33
_SEV_MODERATE_MAX = 0.66


# ---------------------------------------------------------------------------
# Custom Keras objects (required to reload the saved model)
# ---------------------------------------------------------------------------


class AdaptiveContrastNorm(layers.Layer):
    """Per-sample adaptive contrast normalisation."""

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(shape=(1, 1, 1, channels), name="gamma", initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=(1, 1, 1, channels), name="beta", initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x, training=None):
        mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        sigma = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + self.epsilon
        return self.gamma * ((x - mu) / sigma) + self.beta

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


class MaskedHuberLoss(keras.losses.Loss):
    """Huber loss masked to dyslexia samples only (required for model reload)."""

    def __init__(self, delta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self._huber_fn = Huber(delta=delta, reduction="none")

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, 1]), tf.float32)
        mask = tf.cast(y_true > 0.0, tf.float32)
        per_sample = self._huber_fn(y_true, y_pred)
        masked = per_sample * tf.squeeze(mask, axis=-1)
        n_dyslexia = tf.reduce_sum(mask) + 1e-8
        return tf.reduce_sum(masked) / n_dyslexia

    def get_config(self):
        return {**super().get_config(), "delta": self.delta}


_CUSTOM_OBJECTS = {
    "AdaptiveContrastNorm": AdaptiveContrastNorm,
    "MaskedHuberLoss": MaskedHuberLoss,
}


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class DyslexiaLensInference:
    """
    Load a saved DyslexiaLens model and run single-sample inference.

    Args:
        model_path:  Path to the saved .keras model.
        threshold:   Decision boundary (default: 0.52).
    """

    def __init__(
        self,
        model_path: str = "models/dyslexialens_model.keras",
        threshold: float = DECISION_THRESHOLD,
    ):
        self.threshold = threshold
        self.model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, path: str) -> keras.Model:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: '{path}'. "
                "Download dyslexialens_model.keras and place it in the models/ directory."
            )
        return keras.models.load_model(path, custom_objects=_CUSTOM_OBJECTS)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_path: str) -> tf.Tensor:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: '{image_path}'")
        raw = tf.io.read_file(image_path)
        try:
            img = tf.image.decode_png(raw, channels=IMG_CHANNELS)
        except Exception:
            img = tf.image.decode_jpeg(raw, channels=IMG_CHANNELS)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return tf.expand_dims(img, 0)

    def _preprocess_features(self, features: dict[str, float]) -> tf.Tensor:
        missing = [c for c in FEATURE_COLS if c not in features]
        if missing:
            raise ValueError(f"Missing feature(s): {missing}. Required: {FEATURE_COLS}")
        vec = np.array([[float(features[c]) for c in FEATURE_COLS]], dtype=np.float32)
        vec_clipped = np.clip(vec, 0.0, 1.0).astype(np.float32)
        return tf.constant(vec_clipped, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_level(predicted_class: int, severity_score: float) -> str:
        if predicted_class == 0:
            return "None"
        if severity_score < _SEV_MILD_MAX:
            return "Mild"
        if severity_score < _SEV_MODERATE_MAX:
            return "Moderate"
        return "Severe"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image_path: str, features: dict[str, float]) -> dict:
        """
        Run inference on a single handwriting image.

        Args:
            image_path: Path to the handwriting image.
            features:   Dict of 6 pre-scaled clinical features.

        Returns:
            Dict with classification, probabilities, severity info.
        """
        img = self._preprocess_image(image_path)
        feat = self._preprocess_features(features)

        p_cls, p_sev = self.model(
            {"image_input": img, "feature_input": feat},
            training=False,
        )

        prob_cls = float(p_cls.numpy().flatten()[0])
        prob_sev = float(p_sev.numpy().flatten()[0])
        predicted_class = int(prob_cls >= self.threshold)
        severity_score = round(prob_sev, 4) if predicted_class == 1 else 0.0

        SEVERITY_THRESHOLD = 1 
        
        if (prob_cls >= self.threshold) and (severity_score >= SEVERITY_THRESHOLD):
            classification = "Dyslexia"
            predicted_class = 1
        else:
            classification = "Control"
            predicted_class = 0

        return {
            "classification": classification,
            "dyslexia_probability": round(prob_cls, 4),
            "predicted_class": predicted_class,
            "severity_score": severity_score,
            "severity_level": self._severity_level(1, prob_sev),
            "threshold_used": self.threshold,
        }
