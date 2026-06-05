"""
EMNIST OCR — Inference Engine
================================
Loads the EMNIST character recognition model and processes a full grid sheet.

Classes:
    OcrInference      — full EMNIST 62-class OCR (digits + upper + lower).
    OcrGridInference  — uppercase A-Z grid-form OCR; also returns total row count.
"""

from __future__ import annotations

import os
import warnings

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# EMNIST character mapping — indices match the training label order
EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_CELL_SIZE = 28
_BLANK_RATIO_THRESHOLD = 1.5  # % pixels below which a cell is considered blank
_GRID_CELL_MIN = 20
_GRID_CELL_MAX = 200
_ROW_Y_TOLERANCE = 20


# ---------------------------------------------------------------------------
# Custom Keras objects
# ---------------------------------------------------------------------------


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(1, channels // self.reduction_ratio), activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, x):
        scale = self.gap(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = self.reshape(scale)
        return x * scale

    def get_config(self):
        return {**super().get_config(), "reduction_ratio": self.reduction_ratio}


class LabelSmoothingCrossEntropy(keras.losses.Loss):
    def __init__(self, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[-1] == 1):
            y_true = tf.one_hot(
                tf.cast(tf.reshape(y_true, [-1]), tf.int32),
                tf.cast(num_classes, tf.int32),
            )
        smooth_labels = y_true * (1.0 - self.smoothing) + self.smoothing / num_classes
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        return tf.reduce_mean(-tf.reduce_sum(smooth_labels * log_probs, axis=-1))

    def get_config(self):
        return {**super().get_config(), "smoothing": self.smoothing}


def _mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


_CUSTOM_OBJECTS = {
    "ChannelAttention": ChannelAttention,
    "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy,
    "mish": _mish,
}


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------


class OcrInference:
    """
    EMNIST grid-sheet OCR engine.

    Args:
        model_path: Path to the saved best_model.keras file.
    """

    def __init__(self, model_path: str = "models/best_model.keras"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"OCR model not found: '{model_path}'. "
                "Place best_model.keras inside the models/ directory."
            )
        self.model = keras.models.load_model(model_path, custom_objects=_CUSTOM_OBJECTS)

    # ------------------------------------------------------------------
    # Grid segmentation
    # ------------------------------------------------------------------

    def _segment_cells(self, thresh: np.ndarray) -> list[list[tuple]]:
        """
        Detect grid cells via morphological operations.

        Returns:
            List of rows, each row is a sorted list of (x, y, w, h) tuples.
        """
        scale = _CELL_SIZE
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        grid_mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)
        grid_mask = cv2.threshold(grid_mask, 10, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            cv2.boundingRect(c)
            for c in contours
            if _GRID_CELL_MIN < cv2.boundingRect(c)[2] < _GRID_CELL_MAX
            and _GRID_CELL_MIN < cv2.boundingRect(c)[3] < _GRID_CELL_MAX
        ]
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        rows: list[list[tuple]] = []
        current_row: list[tuple] = []
        prev_y = boxes_sorted[0][1]

        for box in boxes_sorted:
            if abs(box[1] - prev_y) > _ROW_Y_TOLERANCE:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                prev_y = box[1]
            else:
                current_row.append(box)

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        return rows

    # ------------------------------------------------------------------
    # Cell classification
    # ------------------------------------------------------------------

    def _classify_cell(self, cell_thresh: np.ndarray) -> str | None:
        """Return predicted character or None if the cell is blank."""
        if cell_thresh.size == 0:
            return None

        text_ratio = cv2.countNonZero(cell_thresh) / cell_thresh.size * 100
        if text_ratio <= _BLANK_RATIO_THRESHOLD:
            return None

        cell = cv2.resize(cell_thresh, (_CELL_SIZE, _CELL_SIZE), interpolation=cv2.INTER_AREA)
        inp = cell.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=(0, -1))  # (1, 28, 28, 1)

        logits = self.model.predict(inp, verbose=0)
        idx = int(np.argmax(logits, axis=-1)[0])
        return EMNIST_MAPPING[idx] if idx < len(EMNIST_MAPPING) else "?"

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> str:
        """
        Run OCR on a raw grid sheet image.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG).

        Returns:
            Extracted text string with rows separated by newlines.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        rows = self._segment_cells(thresh)
        if not rows:
            return ""

        extracted: list[str] = []
        padding = 5

        for row in rows:
            row_text = ""
            has_content = False
            space_allocated = False

            for x, y, w, h in row:
                cell = thresh[y + padding : y + h - padding, x + padding : x + w - padding]
                char = self._classify_cell(cell)

                if char is not None:
                    row_text += char
                    has_content = True
                    space_allocated = False
                elif has_content and not space_allocated:
                    row_text += " "
                    space_allocated = True

            stripped = row_text.rstrip()
            if stripped:
                extracted.append(stripped)

        return "\n".join(extracted)


# ---------------------------------------------------------------------------
# Grid-form OCR engine  (A-Z only, also returns total row count)
# ---------------------------------------------------------------------------

# app2.py maps prediction index directly to A-Z (26 classes only)
_GRID_CHAR_MAPPING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class OcrGridInference:
    """
    Grid-form handwriting OCR engine.

    Detects individual letter cells in a ruled/grid paper image and returns
    the extracted uppercase text together with the total number of detected rows.

    The model is expected to produce 26-class logits (A-Z).

    Args:
        model_path: Path to the saved ``ocr_model_v4.keras`` file.
    """

    def __init__(self, model_path: str = "models/ocr_model_v4.keras"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"OCR grid model not found: '{model_path}'. "
                "Place ocr_model_v4.keras inside the models/ directory."
            )
        self.model = keras.models.load_model(model_path, custom_objects=_CUSTOM_OBJECTS)

    # ------------------------------------------------------------------
    # Grid segmentation (shared logic, identical to OcrInference)
    # ------------------------------------------------------------------

    def _segment_cells(self, thresh: np.ndarray) -> list[list[tuple]]:
        """Detect grid cells via morphological line detection.

        Returns:
            List of rows; each row is a list of ``(x, y, w, h)`` tuples sorted
            left-to-right.
        """
        scale = _CELL_SIZE
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        grid_mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)
        grid_mask = cv2.threshold(grid_mask, 10, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            cv2.boundingRect(c)
            for c in contours
            if _GRID_CELL_MIN < cv2.boundingRect(c)[2] < _GRID_CELL_MAX
            and _GRID_CELL_MIN < cv2.boundingRect(c)[3] < _GRID_CELL_MAX
        ]
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        rows: list[list[tuple]] = []
        current_row: list[tuple] = []
        prev_y = boxes_sorted[0][1]

        for box in boxes_sorted:
            if abs(box[1] - prev_y) > _ROW_Y_TOLERANCE:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                prev_y = box[1]
            else:
                current_row.append(box)

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        return rows

    # ------------------------------------------------------------------
    # Cell classification (A-Z mapping)
    # ------------------------------------------------------------------

    def _classify_cell(self, cell_thresh: np.ndarray) -> str | None:
        """Return predicted uppercase letter or ``None`` if the cell is blank."""
        if cell_thresh.size == 0:
            return None

        density = cv2.countNonZero(cell_thresh) / cell_thresh.size * 100
        if density <= _BLANK_RATIO_THRESHOLD:
            return None

        cell = cv2.resize(cell_thresh, (_CELL_SIZE, _CELL_SIZE), interpolation=cv2.INTER_AREA)
        inp = cell.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=(0, -1))  # (1, 28, 28, 1)

        logits = self.model.predict(inp, verbose=0)
        idx = int(np.argmax(logits, axis=-1)[0])
        return _GRID_CHAR_MAPPING[idx] if idx < len(_GRID_CHAR_MAPPING) else "?"

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> tuple[int, str]:
        """
        Run grid-form OCR on a raw image.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG).

        Returns:
            A ``(total_rows_detected, result_text)`` tuple where ``result_text``
            contains rows separated by newlines and blank cells represented by
            a single space character.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format or corrupted file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        rows = self._segment_cells(thresh)
        if not rows:
            return 0, ""

        extracted: list[str] = []
        padding = 5

        for row in rows:
            row_text = ""
            has_content = False
            space_allocated = False

            for x, y, w, h in row:
                cell = thresh[y + padding : y + h - padding, x + padding : x + w - padding]
                char = self._classify_cell(cell)

                if char is not None:
                    row_text += char
                    has_content = True
                    space_allocated = False
                elif has_content and not space_allocated:
                    row_text += " "
                    space_allocated = True

            stripped = row_text.rstrip()
            if stripped:
                extracted.append(stripped)

        return len(rows), "\n".join(extracted)
