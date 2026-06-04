"""
DyslexiaLens — Feature Extractor
=================================
Computes 6 engineered clinical features from a raw handwriting image
and applies inline MinMax scaling (no external .pkl needed).

Feature names (canonical order):
    stroke_density, center_of_mass_x, center_of_mass_y,
    bounding_box_ratio, stroke_transitions, horizontal_symmetry
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Canonical feature list — must match training order exactly
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "stroke_density",
    "center_of_mass_x",
    "center_of_mass_y",
    "bounding_box_ratio",
    "stroke_transitions",
    "horizontal_symmetry",
]

# ---------------------------------------------------------------------------
# Hardcoded scaler parameters (fitted on Train split, 143 383 samples)
# MinMaxScaler: scaled = (x - min) / (max - min)
# ---------------------------------------------------------------------------

_SCALER_MIN = np.array(
    [0.014031, 6.803000, 6.709700, 0.178600, 0.428600, 0.561224],
    dtype=np.float32,
)
_SCALER_MAX = np.array(
    [0.440051, 21.386000, 19.867900, 8.333300, 5.571400, 1.000000],
    dtype=np.float32,
)
_SCALER_RANGE = _SCALER_MAX - _SCALER_MIN

# Processing constants
_EXTRACT_SIZE = (128, 128)
_COM_CELL_SIZE = 8  # pixel → cell-unit conversion


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale(raw: np.ndarray) -> np.ndarray:
    scaled = (raw - _SCALER_MIN) / (_SCALER_RANGE + 1e-8)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _load_grayscale(image_path: Union[str, Path]) -> np.ndarray:
    path = str(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: '{path}'")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: '{path}'")
    return img


def _binarise(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = np.mean(binary > 127)
    if ink_ratio > 0.90 or ink_ratio < 0.01:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


def _stroke_density(binary: np.ndarray) -> float:
    return float(np.sum(binary > 127)) / float(binary.size)


def _center_of_mass(binary: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(binary > 127)
    if len(coords) == 0:
        h, w = binary.shape
        return float(w / 2 / _COM_CELL_SIZE), float(h / 2 / _COM_CELL_SIZE)
    return (
        float(coords[:, 1].mean()) / _COM_CELL_SIZE,
        float(coords[:, 0].mean()) / _COM_CELL_SIZE,
    )


def _bounding_box_ratio(binary: np.ndarray) -> float:
    coords = np.argwhere(binary > 127)
    if len(coords) < 2:
        return 1.0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return float(max(x_max - x_min, 1)) / float(max(y_max - y_min, 1))


def _stroke_transitions(binary: np.ndarray) -> float:
    ink = (binary > 127).astype(np.int8)
    total = sum(int(np.sum(np.abs(np.diff(row)))) for row in ink)
    return float(total) / max(binary.shape[0], 1)


def _horizontal_symmetry(binary: np.ndarray) -> float:
    mid = binary.shape[1] // 2
    left = binary[:, :mid].astype(float)
    right = np.fliplr(binary[:, mid : mid + mid]).astype(float)
    min_w = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:, :min_w] - right[:, :min_w])
    return 1.0 - (diff.mean() / 255.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    image_path: Union[str, Path],
    scaled: bool = True,
) -> dict[str, float]:
    """
    Compute all 6 clinical features from a handwriting image.

    Args:
        image_path: Path to image (.png, .jpg, .bmp, .tiff).
        scaled:     Return MinMax-scaled values [0, 1] (default) or raw.

    Returns:
        Dict keyed by FEATURE_COLS names.
    """
    img = _load_grayscale(image_path)
    img = cv2.resize(img, _EXTRACT_SIZE, interpolation=cv2.INTER_LINEAR)
    binary = _binarise(img)

    com_x, com_y = _center_of_mass(binary)
    raw = np.array(
        [
            _stroke_density(binary),
            com_x,
            com_y,
            _bounding_box_ratio(binary),
            _stroke_transitions(binary),
            _horizontal_symmetry(binary),
        ],
        dtype=np.float32,
    )

    values = _scale(raw) if scaled else raw
    return {col: round(float(v), 6) for col, v in zip(FEATURE_COLS, values)}
