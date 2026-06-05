"""EMNIST OCR predict endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Security, status

from app.core.config import get_settings
from app.core.security import require_api_key
from app.schemas.ocr import (
    OcrGridPredictRequest,
    OcrGridPredictResponse,
)
from app.services.ocr.inference import OcrGridInference, OcrInference
from app.utils.image import decode_base64_image

router = APIRouter(prefix="/ocr", tags=["EMNIST OCR"])


def _get_engine() -> OcrInference:
    if not hasattr(_get_engine, "_instance"):
        settings = get_settings()
        _get_engine._instance = OcrInference(model_path=settings.ocr_model_path)
    return _get_engine._instance


def _get_grid_engine() -> OcrGridInference:
    if not hasattr(_get_grid_engine, "_instance"):
        settings = get_settings()
        _get_grid_engine._instance = OcrGridInference(model_path=settings.ocr_model_path)
    return _get_grid_engine._instance


@router.post(
    "/predict",
    response_model=OcrGridPredictResponse,
    summary="Extract uppercase handwritten letters from a ruled/grid form image",
    description=(
        "Detects individual letter cells in a grid-ruled paper image using "
        "morphological line detection, classifies each cell as an uppercase "
        "letter (A-Z), and returns the extracted text together with the total "
        "number of detected rows."
    ),
)
def predict_grid(
    payload: OcrGridPredictRequest,
    _: str = Security(require_api_key),
) -> OcrGridPredictResponse:
    try:
        image_bytes = decode_base64_image(payload.image_base64)
        total_rows, result_text = _get_grid_engine().predict(image_bytes)
        return OcrGridPredictResponse(
            status="success",
            total_rows_detected=total_rows,
            result_text=result_text,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
