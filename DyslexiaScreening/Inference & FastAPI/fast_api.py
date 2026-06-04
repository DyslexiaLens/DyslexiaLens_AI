"""DyslexiaLens predict endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Security, status

from app.core.config import get_settings
from app.core.security import require_api_key
from app.schemas.dyslexia import DyslexiaPredictRequest, DyslexiaPredictResponse
from app.services.dyslexia.feature_extractor import FEATURE_COLS, extract_features
from app.services.dyslexia.inference import DyslexiaLensInference
from app.utils.image import cleanup, decode_base64_image, write_temp_image

router = APIRouter(prefix="/dyslexia", tags=["DyslexiaLens"])


def _get_engine() -> DyslexiaLensInference:
    """Lazy singleton — loaded once per worker process."""
    if not hasattr(_get_engine, "_instance"):
        settings = get_settings()
        _get_engine._instance = DyslexiaLensInference(
            model_path=settings.dyslexia_model_path,
            threshold=settings.decision_threshold,
        )
    return _get_engine._instance


def _resolve_features(
    payload_features: dict | list | None,
    image_path: str,
) -> dict[str, float]:
    """
    Return a dict of scaled features, auto-extracting from the image when
    no features are provided.
    """
    if payload_features is None:
        return extract_features(image_path)

    if isinstance(payload_features, list):
        if len(payload_features) != len(FEATURE_COLS):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"`features` list must have exactly {len(FEATURE_COLS)} values.",
            )
        return dict(zip(FEATURE_COLS, [float(v) for v in payload_features]))

    return {col: float(payload_features.get(col, 0.0)) for col in FEATURE_COLS}


@router.post(
    "/predict",
    response_model=DyslexiaPredictResponse,
    summary="Classify a handwriting sheet for dyslexia indicators",
)
def predict(
    payload: DyslexiaPredictRequest,
    _: str = Security(require_api_key),
) -> DyslexiaPredictResponse:
    image_path: str | None = None
    try:
        image_bytes = decode_base64_image(payload.image_base64)
        image_path = write_temp_image(image_bytes)
        features = _resolve_features(payload.features, image_path)
        result = _get_engine().predict(image_path, features)

        return DyslexiaPredictResponse(
            status="success",
            result_text=result["classification"],
            label=result["classification"],
            dyslexia_probability=result["dyslexia_probability"],
            predicted_class=result["predicted_class"],
            severity_score=result["severity_score"],
            severity_level=result["severity_level"],
            threshold_used=result["threshold_used"],
            features=features,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        if image_path:
            cleanup(image_path)