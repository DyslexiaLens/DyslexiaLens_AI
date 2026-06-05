from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class DyslexiaPredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (with or without data URI prefix)")
    features: Union[dict[str, float], list[float], None] = Field(
        default=None,
        description=(
            "Optional pre-computed features. "
            "Pass a dict keyed by feature name, a list of 6 floats in canonical order, "
            "or omit to auto-extract from the image."
        ),
    )


class DyslexiaPredictResponse(BaseModel):
    status: str
    result_text: str
    label: str
    dyslexia_probability: float
    predicted_class: int
    severity_score: float
    severity_level: str
    threshold_used: float
    features: dict[str, float]
