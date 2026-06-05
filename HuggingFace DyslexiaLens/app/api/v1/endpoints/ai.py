"""AI (Gemini) endpoint — practice sentence generator."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Security, status

from app.core.config import get_settings
from app.core.security import require_api_key
from app.schemas.ai import GenerateTextRequest, GenerateTextResponse
from app.services.ai import gemini

router = APIRouter(prefix="/ai", tags=["Generative AI"])


@router.post(
    "/generate-text",
    response_model=GenerateTextResponse,
    summary="Generate a handwriting practice sentence via Gemini",
)
def generate_text(
    payload: GenerateTextRequest,
    _: str = Security(require_api_key),
) -> GenerateTextResponse:
    """
    Generate a clean practice sentence with:
    - Configurable language (Indonesian / English)
    - Configurable word count
    - Configurable max letters per word
    - No punctuation
    """
    try:
        sentence = gemini.generate_sentence(
            language=payload.language,
            word_count=payload.word_count,
            max_letters=payload.max_letters,
        )
        settings = get_settings()
        return GenerateTextResponse(
            status="success",
            sentence=sentence,
            word_count=payload.word_count,
            max_letters=payload.max_letters,
            language=payload.language,
            model_used=settings.gemini_model,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
