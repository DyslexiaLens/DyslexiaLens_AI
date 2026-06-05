from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateTextRequest(BaseModel):
    language: str = Field(
        default="id",
        description="Language of the generated sentence: 'id' (Indonesian) or 'en' (English)",
    )
    word_count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of words in the generated sentence",
    )
    max_letters: int = Field(
        default=8,
        ge=1,
        le=8,
        description="Maximum number of letters allowed per word",
    )


class GenerateTextResponse(BaseModel):
    status: str
    sentence: str
    word_count: int
    max_letters: int
    language: str
    model_used: str
