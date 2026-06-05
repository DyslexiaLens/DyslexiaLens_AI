"""
Gemini Generative AI service
=============================
Generates a sentence for handwriting practice with configurable:
  - language       : 'id' (Indonesian) or 'en' (English)
  - word_count     : number of words in the sentence
  - max_letters    : maximum letter count per word

Rules enforced both in the prompt and in post-processing:
  - No punctuation
  - Every word must be <= max_letters characters
  - Exactly word_count words
"""

from __future__ import annotations

import re

import google.generativeai as genai

from app.core.config import get_settings


def _client() -> genai.GenerativeModel:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(settings.gemini_model)


def _strip_punctuation(text: str) -> str:
    """Remove all punctuation and extra whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(cleaned.split())


def _validate_words(words: list[str], max_letters: int) -> list[str]:
    """Keep only words that are within the max_letters limit."""
    return [w for w in words if len(w) <= max_letters]


def generate_sentence(
    language: str = "id",
    word_count: int = 5,
    max_letters: int = 8,
) -> str:
    """
    Generate a practice sentence via Gemini.

    Args:
        language   : 'id' for Indonesian, 'en' for English.
        word_count : Exact number of words in the output.
        max_letters: Maximum letters per word.

    Returns:
        A clean sentence string with no punctuation.

    Raises:
        RuntimeError : If GEMINI_API_KEY is missing.
        ValueError   : If Gemini fails to produce a valid sentence after retries.
    """
    lang_name = "Bahasa Indonesia" if language == "id" else "English"

    prompt = (
        f"Generate exactly {word_count} words in {lang_name} for a handwriting practice sentence.\n"
        f"Rules:\n"
        f"- Each word must have at most {max_letters} letters.\n"
        f"- No punctuation at all (no commas, periods, hyphens, apostrophes, etc.).\n"
        f"- Output only the words separated by single spaces, nothing else.\n"
        f"- Do not include any explanation, numbering, or extra text.\n"
        f"- Make the words form a natural, meaningful sentence.\n"
        f"- All words must be common, everyday words.\n"
        f"\n"
        f"Output (exactly {word_count} words, each <= {max_letters} letters, no punctuation):"
    )

    model = _client()

    for attempt in range(3):
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip any punctuation Gemini might still include
        cleaned = _strip_punctuation(raw)
        words = cleaned.split()

        # Filter words that are too long
        valid_words = _validate_words(words, max_letters)

        if len(valid_words) >= word_count:
            # Trim to exact count
            return " ".join(valid_words[:word_count])

        # Retry with a stricter prompt if we got too few valid words
        prompt = (
            f"Write exactly {word_count} common {lang_name} words as a sentence.\n"
            f"STRICT: every single word must be {max_letters} letters or fewer.\n"
            f"STRICT: zero punctuation — only letters and spaces.\n"
            f"Output only the {word_count} words separated by spaces. Nothing else."
        )

    raise ValueError(
        f"Could not generate a valid sentence with {word_count} words "
        f"each <= {max_letters} letters after 3 attempts."
    )
