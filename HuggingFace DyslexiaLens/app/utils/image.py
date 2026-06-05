"""Shared image utilities — base64 decode + temp file helpers."""

from __future__ import annotations

import base64
import os
import tempfile


def decode_base64_image(image_base64: str, suffix: str = ".png") -> bytes:
    """Strip optional data-URI prefix and return raw bytes."""
    encoded = image_base64.split(",")[-1]
    return base64.b64decode(encoded)


def write_temp_image(image_bytes: bytes, suffix: str = ".png") -> str:
    """Write bytes to a named temporary file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        return tmp.name


def cleanup(path: str) -> None:
    """Remove a file silently."""
    try:
        os.unlink(path)
    except OSError:
        pass
