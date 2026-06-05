from fastapi import APIRouter

from app.api.v1.endpoints import ai, dyslexia, ocr

router = APIRouter()

router.include_router(dyslexia.router)
router.include_router(ocr.router)
router.include_router(ai.router)
