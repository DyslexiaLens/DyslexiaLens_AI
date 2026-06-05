from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(v1_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["Health"])
    def health() -> dict:
        return {"status": "ok", "service": settings.app_title, "version": settings.app_version}

    return app


app = create_app()
