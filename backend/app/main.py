"""FastAPI application factory and server entrypoint."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from backend.app.api.routes import health, languages, recognize
from backend.app.core.config import get_settings
from backend.app.core.exceptions import SignBridgeError


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforce maximum incoming payload size."""

    def __init__(self, app: ASGIApp, max_bytes: int = 65536) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "PAYLOAD_TOO_LARGE",
                            "detail": f"Payload exceeds {self.max_bytes} bytes",
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="SignBridge API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/openapi.json",
    )

    # 1. Middlewares
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=settings.max_request_bytes,
    )

    origins = (
        settings.cors_origins
        if isinstance(settings.cors_origins, list)
        else [settings.cors_origins]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Exception Handlers
    @app.exception_handler(SignBridgeError)
    async def signbridge_exception_handler(_: Request, exc: SignBridgeError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "detail": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "VALIDATION_ERROR", "detail": str(exc)},
        )

    # 3. Include API Routers (base path /api) and WebSockets
    app.include_router(health.router, prefix="/api")
    app.include_router(languages.router, prefix="/api")
    app.include_router(recognize.router, prefix="/api")
    app.add_api_websocket_route("/ws/recognize", recognize.websocket_recognize)

    # 4. Mount frontend static assets at / after /api and /ws
    frontend_dir = Path("frontend")
    if frontend_dir.exists() and frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()
