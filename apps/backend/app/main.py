"""FastAPI application entry point."""

import asyncio
import faulthandler
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Fix for Windows: Use ProactorEventLoop for subprocess support (Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.database import db
from app.llm import AIServiceError
from app.pdf import close_pdf_renderer, init_pdf_renderer
from app.routers import config_router, enrichment_router, health_router, jobs_router, resumes_router


def _configure_application_logging() -> None:
    """Set application log level from configuration."""
    numeric_level = getattr(logging, settings.log_level, logging.INFO)
    logging.getLogger("app").setLevel(numeric_level)


def _enable_fault_diagnostics() -> None:
    """Enable fatal traceback dumping for crashes without Python exceptions."""
    if faulthandler.is_enabled():
        return
    try:
        faulthandler.enable(all_threads=True)
        logger.info("Enabled faulthandler for backend diagnostics.")
    except Exception as e:
        logger.warning("Failed to enable faulthandler: %s", e)


_configure_application_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    _enable_fault_diagnostics()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # PDF renderer uses lazy initialization - will initialize on first use
    # await init_pdf_renderer()
    yield
    # Shutdown - wrap each cleanup in try-except to ensure all resources are released
    try:
        await close_pdf_renderer()
    except Exception as e:
        logger.error(f"Error closing PDF renderer: {e}")

    try:
        db.close()
    except Exception as e:
        logger.error(f"Error closing database: {e}")


app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered resume tailoring for job descriptions",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware - origins configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.effective_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AIServiceError)
async def handle_ai_service_error(_: object, exc: AIServiceError) -> JSONResponse:
    """Return a stable error contract for provider/model failures."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": exc.error_code,
        },
    )

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(resumes_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(enrichment_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Resume Matcher API",
        "version": __version__,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
