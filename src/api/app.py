from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import get_settings

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect to Weaviate and ensure the MedicalChunk schema exists.
    Shutdown: close the Weaviate connection.

    Skipped entirely when TESTING=true so test suites don't require live services.
    """
    settings = get_settings()

    if not settings.testing:
        from src.services import weaviate_client
        try:
            weaviate_client.connect()
            weaviate_client.ensure_schema()
            logger.info("All services ready.")
        except Exception as exc:
            logger.error("Startup failed: %s", exc)
            raise

    yield  # application runs here

    if not settings.testing:
        from src.services import weaviate_client
        weaviate_client.close()


app = FastAPI(
    title="Medical Healthcare RAG",
    description=(
        "Multimodal medical document assistant. "
        "Upload documents, ask questions, get grounded answers with source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------
from src.api.routes.health import router as health_router  # noqa: E402, I001
from src.api.routes.ingest import router as ingest_router  # noqa: E402
from src.api.routes.query import router as query_router  # noqa: E402
from src.api.routes.system import router as system_router
from src.api.routes.patients import router as patients_router

app.include_router(health_router, tags=["health"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(system_router, prefix="/system", tags=["system"])
app.include_router(patients_router, prefix="/patients", tags=["patients"])
