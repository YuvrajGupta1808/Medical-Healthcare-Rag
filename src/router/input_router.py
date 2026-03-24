"""
Input Router Node — Step 2.

Normalises every supported input modality (text / image / audio / PDF / video)
into a single, consistent RAGState that every downstream node can rely on.

Usage pattern
-------------
API layer (pre-graph)::

    router = InputRouter()
    state = await router.route(modality=Modality.IMAGE, question="...", file=upload)
    result = await rag_pipeline.ainvoke(state)

LangGraph node (in-graph)::

    graph.add_node("input_router", input_router_node)

Stub markers
------------
- _stub_describe_image   — replace with Vertex AI Gemini multimodal vision (Step 7)
- _stub_transcribe_audio — replace with Whisper / Vertex AI speech-to-text (Step 7)
- _stub_describe_video   — replace with video transcription service (Step 7)

Invariants after routing
------------------------
Every modality satisfies ALL of the following::

    state["request_id"]        — non-empty UUID string
    state["query_text"]        — non-empty string used for retrieval
    state["original_question"] — raw user question before enrichment
    state["modality"]          — Modality enum value
    state["modality_metadata"] — dict (may be empty for TEXT)
    state["storage_ref"]       — str key for binary payloads; None for TEXT
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import UploadFile

from src.pipeline.state import RAGState
from src.schemas.query import Modality

logger = structlog.get_logger(__name__)


class InputRouter:
    """
    Normalises raw multipart inputs into a partial RAGState.

    Used by the API layer before handing state to the LangGraph pipeline.
    Reads file bytes, generates a storage_ref key, and enriches query_text
    with modality-specific context via stub methods.
    """

    async def route(
        self,
        modality: Modality,
        question: str,
        file: UploadFile | None = None,
        request_id: str | None = None,
    ) -> RAGState:
        """
        Dispatch to the appropriate modality handler.

        Args:
            modality:   Input modality type.
            question:   User's text question (required for all modalities).
            file:       Uploaded file; required for all non-TEXT modalities.
            request_id: Caller-supplied ID; a new UUID4 is generated if omitted.

        Returns:
            Partial RAGState with all Step-2 fields populated.

        Raises:
            ValueError: If ``file`` is None for a binary modality.
        """
        request_id = request_id or str(uuid.uuid4())
        log = logger.bind(request_id=request_id, modality=modality.value)
        log.info("input_router.dispatch", question_preview=question[:80])

        if modality == Modality.TEXT:
            state = await self._route_text(question=question, request_id=request_id)
        else:
            if file is None:
                raise ValueError(
                    f"A file upload is required for modality={modality.value!r}."
                )
            handlers = {
                Modality.IMAGE: self._route_image,
                Modality.AUDIO: self._route_audio,
                Modality.PDF: self._route_pdf,
                Modality.VIDEO: self._route_video,
            }
            state = await handlers[modality](
                file=file, question=question, request_id=request_id
            )

        log.info(
            "input_router.complete",
            query_text_preview=state["query_text"][:100],
            storage_ref=state.get("storage_ref"),
        )
        return state

    # ------------------------------------------------------------------
    # Per-modality handlers
    # ------------------------------------------------------------------

    async def _route_text(self, question: str, request_id: str) -> RAGState:
        return {  # type: ignore[return-value]
            "request_id": request_id,
            "query_text": question,
            "original_question": question,
            "modality": Modality.TEXT,
            "modality_metadata": {},
            "storage_ref": None,
        }

    async def _route_image(
        self, file: UploadFile, question: str, request_id: str
    ) -> RAGState:
        content = await file.read()
        storage_ref = f"images/{request_id}/{file.filename}"
        description = self._stub_describe_image(content, file.content_type)
        # Enrich retrieval query with the (stubbed) image description.
        query_text = f"{question}. [Image context: {description}]"
        return {  # type: ignore[return-value]
            "request_id": request_id,
            "query_text": query_text,
            "original_question": question,
            "modality": Modality.IMAGE,
            "modality_metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "description": description,
            },
            "storage_ref": storage_ref,
        }

    async def _route_audio(
        self, file: UploadFile, question: str, request_id: str
    ) -> RAGState:
        content = await file.read()
        storage_ref = f"audio/{request_id}/{file.filename}"
        transcript = self._stub_transcribe_audio(content, file.content_type)
        # If no text question is supplied, the audio transcript IS the query.
        if question.strip():
            query_text = f"{question}. [Transcript: {transcript}]"
            original_question = question
        else:
            query_text = transcript
            original_question = transcript
        return {  # type: ignore[return-value]
            "request_id": request_id,
            "query_text": query_text,
            "original_question": original_question,
            "modality": Modality.AUDIO,
            "modality_metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "transcript": transcript,
            },
            "storage_ref": storage_ref,
        }

    async def _route_pdf(
        self, file: UploadFile, question: str, request_id: str
    ) -> RAGState:
        content = await file.read()
        storage_ref = f"pdfs/{request_id}/{file.filename}"
        # On the query path the text question drives retrieval; the PDF itself
        # is stored by reference for downstream context extraction (Step 3+).
        return {  # type: ignore[return-value]
            "request_id": request_id,
            "query_text": question,
            "original_question": question,
            "modality": Modality.PDF,
            "modality_metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
            },
            "storage_ref": storage_ref,
        }

    async def _route_video(
        self, file: UploadFile, question: str, request_id: str
    ) -> RAGState:
        content = await file.read()
        storage_ref = f"videos/{request_id}/{file.filename}"
        description = self._stub_describe_video(content, file.content_type)
        query_text = f"{question}. [Video context: {description}]"
        return {  # type: ignore[return-value]
            "request_id": request_id,
            "query_text": query_text,
            "original_question": question,
            "modality": Modality.VIDEO,
            "modality_metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "description": description,
            },
            "storage_ref": storage_ref,
        }

    # ------------------------------------------------------------------
    # Stubs — replace in Step 7
    # ------------------------------------------------------------------

    def _stub_describe_image(
        self, content: bytes, content_type: str | None = None
    ) -> str:
        """STUB — replace with Vertex AI Gemini multimodal vision in Step 7."""
        return "[STUB: image description pending Vertex AI integration]"

    def _stub_transcribe_audio(
        self, content: bytes, content_type: str | None = None
    ) -> str:
        """STUB — replace with Whisper / Vertex AI speech-to-text in Step 7."""
        return "[STUB: audio transcription pending Whisper integration]"

    def _stub_describe_video(
        self, content: bytes, content_type: str | None = None
    ) -> str:
        """STUB — replace with video transcription service in Step 7."""
        return "[STUB: video description pending transcription integration]"


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def input_router_node(state: RAGState) -> dict:
    """
    LangGraph node — validates that required Step-2 fields are present.

    The API layer pre-processes uploads via ``InputRouter.route()`` before
    invoking the graph, so this node acts as a guard that confirms the state
    contract before retrieval begins.

    Pre-conditions (set by API layer via InputRouter.route()):
        - ``state["request_id"]`` — non-empty
        - ``state["query_text"]`` — non-empty
        - ``state["modality"]``   — Modality enum value

    On failure:
        Sets ``state["error"]``; downstream nodes should short-circuit.
    """
    log = structlog.get_logger(__name__).bind(
        request_id=state.get("request_id", "unknown"),
        modality=str(state.get("modality")),
    )

    missing = [
        field
        for field in ("request_id", "query_text", "modality")
        if not state.get(field)
    ]
    if missing:
        error_msg = f"input_router_node: missing required fields: {missing}"
        log.error("input_router_node.validation_failed", missing=missing)
        return {"error": error_msg}

    log.info(
        "input_router_node.validated",
        query_text_preview=state["query_text"][:100],
    )
    return {}  # no-op: state is already normalised by InputRouter.route()


# ---------------------------------------------------------------------------
# Convenience builder — used by the /query route for text-only queries
# ---------------------------------------------------------------------------

def build_initial_state(
    query_text: str,
    modality_type: str = "text",
    doc_id: str | None = None,
    image_bytes: bytes | None = None,
    audio_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RAGState:
    """Build the initial RAGState for a text query (Step 1 entry point)."""
    return RAGState(
        query_text=query_text,
        modality_type=modality_type,
        image_bytes=image_bytes,
        audio_ref=audio_ref,
        doc_id=doc_id,
        retrieved_chunks=[],
        answer="",
        citations=[],
        prefer_retrieved_image=False,
        include_pdf_export=False,
        image_url=None,
        metadata=metadata or {},
        error=None,
    )
