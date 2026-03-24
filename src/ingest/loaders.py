from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RawPage:
    """A single logical page / block of text before chunking."""
    text: str
    page_number: int
    section: str = ""
    doc_id: str = ""
    doc_title: str = ""


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def load_pdf(
    path: str | Path,
    doc_id: str,
    doc_title: str = "",
) -> list[RawPage]:
    """
    Load a PDF file and return one RawPage per page with extracted text.
    Pages with no text content are skipped.
    """
    import fitz  # pymupdf

    path = Path(path)
    pages: list[RawPage] = []
    current_section = ""

    doc = fitz.open(str(path))
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            if not text.strip():
                continue

            # Heuristic: treat short all-caps lines near the top as section headers
            lines = text.split("\n")
            for line in lines[:5]:
                stripped = line.strip()
                if 4 < len(stripped) < 80 and stripped.replace(" ", "").isupper():
                    current_section = stripped.title()
                    break

            pages.append(
                RawPage(
                    text=text,
                    page_number=page_idx + 1,
                    section=current_section,
                    doc_id=doc_id,
                    doc_title=doc_title or path.stem,
                )
            )
    finally:
        doc.close()

    logger.info("Loaded %d pages from '%s' (doc_id=%s)", len(pages), path.name, doc_id)
    return pages


# ---------------------------------------------------------------------------
# Plain text / Markdown loader
# ---------------------------------------------------------------------------

def load_text(
    path: str | Path,
    doc_id: str,
    doc_title: str = "",
) -> list[RawPage]:
    """Load a .txt or .md file as a single RawPage."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    logger.info("Loaded text file '%s' (doc_id=%s, chars=%d)", path.name, doc_id, len(text))
    return [
        RawPage(
            text=text,
            page_number=1,
            section="",
            doc_id=doc_id,
            doc_title=doc_title or path.stem,
        )
    ]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_document(
    path: str | Path,
    doc_id: str,
    doc_title: str = "",
) -> list[RawPage]:
    """Route to the correct loader by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    match suffix:
        case ".pdf":
            return load_pdf(path, doc_id=doc_id, doc_title=doc_title)
        case ".txt" | ".md":
            return load_text(path, doc_id=doc_id, doc_title=doc_title)
        case _:
            raise ValueError(
                f"Unsupported document type: '{suffix}'. "
                "Supported: .pdf, .txt, .md"
            )
