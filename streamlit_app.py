"""
Medical Healthcare RAG — Streamlit Frontend
============================================
A full-featured UI for the Medical Healthcare RAG FastAPI backend.
Pages: Home · Full flow test · Ingest · Ask · System Health
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Medical Healthcare RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
        /* Hero banner */
        .hero {
            background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
            padding: 2.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0 0 0.5rem 0; font-size: 2.2rem; }
        .hero p  { margin: 0; opacity: 0.9; font-size: 1.05rem; }

        /* Feature / info cards */
        .card {
            background: #f5f7ff;
            border: 1px solid #dde3f7;
            border-left: 5px solid #1976d2;
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            height: 100%;
        }
        .card h4 { margin: 0 0 0.5rem 0; color: #0d47a1; }
        .card p  { margin: 0; font-size: 0.92rem; color: #444; }

        /* Answer box */
        .answer-box {
            background: #e8f5e9;
            border: 1px solid #66bb6a;
            border-radius: 8px;
            padding: 1.4rem;
            margin: 0.8rem 0;
        }
        .answer-box h4 { color: #2e7d32; margin: 0 0 0.6rem 0; }

        /* Abstain warning */
        .abstain-box {
            background: #fff8e1;
            border: 1px solid #ffb300;
            border-radius: 8px;
            padding: 1.2rem;
        }

        /* Citation card */
        .citation-card {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #fb8c00;
            border-radius: 6px;
            padding: 0.9rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.9rem;
        }

        /* Pipeline step */
        .step {
            background: #e3f2fd;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            height: 100%;
        }
        .step h5 { margin: 0 0 0.4rem 0; color: #0d47a1; }
        .step p  { margin: 0; font-size: 0.88rem; color: #555; }

        /* Sidebar nav label */
        div[data-testid="stRadio"] label { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Constants
# ============================================================
DEFAULT_API_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")

MODALITY_META: dict[str, dict] = {
    "text":  {"icon": "💬", "label": "Text",  "desc": "Plain-text question — no file needed",      "types": None},
    "image": {"icon": "🖼️", "label": "Image", "desc": "Upload an image and ask about it",          "types": ["jpg", "jpeg", "png", "gif", "webp"]},
    "audio": {"icon": "🎙️", "label": "Audio", "desc": "Upload audio — transcript drives retrieval","types": ["mp3", "wav", "ogg", "m4a", "flac"]},
    "pdf":   {"icon": "📄", "label": "PDF",   "desc": "Upload a PDF as query context",             "types": ["pdf"]},
    "video": {"icon": "🎬", "label": "Video", "desc": "Upload video — description drives retrieval","types": ["mp4", "avi", "mov", "webm", "mkv"]},
}

# Full-flow playbook: use after ingesting `sample_medical.txt` (or similar diabetes overview).
FLOW_E2E_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "A1",
        "goal": "Broad grounded summary",
        "question": "What is type 2 diabetes and what are common risk factors?",
        "good_signs": "Mentions insulin resistance, obesity/inactivity/family history; cites retrieved chunks.",
    },
    {
        "id": "A2",
        "goal": "Numeric / diagnostic criteria",
        "question": "What fasting glucose and HbA1c thresholds are used as diagnostic criteria in the document?",
        "good_signs": "126 mg/dL fasting (two occasions), HbA1c ≥ 6.5%; quotes should match the corpus.",
    },
    {
        "id": "A3",
        "goal": "Treatment / medication",
        "question": "What first-line medication is mentioned for type 2 diabetes management?",
        "good_signs": "Metformin + lifestyle; evidence tied to the doc.",
    },
    {
        "id": "A4",
        "goal": "Complications",
        "question": "List complications of poorly controlled diabetes mentioned in the text.",
        "good_signs": "Nephropathy, retinopathy, neuropathy, cardiovascular risk; grounded list.",
    },
    {
        "id": "A5",
        "goal": "Acute complication",
        "question": "What is hypoglycaemia, what symptoms are listed, and how is it treated?",
        "good_signs": "Glucose <70 mg/dL, tremor/sweating/confusion, carbohydrate intake.",
    },
    {
        "id": "B1",
        "goal": "Lexical / BM25-friendly",
        "question": "polyuria polydipsia diabetes symptoms",
        "good_signs": "Hybrid search should surface the symptoms paragraph.",
    },
    {
        "id": "C1",
        "goal": "Adversarial (out-of-corpus)",
        "question": "What is the stock price of Apple Inc. next Tuesday?",
        "good_signs": "Abstain or explicit ‘not in documents’; ideally no fake citations.",
    },
]


def _sample_medical_fixture_text() -> str:
    p = Path(__file__).resolve().parent / "tests" / "fixtures" / "sample_medical.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return (
        "Diabetes Mellitus: Clinical Overview\n\n"
        "Type 2 diabetes is the most common form. Management typically involves "
        "metformin.\n\n"
        "Diagnostic criteria include fasting plasma glucose of 126 mg/dL or higher, "
        "or HbA1c of 6.5 percent or higher.\n"
    )


ABSTAIN_SNIPPET = "I cannot find sufficient evidence"

# ============================================================
# Session-state defaults
# ============================================================
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

if "ingest_history" not in st.session_state:
    st.session_state["ingest_history"] = []

# ============================================================
# Helpers
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_url: str) -> dict | None:
    """Call /health and return JSON, or None if unreachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{api_url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def _status_badge(status: str) -> str:
    icons = {"ok": "✅", "degraded": "⚠️", "not_connected": "❌"}
    return icons.get(status, "❓")


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🏥 Medical RAG")
    st.caption("Multimodal medical document assistant")
    st.divider()

    api_url: str = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        key="api_url",
        help="FastAPI backend URL (e.g. http://localhost:8000)",
    )

    # Live connection indicators
    health_data = fetch_health(api_url)
    if health_data:
        st.success("✅ API reachable")
        w = health_data.get("weaviate", "not_connected")
        if w == "ok":
            st.success("✅ Weaviate connected")
        elif w == "degraded":
            st.warning("⚠️ Weaviate degraded")
        else:
            st.error("❌ Weaviate disconnected")
        st.caption(f"API v{health_data.get('version', '—')}")
    else:
        st.error("❌ API offline")
        st.caption("Start backend: `uvicorn src.api.app:app --reload`")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧪 Full flow test",
            "📁 Ingest Documents",
            "🔍 Ask Questions",
            "❤️ System Health",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by Gemini Embeddings · Weaviate · Llama 3.1 70B")


# ============================================================
# PAGE — Home
# ============================================================
def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🏥 Medical Healthcare RAG</h1>
            <p>Upload medical documents and get grounded, citation-backed answers
            across multiple input modalities — text, image, audio, PDF, and video.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Feature cards ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="card"><h4>📁 Document Ingestion</h4>
            <p>Upload PDF, TXT, or Markdown files. Documents are chunked into
            ~600-token windows with 100-token overlap and embedded using
            <strong>Gemini multimodal embeddings</strong> (3072-d).</p></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="card"><h4>🔍 Multimodal Querying</h4>
            <p>Ask questions using <strong>text, images, audio, PDFs, or video</strong>.
            Each modality is normalised through an input router before dense
            vector retrieval against Weaviate.</p></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="card"><h4>📖 Grounded Citations</h4>
            <p>Every answer is grounded in retrieved document chunks.
            <strong>Llama 3.1 70B</strong> (via Fireworks AI) generates responses
            with verbatim citations traceable to source documents.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- Pipeline architecture (matches LangGraph in src/pipeline/graph.py) ----
    st.subheader("🏗️ LangGraph pipeline (backend)")
    st.caption("Full path: input_router → retrieve (hybrid) → rerank → generate → citation_gate → output_route")
    r1 = st.columns(3)
    r2 = st.columns(3)
    steps = [
        (
            "1 · Input router",
            "Builds RAGState: query text, optional doc_id filter, modality metadata.",
        ),
        (
            "2 · Hybrid retrieve",
            "Gemini query embedding → Weaviate hybrid (BM25 + vector), top-M chunks "
            "(see HYBRID_ALPHA).",
        ),
        (
            "3 · Rerank",
            "Cohere rerank if COHERE_API_KEY is set; else top-K by retrieval score.",
        ),
        (
            "4 · Generate",
            "Fireworks LLM: answer + citation objects from retrieved context.",
        ),
        (
            "5 · Citation gate",
            "Abstains if there are no valid citations (trust / safety).",
        ),
        (
            "6 · Output route",
            "Fills image_url / pdf_url when top chunk modality + storage_ref warrant it.",
        ),
    ]
    for col, (title, blurb) in zip(r1 + r2, steps, strict=True):
        with col:
            st.markdown(
                f"""<div class="step"><h5>{title}</h5><p>{blurb}</p></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- Tech stack ----
    st.subheader("⚙️ Technology Stack")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
| Component        | Technology                          |
|------------------|-------------------------------------|
| REST API         | FastAPI + Uvicorn                   |
| Pipeline         | LangGraph (StateGraph)              |
| Vector DB        | Weaviate (HNSW / cosine)            |
| Embeddings       | `gemini-embedding-exp-03-07` 3072-d |
"""
        )
    with t2:
        st.markdown(
            """
| Component        | Technology                          |
|------------------|-------------------------------------|
| LLM Generation   | Fireworks AI · Llama 3.1 70B        |
| PDF Loading      | PyMuPDF (fitz)                      |
| Tokenisation     | tiktoken `cl100k_base`              |
| Frontend         | Streamlit                           |
"""
        )

    # ---- Quick-start ----
    st.markdown("---")
    with st.expander("🚀 Quick-start Guide"):
        st.markdown(
            """
1. **Environment**: `cp .env.example .env` — set `FIREWORKS_API_KEY`, `GEMINI_API_KEY`, and a Fireworks model your account allows.
2. **Stack**: `docker compose -f docker-compose.dev.yml up -d --build` — Weaviate, backend, Streamlit (optional MinIO, Next UI on port 3000).
3. **Guided test**: **🧪 Full flow test** — sample document, ingest steps, and example questions with what “good” looks like.
4. **Ad hoc**: **📁 Ingest Documents** → **🔍 Ask Questions** (text-only for `/query` today).
"""
        )


# ============================================================
# PAGE — Ingest Documents
# ============================================================
def render_ingest(api_url: str) -> None:
    st.title("📁 Ingest Documents")
    st.markdown(
        "Upload medical documents to the Weaviate vector store. "
        "Supported formats: **PDF** · **TXT** · **Markdown (.md)**"
    )
    st.divider()

    col_upload, col_meta = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your document here",
            type=["pdf", "txt", "md"],
            help="Recommended: ≤ 50 MB. Larger files may take longer to process.",
        )
        if uploaded_file:
            ext = uploaded_file.name.rsplit(".", 1)[-1].upper()
            st.info(
                f"📄 **{uploaded_file.name}** · {ext} · {uploaded_file.size:,} bytes"
            )

    with col_meta:
        st.markdown("#### Document Metadata")
        doc_id = st.text_input(
            "Document ID (optional)",
            placeholder="Auto-generated UUID4 if left empty",
            help="Stable identifier — reuse the same ID to update a document.",
        )
        doc_title = st.text_input(
            "Document Title (optional)",
            placeholder="Inferred from filename if left empty",
        )

    st.divider()

    ingest_clicked = st.button(
        "⬆️ Ingest Document",
        type="primary",
        disabled=uploaded_file is None,
    )

    if ingest_clicked:
        assert uploaded_file is not None  # button is disabled otherwise
        with st.spinner(f"Ingesting `{uploaded_file.name}` — embedding chunks, please wait…"):
            try:
                mime = uploaded_file.type or "application/octet-stream"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime)}
                data = {
                    "doc_id": doc_id.strip(),
                    "doc_title": doc_title.strip(),
                }

                with httpx.Client(timeout=180.0) as client:
                    resp = client.post(f"{api_url}/ingest", files=files, data=data)

                if resp.status_code == 201:
                    result = resp.json()
                    st.success("✅ Document ingested successfully!")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("📄 Title", result["doc_title"])
                    m2.metric("✂️ Chunks created", result["chunk_count"])
                    m3.metric("🆔 Doc ID (short)", result["doc_id"][:12] + "…")

                    # Store in session history
                    st.session_state["ingest_history"].append(
                        {
                            "filename": uploaded_file.name,
                            "doc_id": result["doc_id"],
                            "doc_title": result["doc_title"],
                            "chunk_count": result["chunk_count"],
                        }
                    )

                    with st.expander("📋 Full API Response"):
                        st.json(result)

                elif resp.status_code == 422:
                    detail = resp.json().get("detail", "Validation error")
                    st.error(f"❌ Validation error: {detail}")
                else:
                    st.error(f"❌ Ingest failed (HTTP {resp.status_code}): {resp.text}")

            except httpx.ConnectError:
                st.error(
                    f"❌ Cannot connect to `{api_url}`. "
                    "Make sure the FastAPI backend is running."
                )
            except httpx.TimeoutException:
                st.error("❌ Request timed out. Large documents can take a while — try again.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error: {exc}")

    # ---- Delete by Doc ID ----
    st.divider()
    st.subheader("🗑️ Remove Indexed Document")
    st.caption("Deletes all chunks for a given `doc_id` from Weaviate.")

    col_del_input, col_del_btn = st.columns([4, 1])
    with col_del_input:
        delete_doc_id = st.text_input(
            "Document ID to delete",
            placeholder="Paste full doc_id (UUID)",
            help="This removes the indexed chunks for this document.",
        )
    with col_del_btn:
        st.write("")
        st.write("")
        delete_clicked = st.button(
            "🗑️ Delete",
            type="secondary",
            disabled=not bool(delete_doc_id.strip()),
        )

    if delete_clicked:
        target_doc_id = delete_doc_id.strip()
        with st.spinner(f"Deleting chunks for `{target_doc_id}`…"):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.delete(f"{api_url}/ingest/{target_doc_id}")

                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"✅ Removed {result.get('deleted_chunks', 0)} chunk(s) "
                        f"for doc_id `{result.get('doc_id', target_doc_id)}`."
                    )
                    st.session_state["ingest_history"] = [
                        item
                        for item in st.session_state["ingest_history"]
                        if item.get("doc_id") != target_doc_id
                    ]
                    with st.expander("📋 Delete API Response"):
                        st.json(result)
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", "Validation error")
                    st.error(f"❌ Validation error: {detail}")
                else:
                    st.error(f"❌ Delete failed (HTTP {resp.status_code}): {resp.text}")
            except httpx.ConnectError:
                st.error(
                    f"❌ Cannot connect to `{api_url}`. "
                    "Make sure the FastAPI backend is running."
                )
            except httpx.TimeoutException:
                st.error("❌ Delete request timed out.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error: {exc}")

    # ---- Ingest history ----
    history = st.session_state["ingest_history"]
    if history:
        st.divider()
        st.subheader("📜 Ingestion History (this session)")
        for item in reversed(history):
            st.markdown(
                f"- **{item['doc_title']}** (`{item['filename']}`) · "
                f"{item['chunk_count']} chunks · ID `{item['doc_id'][:16]}…`"
            )

    # ---- How it works ----
    st.divider()
    with st.expander("ℹ️ How document ingestion works"):
        st.markdown(
            """
1. **Load** — PDF pages are extracted with PyMuPDF; `.txt`/`.md` are read as a single block.
2. **Chunk** — Text is split into overlapping windows of ~600 tokens (100-token stride) using `tiktoken cl100k_base`.
3. **Embed** — Each chunk is embedded concurrently via `gemini-embedding-exp-03-07` (3072-dimensional vectors).
4. **Store** — Vectors + metadata are batch-upserted into Weaviate's `MedicalChunk` collection (HNSW / cosine).
"""
        )


# ============================================================
# PAGE — Ask Questions
# ============================================================
def _render_pipeline_trace(result: dict) -> None:
    """
    Backend does not return per-node telemetry; infer a teaching trace from QueryResponse.
    """
    answer = (result.get("answer") or "").strip()
    citations = result.get("citations") or []
    n_cit = len(citations)
    abstain = ABSTAIN_SNIPPET in answer
    image_url = result.get("image_url")
    pdf_url = result.get("pdf_url")
    q_echo = (result.get("query") or "").strip()

    def line(icon: str, title: str, detail: str) -> None:
        st.markdown(f"{icon} **{title}** — {detail}")

    with st.expander("🧭 Pipeline trace (inferred from response)", expanded=True):
        line("✅", "1 · Input router", "Query accepted and built initial LangGraph state.")
        if q_echo:
            st.caption(f"Echo: `{q_echo[:120]}{'…' if len(q_echo) > 120 else ''}`")
        if n_cit > 0 or (answer and not abstain):
            line(
                "✅",
                "2 · Hybrid retrieve",
                f"Retrieval likely returned useful context ({n_cit} citation(s) in response).",
            )
        elif abstain:
            line(
                "⚠️",
                "2 · Hybrid retrieve",
                "Either no/few chunks matched, or later stages dropped citations — check ingest & doc filter.",
            )
        else:
            line("❓", "2 · Hybrid retrieve", "Unclear from response alone.")
        line(
            "ℹ️",
            "3 · Rerank",
            "Fireworks Qwen3-8B Serverless Rerank applied. (Scores are calculated internally ahead of LLM generation).",
        )
        if answer:
            line("✅", "4 · Generate", f"Answer length {len(answer)} chars.")
        else:
            line("❌", "4 · Generate", "No answer field.")
        if abstain:
            line("⚠️", "5 · Citation gate", "Abstain path — no surviving citations to ground the answer.")
        elif n_cit > 0:
            line("✅", "5 · Citation gate", f"{n_cit} citation(s) passed the gate.")
        else:
            line("⚠️", "5 · Citation gate", "Answer present but no structured citations in payload.")
        if image_url or (pdf_url and pdf_url != "EXPORT_PENDING"):
            line("✅", "6 · Output route", f"Media hints set (image={bool(image_url)}, pdf={bool(pdf_url)}).")
        else:
            line("✅", "6 · Output route", "Text-first response (no image / pending PDF export).")


def _render_query_result(result: dict) -> None:
    """Render POST /query JSON (QueryResponse shape)."""
    st.divider()
    st.subheader("📊 Query result")

    _render_pipeline_trace(result)

    answer: str | None = result.get("answer")
    if answer:
        if ABSTAIN_SNIPPET in answer:
            st.markdown(
                f"""<div class="abstain-box">
                    ⚠️ <strong>Citation gate / abstain</strong><br>{answer}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="answer-box">
                    <h4>💡 Answer</h4>
                    {answer}
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No `answer` in JSON — check backend logs and pipeline errors.")

    citations_raw = result.get("citations") or []
    if citations_raw:
        st.markdown(f"#### 📎 Citations ({len(citations_raw)})")
        for cit in citations_raw:
            if isinstance(cit, dict):
                cid = cit.get("chunk_id", "")
                title = cit.get("doc_title", "")
                quote = cit.get("quote", "")
                page = cit.get("page")
                section = cit.get("section")
                meta_bits = []
                if page is not None:
                    meta_bits.append(f"p.{page}")
                if section:
                    meta_bits.append(str(section))
                meta_str = " · ".join(meta_bits) if meta_bits else ""
                st.markdown(
                    f'<div class="citation-card"><strong>{title}</strong>'
                    f' <code>{cid}</code><br><em>{quote[:400]}{"…" if len(str(quote)) > 400 else ""}</em>'
                    f"{'<br><small>' + meta_str + '</small>' if meta_str else ''}</div>",
                    unsafe_allow_html=True,
                )
                cit_img_url = cit.get("image_url")
                if cit_img_url:
                    st.image(cit_img_url, use_container_width=True, caption=f"Image referenced in {title}")

    image_url = result.get("image_url")
    if image_url:
        st.markdown("#### 🖼️ Image URL (output router)")
        st.image(image_url, caption=image_url)

    pdf_url = result.get("pdf_url")
    if pdf_url:
        st.markdown(f"#### 📄 PDF hint: `{pdf_url}`")

    low = result.get("low_confidence")
    conf = result.get("confidence_score")
    if low is not None or conf is not None:
        st.caption(f"confidence_score={conf!r}, low_confidence={low!r}")

    with st.expander("📋 Raw JSON response"):
        st.json(result)


def render_query(api_url: str) -> None:
    st.title("🔍 Ask Questions")
    st.markdown(
        "Submit queries against your ingested medical documents. "
        "The backend route supports both text and multimodal uploads."
    )
    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Modality picker
        mod_keys = list(MODALITY_META.keys())
        mod_labels = [
            f"{MODALITY_META[m]['icon']}  {MODALITY_META[m]['label']}" for m in mod_keys
        ]
        mod_idx = st.selectbox(
            "Input Modality",
            options=range(len(mod_keys)),
            format_func=lambda i: mod_labels[i],
        )
        modality = mod_keys[mod_idx]
        st.caption(MODALITY_META[modality]["desc"])

        label = "Your Question" if modality == "text" else "Your Question (Optional)"
        question = st.text_area(
            label,
            placeholder=(
                "e.g. What are the contraindications of metformin in patients "
                "with renal impairment?"
            ),
            height=130,
        )


    with col_right:
        st.markdown("#### Query Options")
        doc_filter = st.text_input(
            "Filter by Document ID",
            placeholder="Leave empty to search all documents",
            help="Restrict retrieval to one ingested document",
        )
        top_k = st.slider(
            "Top-K retrieval chunks",
            min_value=1, max_value=20, value=5,
            help="How many chunks to retrieve before generation",
        )

        # File upload — now fully supported via multimodal endpoint file uploader
        uploaded_query_file = None
        if modality != "text":
            file_types = MODALITY_META[modality]["types"]
            uploaded_query_file = st.file_uploader(
                f"{MODALITY_META[modality]['icon']} Upload {modality} file",
                type=file_types,
                help=f"Required for the '{modality}' modality.",
            )

    st.divider()

    # Validation
    has_question = bool(question.strip())
    
    if modality == "text":
        can_submit = has_question
    else:
        can_submit = uploaded_query_file is not None

    if modality == "text" and not has_question and question:
        st.warning("Text queries cannot be empty.")
    elif modality != "text" and uploaded_query_file is None:
        st.warning(f"Please upload an {modality} file.")

    submit = st.button("🔍 Submit Query", type="primary", disabled=not can_submit)

    if submit:
        with st.spinner("Processing your query…"):
            try:
                with httpx.Client(timeout=90.0) as client:
                    if modality == "text":
                        payload = {
                            "query": question.strip(),
                            "doc_id": doc_filter.strip() or None,
                            "top_k": int(top_k),
                        }
                        resp = client.post(
                            f"{api_url}/query",
                            json=payload,
                        )
                    else:
                        assert uploaded_query_file is not None
                        # Multimodal endpoint supports multipart
                        form_data = {
                            "query": question.strip(),
                            "modality": modality,
                            "doc_id": doc_filter.strip() or None,
                            "top_k": int(top_k),
                        }
                        f_mime = uploaded_query_file.type or "application/octet-stream"
                        files = {"file": (uploaded_query_file.name, uploaded_query_file.getvalue(), f_mime)}
                        resp = client.post(
                            f"{api_url}/query/multimodal",
                            data=form_data,
                            files=files,
                        )

                if resp.status_code == 200:
                    result = resp.json()
                    # Store in session history
                    st.session_state["query_history"].append(
                        {
                            "question": question.strip(),
                            "modality": modality,
                            "answer": result.get("answer"),
                            "request_id": result.get("request_id", ""),
                        }
                    )
                    _render_query_result(result)

                elif resp.status_code == 422:
                    detail = resp.json().get("detail", "Validation error")
                    st.error(f"❌ Validation error: {detail}")
                else:
                    st.error(f"❌ Query failed (HTTP {resp.status_code}): {resp.text}")

            except httpx.ConnectError:
                st.error(
                    f"❌ Cannot connect to `{api_url}`. "
                    "Make sure the FastAPI backend is running."
                )
            except httpx.TimeoutException:
                st.error("❌ Request timed out.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error: {exc}")

    # ---- Query history ----
    history = st.session_state["query_history"]
    if len(history) > 1:
        st.divider()
        st.subheader("📜 Query History (this session)")
        for item in reversed(history[:-1]):  # omit the most recent (already shown)
            icon = MODALITY_META.get(item["modality"], {}).get("icon", "❓")
            ans = item["answer"] or "_No answer yet_"
            short_ans = (ans[:120] + "…") if len(ans) > 120 else ans
            with st.expander(f"{icon} {item['question'][:80]}"):
                st.markdown(f"**Answer:** {short_ans}")
                st.caption(f"Request ID: {item['request_id'][:12]}… · Modality: {item['modality']}")

    # ---- How it works ----
    st.divider()
    with st.expander("ℹ️ How querying works"):
        st.markdown(
            """
1. **Input router** — Builds `RAGState` from your text question, optional `doc_id`, `top_k`.
2. **Hybrid retrieve** — Gemini embedding + Weaviate hybrid (BM25 + vector).
3. **Rerank** — Cohere if configured; else score-order truncation to top-K.
4. **Generate** — Fireworks LLM produces `answer` + structured `citations`.
5. **Citation gate** — Replaces answer with abstain text if there are no citations.
6. **Output route** — May set `image_url` / `pdf_url` from top retrieved chunk.
"""
        )


# ============================================================
# PAGE — Full flow test (playbook)
# ============================================================
def render_full_flow_test(api_url: str) -> None:
    """Guided E2E: stack → ingest → query, aligned with LangGraph steps."""
    st.title("🧪 Full flow test")
    st.markdown(
        "Walk through the same path the **backend** runs: ingest into Weaviate, then "
        "`/query` through **input_router → hybrid retrieve → rerank → generate → "
        "citation_gate → output_route**."
    )
    st.divider()

    st.subheader("① Check the stack")
    health_data = fetch_health(api_url)
    if health_data:
        st.success(f"API `{api_url}` reachable — Weaviate: **{health_data.get('weaviate', '?')}**")
    else:
        st.error(f"API not reachable at `{api_url}`. Start compose or fix **API Base URL** in the sidebar.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Backend", "http://localhost:8000" if "localhost" in api_url or "127.0.0.1" in api_url else "see compose")
    with c2:
        st.metric("Weaviate REST", "http://localhost:8080")
    with c3:
        st.metric("Streamlit", "http://localhost:8501")

    st.caption(
        "Compose profile also starts MinIO (9000/9001) and optional Next UI (3000) when defined in "
        "`docker-compose.dev.yml`."
    )

    st.divider()
    st.subheader("② Seed corpus")
    sample_text = _sample_medical_fixture_text()
    st.download_button(
        label="⬇️ Download sample_medical.txt",
        data=sample_text.encode("utf-8"),
        file_name="sample_medical.txt",
        mime="text/plain",
        help="Upload this file on **📁 Ingest Documents** to create predictable Q&A.",
    )
    with st.expander("Preview sample document (first ~800 chars)"):
        st.code(sample_text[:800] + ("…" if len(sample_text) > 800 else ""), language=None)

    st.divider()
    st.subheader("③ Ingest")
    st.markdown(
        "1. Open **📁 Ingest Documents** in the sidebar.\n"
        "2. Upload `sample_medical.txt` (leave **Document ID** empty for a new UUID).\n"
        "3. Note **doc_id** and **chunk_count** — expect **multiple chunks** for this file.\n"
        "4. (Optional) Paste **Document ID** into **Filter by Document ID** when querying "
        "to restrict retrieval to this doc only."
    )

    st.divider()
    st.subheader("④ Ask — example questions & what “good” looks like")
    st.markdown(
        "Use **🔍 Ask Questions** with **Text** modality. After running, expand **Pipeline trace** "
        "on the result."
    )
    playbook_rows = [
        {
            "ID": row["id"],
            "Goal": row["goal"],
            "Question (copy into Ask page)": row["question"],
            "Good signs": row["good_signs"],
        }
        for row in FLOW_E2E_QUESTIONS
    ]
    st.dataframe(
        playbook_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("⑤ Interpret results")
    st.markdown(
        """
| Observation | Likely meaning |
|-------------|----------------|
| Answer + several citations | Retrieve + generate + citation gate happy path |
| Abstain message | No citations survived (empty retrieval, model omitted cites, or gate) |
| Citations with short quotes | Check quotes appear in your uploaded file |
| `image_url` / `pdf_url` | Output router; usually needs image/PDF chunks with `storage_ref` |
| Timeouts | Large ingest or slow LLM — increase patience or reduce corpus size |

**Ideas to try next:** re-ingest after editing the file (same `doc_id` replaces chunks); lower **Top-K**
to stress precision; ask **C1** after *only* ingesting medical text to see abstain behavior.
"""
    )

    st.divider()
    st.subheader("⑥ Automated tests (no UI)")
    st.code("uv run pytest tests/ -q", language="bash")


# ============================================================
# PAGE — System Health
# ============================================================
def render_health(api_url: str) -> None:
    st.title("❤️ System Health")
    st.markdown("Monitor the API and its service dependencies.")
    st.divider()

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        refresh = st.button("🔄 Refresh", type="primary")

    if refresh:
        fetch_health.clear()  # bust the cache

    st.divider()

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{api_url}/health")

        if resp.status_code == 200:
            h = resp.json()

            api_st = h.get("status", "unknown")
            wav_st = h.get("weaviate", "not_connected")
            version = h.get("version", "—")

            c1, c2, c3 = st.columns(3)
            with c1:
                if api_st == "ok":
                    st.success(f"{_status_badge(api_st)} API")
                    st.caption("All systems operational")
                else:
                    st.error(f"❌ API: {api_st}")

            with c2:
                if wav_st == "ok":
                    st.success(f"{_status_badge(wav_st)} Weaviate")
                    st.caption("Vector DB connected")
                elif wav_st == "degraded":
                    st.warning(f"{_status_badge(wav_st)} Weaviate")
                    st.caption("Degraded — check Weaviate logs")
                else:
                    st.error(f"{_status_badge(wav_st)} Weaviate")
                    st.caption("Not connected")

            with c3:
                st.info(f"ℹ️ Version: **{version}**")
                st.caption("Medical Healthcare RAG")

            st.divider()
            st.subheader("📋 Raw /health Response")
            st.json(h)

        else:
            st.error(f"❌ /health returned HTTP {resp.status_code}")

    except httpx.ConnectError:
        st.error(f"❌ Cannot reach `{api_url}`")
        st.info(
            "To start the backend:\n"
            "```bash\n"
            "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload\n"
            "```"
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")

    # ---- Config reference ----
    st.divider()
    st.subheader("🔧 Configuration Reference")
    st.markdown(f"**API URL configured:** `{api_url}`")

    with st.expander("📦 Service Dependencies"):
        st.markdown(
            """
| Service        | Role                              | Default endpoint          |
|----------------|-----------------------------------|---------------------------|
| FastAPI        | REST API backend                  | `http://localhost:8000`   |
| Weaviate       | Vector database (HNSW / cosine)   | `http://localhost:8080`   |
| Google AI Studio | Gemini multimodal embeddings    | `aistudio.google.com`     |
| Fireworks AI   | LLM generation (Llama 3.1 70B)    | `api.fireworks.ai`        |
"""
        )

    with st.expander("🌱 Required Environment Variables"):
        st.code(
            """\
FIREWORKS_API_KEY=fw_...
GEMINI_API_KEY=AIza...
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=          # optional for local Weaviate
FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct
GEMINI_EMBEDDING_MODEL=gemini-embedding-exp-03-07
PROMPT_VERSION=v1
RETRIEVAL_TOP_K=5
""",
            language="bash",
        )


# ============================================================
# Router
# ============================================================
if page == "🏠 Home":
    render_home()
elif page == "🧪 Full flow test":
    render_full_flow_test(api_url)
elif page == "📁 Ingest Documents":
    render_ingest(api_url)
elif page == "🔍 Ask Questions":
    render_query(api_url)
elif page == "❤️ System Health":
    render_health(api_url)
