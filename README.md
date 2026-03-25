# 🩺 MedQuery: Enterprise Clinical-RAG Pipeline

MedQuery is a specialized Multimodal RAG (Retrieval-Augmented Generation) system designed for the medical and healthcare domain. It enables clinicians to query across diverse patient records (PDFs, Images, Audio, and Video) with full context isolation and grounded AI responses.

---

## 🚀 Core Technology Stack

- **Backend Logic**: [FastAPI](https://fastapi.tiangolo.com/) + [LangGraph](https://python.langchain.com/docs/langgraph) (Custom Clinical Workflows)
- **Vector Intelligence**: [Weaviate](https://weaviate.io/) (Hybrid Search + Multimodal Vectorization)
- **AI Models**: 
  - **Gemini Multimodal Embeddings** (Unified Vector Space)
  - **Fireworks Llama 3.1 70B** (Grounded Reasoning & Tool Use)
  - **Fireworks Whisper-v3-Turbo** (Medical Transcription)
- **Frontend Dashboard**: [Next.js 14](https://nextjs.org/) (Real-time SSE traces + Multimodal UI)
- **Infrastructure**: Docker + MinIO (Local Evidence Storage)

---

## 🧪 Key Features

- **Isolated Patient Context**: Every vector chunk is tagged with a `patient_id`. Global searches are strictly scoped to the active clinical case.
- **Multimodal Grounding**: Extract and display visual evidence (charts, timelines, and diagnostics) directly from Weaviate alongside the text answers.
- **SSE-Powered Execution Traces**: Observe the live pipeline status (**Routing → Retrieving → Generation**) as it executes.
- **Doctor-Centric Design**: Professional branding and high-contrast typography ("Robert Martinez") optimized for clinical environments.

---

## 🛠 Project Setup

### 1. Backend Engine (FastAPI)
The backend manages the LangGraph retrieval pipeline and Weaviate sync.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the engine
uvicorn src.main:app --reload --port 8000
```

### 2. Clinical Dashboard (Next.js)
The dashboard provides a real-time interface for ingestion and clinical dialogue.

```bash
cd ui
npm install
npm run dev # Runs explicitly on Port 3005
```

### 3. Vector Database (Docker)
Initialize the core storage infrastructure:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

---

## 🔎 Repository Structure

- `/src`: The core Python engine (Pipeline, API, Ingestion, Retrieval).
- `/ui`: Next.js frontend application.
- `/tests/assets`: Sample clinical test data (PDFs, WAV, TXT).
- `/prompts`: Version-controlled clinical instructions for the LLM.

---

## 📋 Roadmap & Guardrails

- [x] Full-Stack Multi-modal Integration
- [x] SSE Result Streaming
- [x] Cross-Modality Visualization (Images in Chat)

---
© 2026 Yuvraj Gupta