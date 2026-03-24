# Project understanding
Build a multimodal medical-document assistant that answers questions using your corpus and always shows source citations.
The core user problem is trust and speed: users need quick answers grounded in exact document evidence.
The smallest useful version is: upload a PDF, ask a text question, get an answer with valid chunk citations.
## Recommended tech stack with reasons
### Frontend: Next.js App Router + Tailwind + Aceternity UI
What it is: React framework with server-first routing and production defaults.
Why fit: fast iteration, clear routing, clean composable UI primitives.
Why better than simpler alternatives: compared to plain React/Vite, you get routing, SSR, and deployment ergonomics out of the box.
Role: query/upload UI, response rendering with citations and optional image preview.
Tradeoffs: slightly more framework conventions.
### Backend: FastAPI (Python 3.12)
What it is: async API framework with typed validation.
Why fit: ideal for async LLM/RAG calls and strict request/response schemas.
Why better: simpler than Django for API-first systems; better typed flow than Flask.
Role: /ingest, /query, /health endpoints.
Tradeoffs: must enforce good async hygiene.
### Orchestration: LangGraph
What it is: state-machine orchestration for LLM pipelines.
Why fit: your workflow is naturally node-based (route → retrieve → rerank → generate → output route).
Why better: clearer than ad-hoc service chains as complexity grows.
Role: deterministic workflow execution and branching.
Tradeoffs: learning curve on graph/state design.
### Vector + hybrid retrieval: Weaviate
What it is: vector database with native BM25+dense hybrid retrieval.
Why fit: healthcare queries need exact-term and semantic retrieval together.
Why better: avoids dual infra (separate vector DB + search engine).
Role: store chunks/metadata/embeddings and serve retrieval.
Tradeoffs: heavier local service than lightweight embedded stores.
### Embeddings: Vertex AI Gemini multimodal embeddings
What it is: shared embedding space for text/image/video signals.
Why fit: unified cross-modal retrieval (text queries can retrieve relevant images).
Why better: simpler than maintaining separate text+image embedding systems and fusion logic.
Role: embed chunks at ingest and queries at retrieval.
Tradeoffs: cloud dependency and credential setup.
### Generation: Fireworks AI — Llama 3.1 70B Instruct
What it is: Meta's Llama 3.1 70B instruction-tuned model served via Fireworks AI inference API.
Why fit: fast inference, OpenAI-compatible API (drop-in client swap), strong instruction-following for structured JSON output.
Why better than GPT-4o: lower cost per token, no OpenAI dependency, same API shape so zero integration overhead; Fireworks AI also offers lower latency for high-throughput RAG workloads.
Model ID: `accounts/fireworks/models/llama-v3p1-70b-instruct`
Role: answer generation from retrieved evidence with mandatory citation JSON.
Tradeoffs: structured output enforcement may need explicit JSON schema prompting; test citation compliance carefully.
### File storage: MinIO local, S3-compatible in prod
What it is: object storage for image/audio/video blobs.
Why fit: keeps binary payloads out of DB; use storage_ref pointers.
Why better: more scalable than filesystem blobs.
Role: media persistence + signed URL serving.
Tradeoffs: one more service.
### Realtime/WebSockets: defer initially
What it is: streaming answer transport.
Why fit: not required for first vertical slices.
Why better to defer: reduces initial complexity.
Role later: token streaming and live progress.
Tradeoffs later: connection lifecycle handling.
### Background jobs/workers: lightweight worker process
What it is: async ingest execution outside request thread.
Why fit: large docs/media ingestion should not block API.
Why better: keeps API latency stable.
Role: ingestion/transcription/indexing jobs.
Tradeoffs: queue/retry semantics required.
### Observability: structlog (+ optional OpenTelemetry)
What it is: structured logs and trace export.
Why fit: need request-level visibility across RAG stages.
Why better: faster diagnosis than plain logs.
Role: request_id, stage timings, error traces.
Tradeoffs: extra setup.
### Testing: pytest + httpx AsyncClient
What it is: Python unit/integration testing stack.
Why fit: natural for FastAPI async endpoints.
Why better: lightweight and composable fixtures/mocks.
Role: node tests, API tests, regression gates.
Tradeoffs: integration tests need services running.
### Docker/dev environment: Docker Compose + multi-stage images
What it is: reproducible local/prod environments.
Why fit: Weaviate + storage + app orchestration locally.
Why better: prevents environment drift and speeds onboarding.
Role: local stack + deploy artifact.
Tradeoffs: slower first-time setup.
## High-level architecture
Input (text/image/audio/pdf/video) enters FastAPI and is normalized by an input router node.
Ingest path chunks content and stores metadata + embeddings + storage references.
Query path performs hybrid retrieval, optional reranking, citation gate, then LLM generation.
A deterministic output router decides text/image/pdf fields in one response schema.
All node data flows through typed shared graph state.
## Step-by-step implementation roadmap
### Step 1: Vertical slice MVP (text-only, cited answers)
Goal
Deliver the smallest end-to-end useful product.
Why this step now
Proves core value before multimodal and infra complexity.
Tech involved
FastAPI, LangGraph, Weaviate, Fireworks AI Llama 3.1 70B.
What to implement
Create /ingest for PDF text chunking and /query for text questions; implement retrieve→generate with citation IDs.
Project files/folders likely affected
src/api/, src/pipeline/, src/ingest/, src/retrieval/, src/generation/, prompts/v1/.
How to test locally
Ingest one PDF, ask 5 known-answer questions, verify cited_chunk_ids exist.
What done looks like
API returns grounded answer plus valid citations for each query.
Common mistakes to avoid
Overbuilding auth/UI/streaming now; keep text-only path first.
What to build next after this passes
Step 2 input normalization and stronger schema contracts.
### Step 2: Input router normalization
Goal
Normalize text/image/audio/pdf/video request shapes into one graph state contract.
Why this step now
Prevents later branching chaos and inconsistent payload handling.
Tech involved
FastAPI multipart parsing, router node, Whisper/GPT-vision stubs.
What to implement
Input router node producing query_text + modality metadata + refs.
Project files/folders likely affected
src/router/input_router.py, src/schemas/query.py, src/pipeline/state.py.
How to test locally
Send sample text/image/audio requests; confirm normalized JSON state shape is identical except modality fields.
What done looks like
Every modality yields valid state and does not break downstream nodes.
Common mistakes to avoid
Mixing modality-specific logic inside retrieval/generation.
What to build next after this passes
Step 3 robust ingest schema and metadata completeness.
### Step 3: Ingest schema + chunk quality
Goal
Reliable chunking and metadata persistence for retrieval and citations.
Why this step now
Retrieval quality depends directly on chunk quality and metadata correctness.
Tech involved
Chunker, Weaviate schema, embedding service integration.
What to implement
500–800 token chunks with overlap, persist modality_type/caption/page/section/doc_id/chunk_id/storage_ref.
Project files/folders likely affected
src/ingest/chunking.py, src/ingest/pipeline.py, src/services/weaviate_client.py.
How to test locally
Ingest sample corpus, inspect random chunks and metadata, verify deterministic chunk IDs.
What done looks like
Corpus objects are queryable with complete metadata and no schema drift.
Common mistakes to avoid
Huge chunks, missing page/section, non-deterministic IDs.
What to build next after this passes
Step 4 hybrid retrieval baseline.
### Step 4: Retrieval baseline (dense + hybrid)
Goal
Establish reliable retrieval for exact terms and semantic intent.
Why this step now
You need measurable retrieval behavior before reranking.
Tech involved
Weaviate hybrid query, embeddings.
What to implement
Top-k retrieval with modality/doc filters; HYBRID_ALPHA config.
Project files/folders likely affected
src/retrieval/dense.py, src/retrieval/hybrid.py, src/utils/config.py.
How to test locally
Run query set (exact term + semantic) and compare dense-only vs hybrid outputs.
What done looks like
Hybrid clearly improves exact-term queries without major semantic regression.
Common mistakes to avoid
No benchmark set, hardcoded k/alpha.
What to build next after this passes
Step 5 reranking for precision.
### Step 5: Rerank + citation enforcement
Goal
Improve precision and block unsupported answers.
Why this step now
After retrieval baseline, precision and trust are highest ROI.
Tech involved
Cohere rerank or cross-encoder, citation gate logic.
What to implement
Top-M→top-K reranking and cited_chunk_ids subset validation with abstain path.
Project files/folders likely affected
src/retrieval/rerank.py, src/generation/citation_gate.py, prompts/v1/.
How to test locally
Use adversarial and out-of-corpus questions; verify abstain behavior and citation subset checks.
What done looks like
No answer ships without valid evidence alignment.
Common mistakes to avoid
Letting LLM cite non-retrieved chunks.
What to build next after this passes
Step 6 deterministic output routing.
### Step 6: Policy/output router
Goal
Return one consistent response schema with text/image/pdf branches.
Why this step now
Stabilizes contract for frontend and clients before deployment.
Tech involved
LangGraph conditional edges, response schemas.
What to implement
Rule-based branch population by retrieval modality and confidence thresholds.
Project files/folders likely affected
src/router/output_router.py, src/schemas/response.py, src/pipeline/graph.py.
How to test locally
Simulate hits per modality and verify only correct fields populate.
What done looks like
Single response shape with deterministic branching.
Common mistakes to avoid
Using a second LLM for policy routing.
What to build next after this passes
Step 7 media retrieval and signed URLs.
### Step 7: Visual/media retrieval slice
Goal
Return exact image-from-corpus when requested.
Why this step now
Completes real multimodal user value, not just text answers.
Tech involved
MinIO/S3, storage service, signed URLs.
What to implement
Ingest image objects, store storage_ref, return signed URL when top evidence is image.
Project files/folders likely affected
src/services/storage.py, src/ingest/loaders.py, src/router/output_router.py.
How to test locally
Ingest figure image, ask “show diagram”, open returned URL and verify exact bytes.
What done looks like
System returns original indexed image, not hallucinated generation.
Common mistakes to avoid
Storing raw binaries in Weaviate or returning permanent public URLs.
What to build next after this passes
Step 8 containerize and local stack hardening.
### Step 8: Containerization + local stack
Goal
Reproducible local stack and deploy-ready images.
Why this step now
Functional core is ready; now eliminate environment drift.
Tech involved
Docker multi-stage, Compose, healthchecks, non-root runtime.
What to implement
Dockerfiles for API/worker, compose for app+weaviate+minio, .dockerignore, health endpoints.
Project files/folders likely affected
docker/, docker-compose.yml, .dockerignore, src/api/app.py.
How to test locally
docker compose up from clean checkout, run ingest/query smoke tests.
What done looks like
One command boots full local system reliably.
Common mistakes to avoid
Root containers, oversized images, missing healthchecks.
What to build next after this passes
Step 9 observability + readiness/retries.
### Step 9: Observability + readiness + retries
Goal
Make the system diagnosable and resilient under startup/race failures.
Why this step now
Needed before stable early deployments.
Tech involved
structlog, request IDs, timing middleware, retry/backoff.
What to implement
Structured logs, stage timings, /health/live and /health/ready, startup dependency checks.
Project files/folders likely affected
src/utils/logging.py, src/utils/retry.py, src/api/middleware.py.
How to test locally
Bring dependencies down/up and verify controlled retries and readiness behavior.
What done looks like
No crash loops on dependency lag; each request traceable by request_id.
Common mistakes to avoid
Unbounded retries and unstructured print logs.
What to build next after this passes
Step 10 staged deployment.
### Step 10: Staged deployment + post-deploy quality gates
Goal
Ship safely, then enforce quality metrics as usage data accumulates.
Why this step now
Deployment should be simple first, then tighten with CI quality gates.
Tech involved
Managed hosting, CI pipelines, RAGAS.
What to implement
Initial deploy with core slices; then add golden set, offline eval, PR fail thresholds.
Project files/folders likely affected
.github/workflows/, eval/, deployment manifests.
How to test locally
Run eval script on golden subset and verify threshold fail behavior.
What done looks like
Deployed app stable; CI blocks regressions once metrics are established.
Common mistakes to avoid
Blocking first deployment on perfect eval coverage.
What to build next after this passes
Optional auth/tenancy, streaming, advanced optimization.
## Local testing strategy
Use short loops: implement one slice, run unit tests, run one integration scenario, run manual smoke query, then freeze before next step.
Baseline commands per slice: lint/typecheck, unit tests, one integration test, curl smoke test.
Keep a small repeatable local fixture corpus for deterministic checks.
Do not begin a new slice until current slice has passing checks and explicit “done” evidence.
## Deployment plan
First deployment should include only stable core slices: steps 1–8 plus basic health endpoints.
Keep advanced quality gates, large eval suites, and optional tenancy as later stages.
Start simple: one API service, one worker, managed/vector service, object storage, immutable image tags.
Then evolve to production: add observability depth, CI image scan/push, eval thresholds, runtime hardening.
Deploy in stages: dev environment first, then limited production traffic, then full rollout.
## Future improvements
MVP scope: steps 1–8.
Near-term improvements: step 9 and 10 quality gates, plus prompt version A/B testing.
Optional later features: JWT tenancy filters, websocket streaming, semantic caching, full OTel traces, domain-specific eval expansion.