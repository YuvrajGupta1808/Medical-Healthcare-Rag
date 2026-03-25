from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.pipeline.state import RAGState


def build_rag_graph():
    """
    Build and compile the RAG LangGraph pipeline.

    Flow: input_router → retrieve → generate → output_route → END

    Each node is an async function that receives the full RAGState
    and returns a partial dict to merge back into state.
    """
    # Import node functions here to avoid circular imports at module load time
    from src.generation.generate import generate_node
    from src.generation.citation_gate import citation_gate_node
    from src.retrieval.hybrid import retrieve_node
    from src.retrieval.rerank import rerank_node
    from src.router.input_router import input_router_node
    from src.router.output_router import output_route_node

    graph: StateGraph = StateGraph(RAGState)

    # Step 2: input normalisation guard (pre-conditions validated here)
    graph.add_node("input_router", input_router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("citation_gate", citation_gate_node)
    graph.add_node("output_route", output_route_node)

    graph.add_edge(START, "input_router")
    graph.add_edge("input_router", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "citation_gate")
    graph.add_edge("citation_gate", "output_route")
    graph.add_edge("output_route", END)

    return graph.compile()


def get_rag_pipeline():
    """Return the compiled pipeline; built lazily on first call."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = build_rag_graph()
    return _rag_pipeline


_rag_pipeline = None

# NOTE: Do NOT call get_rag_pipeline() at module level.
# The graph must be built lazily (inside lifespan or on first request)
# so that import-time failures don't crash the app before startup hooks run.
# Callers should always use: from src.pipeline.graph import get_rag_pipeline
