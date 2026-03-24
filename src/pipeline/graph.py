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
    from src.retrieval.dense import retrieve_node
    from src.router.input_router import input_router_node
    from src.router.output_router import output_route_node

    graph: StateGraph = StateGraph(RAGState)

    # Step 2: input normalisation guard (pre-conditions validated here)
    graph.add_node("input_router", input_router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_route", output_route_node)

    graph.add_edge(START, "input_router")
    graph.add_edge("input_router", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_route")
    graph.add_edge("output_route", END)

    return graph.compile()


def get_rag_pipeline():
    """Return the compiled pipeline; built lazily on first call."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = build_rag_graph()
    return _rag_pipeline


_rag_pipeline = None

# Module-level alias used by the query route
rag_pipeline = get_rag_pipeline()
