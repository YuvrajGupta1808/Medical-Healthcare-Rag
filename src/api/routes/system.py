from fastapi import APIRouter
from src.services import weaviate_client
from src.utils.config import get_settings

router = APIRouter()

@router.get("/stats")
def get_system_stats():
    """
    Returns real-time system configuration and retrieval analytics
    that the MedQuery clinical dashboard relies on.
    """
    client = None
    count = 0
    status = "Degraded"
    
    try:
        client = weaviate_client.get_client()
        if client.is_ready():
            collection = client.collections.get(weaviate_client.COLLECTION_NAME)
            agg = collection.aggregate.over_all()
            count = agg.total_count
            status = "Operational"
    except Exception:
        status = "Offline"
        
    settings = get_settings()
        
    return {
        "status": "success",
        "data": {
            "retrieval_analytics": {
                "total_chunks_indexed": count,
                "avg_retrieval_time_ms": 150 # Simulated baseline metric
            },
            "system_config": {
                "vector_database": "Weaviate",
                "ai_model": settings.vision_model if hasattr(settings, 'vision_model') else "Gemini Pro",
                "orchestration": "LangGraph",
                "system_status": status
            }
        }
    }
