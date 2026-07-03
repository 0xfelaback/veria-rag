import time
from pipecat.services.llm_service import FunctionCallParams
from src.Infrastructure.elastic_search.index import index

from src.Infrastructure.llm.index import response_synthesizer
from src.Infrastructure.llm.index import local_llm
from src.Infrastructure.embedding_model.index import local_embed_model
from loguru import logger

# Cache query engine to avoid recreating it on every call
_cached_query_engine = None


async def query_function_call(params: FunctionCallParams, query: str):
    """Queries the internal context knowledge base to find relevant semantic information."""

    global _cached_query_engine

    try:
        logger.info(f"[RAG] Starting query: '{query}'")
        total_start = time.perf_counter()

        # Create or reuse query engine (cached for async efficiency)
        if _cached_query_engine is None:
            logger.info("[RAG] Creating cached query engine...")
            engine_start = time.perf_counter()
            _cached_query_engine = index.as_query_engine(
                similarity_top_k=3,
                streaming=False,
                response_synthesizer=response_synthesizer,
                llm=local_llm,
                vector_store_kwargs={"num_candidates": 20},
                search_kwargs={"num_candidates": 50},
            )
            engine_time = (time.perf_counter() - engine_start) * 1000
            logger.info(f"[RAG] Query engine creation (cached): {engine_time:.2f}ms")
        else:
            logger.info("[RAG] Using cached query engine")

        # Time query execution (embedding + search + synthesis)
        query_start = time.perf_counter()
        response = await _cached_query_engine.aquery(query)
        query_time = (time.perf_counter() - query_start) * 1000
        logger.info(
            f"[RAG] Query execution (embedding + search + synthesis): {query_time:.2f}ms"
        )

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"[RAG] Total RAG query time: {total_time:.2f}ms")

        await params.result_callback({"retrieved_context": str(response)})
    except Exception as e:
        logger.error(f"[RAG] Query failed: {e}")
        await params.result_callback({"error": f"Failed to retrieve data: {str(e)}"})


async def prewarm_embedding_model():
    """Pre-warms the embedding model by making a dummy embedding call."""
    try:
        logger.info("Pre-warming embedding model (nomic-embed-text)...")
        # dummy embedding call to load the embed model into memory
        await local_embed_model.aget_text_embedding("warmup")
        logger.info("Embedding model pre-warmed successfully")
    except Exception as e:
        logger.error(f"Failed to pre-warm embedding model: {e}")
