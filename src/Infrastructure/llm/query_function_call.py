import time
import asyncio
from pipecat.services.llm_service import FunctionCallParams
from src.Infrastructure.elastic_search.index import index
from src.Infrastructure.embedding_model.index import local_embed_model
from loguru import logger

_cached_retriever = None


async def query_function_call(params: FunctionCallParams, query: str):
    """Queries the internal context knowledge base and injects raw retrieved nodes into the LLM context."""

    global _cached_retriever

    try:
        logger.info(f"[RAG] Starting query: '{query}'")
        total_start = time.perf_counter()

        if _cached_retriever is None:
            logger.info("[RAG] Creating cached retriever...")
            engine_start = time.perf_counter()
            _cached_retriever = index.as_retriever(
                similarity_top_k=2,
                embed_model=local_embed_model,
                verbose=False,
            )
            engine_time = (time.perf_counter() - engine_start) * 1000
            logger.info(f"[RAG] Retriever creation (cached): {engine_time:.2f}ms")
        else:
            logger.info("[RAG] Using cached retriever")

        query_start = time.perf_counter()
        retrieved_nodes = await asyncio.to_thread(_cached_retriever.retrieve, query)
        query_time = (time.perf_counter() - query_start) * 1000
        logger.info(f"[RAG] Retrieval execution time: {query_time:.2f}ms")

        raw_context_segments = []
        for idx, node_with_score in enumerate(retrieved_nodes, start=1):
            try:
                node_text = node_with_score.get_text().strip()
            except Exception:
                try:
                    node_text = node_with_score.get_content().strip()
                except Exception:
                    node_text = str(node_with_score)

            node_text = " ".join(node_text.split())
            node_text = node_text[:1800]
            score = node_with_score.get_score()
            raw_context_segments.append(
                f"--- Retrieved document {idx} (score={score:.3f}) ---\n{node_text}"
            )

        retrieved_context = (
            "\n\n".join(raw_context_segments)
            if raw_context_segments
            else "Context is insufficient."
        )

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"[RAG] Total RAG query time: {total_time:.2f}ms")
        logger.info(
            f"[RAG] Prepared {len(raw_context_segments)} raw context segments for the assistant"
        )

        await params.result_callback(retrieved_context)
    except Exception as e:
        logger.error(f"[RAG] Query failed: {e}")
        await params.result_callback(f"Context retrieval failed: {str(e)}")


async def prewarm_embedding_model():
    """Pre-warms the embedding model by making a dummy embedding call."""
    try:
        logger.info("Pre-warming embedding model (nomic-embed-text)...")
        # dummy embedding call to load the embed model into memory
        await local_embed_model.aget_text_embedding("warmup")
        logger.info("Embedding model pre-warmed successfully")
    except Exception as e:
        logger.error(f"Failed to pre-warm embedding model: {e}")
