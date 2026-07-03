from typing import Sequence
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    AsyncDenseVectorStrategy,
)
from llama_index.core.schema import BaseNode
from src.Infrastructure.dbcontext.context import settings
from src.Infrastructure.llm.index import local_llm
from src.Infrastructure.embedding_model.index import local_embed_model

custom_mapping = {
    "properties": {
        "embedding": {
            "type": "dense_vector",
            "dims": 384,  # value for current embed model: all-minilm:latest.
            "index": True,
            "similarity": "cosine",
            "element_type": "byte",
        }
    }
}

vector_store = ElasticsearchStore(
    es_url=settings.ELASTIC_URL,
    index_name=settings.ES_INDEX_NAME,
    es_user=settings.ELASTIC_USERNAME,
    es_password=settings.ELASTIC_PASSWORD,
    vector_store_kwargs={"_source": ["text"]},
    user_mapping=custom_mapping,
    # retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    llm=local_llm,
    embed_model=local_embed_model,
)
