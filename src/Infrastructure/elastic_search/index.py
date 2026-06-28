from typing import Sequence
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    AsyncDenseVectorStrategy,
)
from llama_index.core.schema import BaseNode
from src.Infrastructure.dbcontext.context import settings
from src.Infrastructure.embedding_model.index import local_embed_model
from src.Infrastructure.llm.index import local_llm

vector_store = ElasticsearchStore(
    es_url=settings.ELASTIC_URL,
    index_name=settings.ES_INDEX_NAME,
    es_user=settings.ELASTIC_USERNAME,
    es_password=settings.ELASTIC_PASSWORD,
    # retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=local_embed_model,
    llm=local_llm,
)


def create_elasticsearch_index(nodes: Sequence[BaseNode]):
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )  # redirects embeddings storage to elastic

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=local_embed_model,
        llm=local_llm,
    )

    return index
