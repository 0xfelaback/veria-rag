from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from src.Infrastructure.dbcontext.context import settings

local_embed_model = OllamaEmbedding(
    model_name=settings.OLLAMA_EMBEDDING_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)
Settings.embed_model = local_embed_model
