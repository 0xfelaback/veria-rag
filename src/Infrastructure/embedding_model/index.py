from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

local_embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)
Settings.embed_model = local_embed_model
