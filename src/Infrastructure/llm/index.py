from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

local_llm = Ollama(
    model="llama3.1:8b-instruct-q4_K_M",
    request_timeout=300.0,
    context_window=8000,
    temperature=0.1,  # determinstic
)

Settings.llm = local_llm
