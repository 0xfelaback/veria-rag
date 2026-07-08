from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from src.Infrastructure.dbcontext.context import settings

prompt_text = """You are an ultra-fast, precise Blockchain Technical Assistant.
            Your sole task is to extract the exact technical answer from the first relevant section found in the provided context.
            Do not scan the entire context if the target protocol, algorithm, or metric is found early.
            Be highly concise and use exact technical terminology (e.g., cryptographic primitives, consensus mechanisms).
            Answer in 1-2 sentences maximum.
            If the context does not contain the explicit answer, state immediately: "Context insufficient."
            Do not extrapolate, theorize, or explain background concepts."""

prompt_temp = "Please I need immediate answers, concise & immediate, no more than a sentence. Form an answer out of the first bit of context data you grab"

local_llm = Ollama(
    model=settings.OLLAMA_LLM_MODEL,
    prompt_key=prompt_temp,
    request_timeout=300.0,
    context_window=4096,
    temperature=0.1,
    streaming=True,
)
response_synthesizer = get_response_synthesizer(
    llm=local_llm,
    response_mode=ResponseMode.COMPACT,
    streaming=True,
)

from pipecat.services.ollama.llm import OLLamaLLMService

llm_pipecat = OLLamaLLMService(
    base_url=f"{settings.OLLAMA_BASE_URL}/v1",
    settings=OLLamaLLMService.Settings(
        model=settings.OLLAMA_LLM_MODEL,
        max_completion_tokens=64,
    ),
    extra_body={"options": {"num_ctx": 2048}},
)

Settings.llm = local_llm
