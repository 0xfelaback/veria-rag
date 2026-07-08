from llama_index.core.prompts import PromptTemplate
from src.Infrastructure.llm.index import local_llm

# 1. Define your template layout
blockchain_prompt = PromptTemplate(
    "{system_instructions}\n\nContext: {context}\n\nQuery: {query}"
)

# 2. Call the stream method to yield tokens
token_stream = local_llm.stream(
    blockchain_prompt,
    system_instructions=local_llm.prompt_key,
    context="Your technical documentation goes here...",
    query="What consensus algorithm is used in leo?",
)
print(list(token_stream))
for token in token_stream:
    print(f"data: {token}\n\n", end="", flush=True)
