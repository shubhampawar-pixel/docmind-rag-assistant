"""
LLM Chain — Local inference via Ollama with Google Gemini API fallback.

Supports three provider modes:
  - "ollama"  — local inference via ChatOllama (qwen2.5:0.5b)
  - "gemini"  — cloud inference via Google Gemini API
  - "auto"    — try Ollama first, fall back to Gemini if Ollama isn't reachable
"""

import urllib.request
import urllib.error

from loguru import logger

from app.config import (
    get_gemini_api_key,
    GEMINI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_CONTEXT_CHARS,
    LLM_PROVIDER,
)


SYSTEM_PROMPT = """You are DocMind, a precise knowledge assistant. You answer questions
based ONLY on the provided context chunks. For each point in your answer,
cite the source like this: [source.pdf, p.12]. If the context does not
contain enough information, say so explicitly. Never make up facts."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with citations):"""


def _is_ollama_running() -> bool:
    """Check if Ollama is reachable at the configured base URL."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_llm(temperature: float = 0.1, provider: str | None = None, api_key: str | None = None):
    """
    Get the appropriate LLM instance based on configuration.

    Args:
        temperature: LLM temperature (0.0-1.0).
        provider: Override for LLM_PROVIDER config. One of "ollama", "gemini", "auto".
        api_key: Optional explicit Gemini API key.

    Returns:
        A LangChain chat model instance.
    """
    effective_provider = provider or LLM_PROVIDER
    active_key = (api_key or "").strip() or get_gemini_api_key()

    if effective_provider == "gemini":
        return _get_gemini_llm(temperature, active_key)

    elif effective_provider == "ollama":
        return _get_ollama_llm(temperature)

    else:  # "auto" mode
        if _is_ollama_running():
            logger.info("Auto mode: Ollama is running — using local model")
            return _get_ollama_llm(temperature)
        elif active_key:
            logger.info("Auto mode: Ollama not available — falling back to Gemini API")
            return _get_gemini_llm(temperature, active_key)
        else:
            raise RuntimeError(
                "No LLM available. Please provide a Gemini API Key in the sidebar or start Ollama locally."
            )


def _get_gemini_llm(temperature: float, api_key: str):
    """Create a Gemini LLM instance."""
    if not api_key:
        raise RuntimeError(
            "Gemini API key is required. Please enter your API Key in the sidebar settings "
            "or configure GEMINI_API_KEY in Streamlit Secrets."
        )

    logger.info(f"Using Google Gemini API with model: {GEMINI_MODEL}")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )


def _get_ollama_llm(temperature: float):
    """Create an Ollama LLM instance."""
    logger.info(f"Using Ollama with model: {OLLAMA_MODEL}")
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


def build_context(chunks: list) -> str:
    """
    Build the context string from retrieved chunks.

    Truncates to MAX_CONTEXT_CHARS to fit within LLM context window.

    Args:
        chunks: List of chunk dicts with "text", "source", "page".

    Returns:
        Formatted context string with source citations.
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")

        chunk_str = f"[Source: {source}, Page {page}]\n{text}"

        if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
            # Truncate this chunk to fit
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:  # Only include if meaningful
                chunk_str = chunk_str[:remaining] + "..."
                context_parts.append(chunk_str)
            break

        context_parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n\n".join(context_parts)


def ask(
    question: str,
    chunks: list,
    temperature: float = 0.1,
    provider: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Ask a question with retrieved context chunks.

    Args:
        question: The user's question.
        chunks: Retrieved and reranked chunks.
        temperature: LLM temperature.
        provider: Override LLM provider ("ollama", "gemini", "auto").
        api_key: Optional Gemini API key.

    Returns:
        The LLM's answer string.
    """
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm(temperature=temperature, provider=provider, api_key=api_key)

    context = build_context(chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])

    chain = prompt | llm

    logger.info(f"Asking LLM: {question[:60]}...")
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    answer = response.content if hasattr(response, "content") else str(response)
    logger.info(f"LLM response received ({len(answer)} chars)")
    return answer
