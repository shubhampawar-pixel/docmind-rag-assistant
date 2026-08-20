"""
LLM Chain — Multi-Tier API Fallback Router.

Routing Hierarchy:
  1. Google Gemini API (gemini-2.0-flash)
  2. Groq API (llama-3.1-8b-instant) — Cloud Backup
  3. Local Ollama (qwen2.5:0.5b) — Local Inference
"""

import urllib.request
import urllib.error

from loguru import logger

from app.config import (
    get_gemini_api_key,
    get_groq_api_key,
    GEMINI_MODEL,
    GROQ_MODEL,
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
    Get the appropriate LLM instance based on configuration and fallback priority.

    Args:
        temperature: LLM temperature (0.0-1.0).
        provider: Override for LLM_PROVIDER config. One of "gemini", "groq", "auto", "ollama".
        api_key: Optional explicit Gemini API key.

    Returns:
        A LangChain chat model instance.
    """
    effective_provider = provider or LLM_PROVIDER
    gemini_key = (api_key or "").strip() or get_gemini_api_key()
    groq_key = get_groq_api_key()

    if effective_provider == "gemini":
        return _get_gemini_llm(temperature, gemini_key)

    elif effective_provider == "groq":
        return _get_groq_llm(temperature, groq_key)

    elif effective_provider == "ollama":
        return _get_ollama_llm(temperature)

    else:  # "auto" mode with multi-tier fallback
        # Tier 1: Gemini API
        if gemini_key:
            try:
                logger.info("Auto mode: Using primary Gemini API")
                return _get_gemini_llm(temperature, gemini_key)
            except Exception as e:
                logger.warning(f"Gemini API init failed: {e}. Falling back to Groq...")

        # Tier 2: Groq API Fallback
        if groq_key:
            try:
                logger.info("Auto mode: Falling back to Groq API")
                return _get_groq_llm(temperature, groq_key)
            except Exception as e:
                logger.warning(f"Groq API init failed: {e}. Falling back to Ollama...")

        # Tier 3: Local Ollama
        if _is_ollama_running():
            logger.info("Auto mode: Falling back to local Ollama")
            return _get_ollama_llm(temperature)

        raise RuntimeError(
            "No active LLM available. Please enter a Gemini or Groq API Key in the sidebar, or start Ollama locally."
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


def _get_groq_llm(temperature: float, api_key: str):
    """Create a Groq LLM instance for cloud fallback."""
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    logger.info(f"Using Groq API with model: {GROQ_MODEL}")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
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
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")

        chunk_str = f"[Source: {source}, Page {page}]\n{text}"

        if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
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
    """Ask a question with retrieved context chunks."""
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
