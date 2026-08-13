"""
Query Rewriter Agent — Rewrites vague or short queries into
clear, standalone search queries using the LLM.

Only rewrites if the query is under 8 words or contains pronouns
like "it", "that", "this". Otherwise returns the original.
"""

import re

from loguru import logger


# Pronouns that indicate the query needs context
_CONTEXT_PRONOUNS = {"it", "that", "this", "these", "those", "they", "them"}

REWRITER_PROMPT = """You are a search query optimizer. The user has asked a question that may
be vague, short, or refer to previous context. Rewrite it as a clear,
standalone search query that will retrieve the most relevant documents.

Original question: {question}

Rewritten query (return ONLY the query, no explanation):"""


def _needs_rewrite(query: str) -> bool:
    """
    Check if the query needs rewriting.

    Returns True if the query is under 8 words or contains context pronouns.
    """
    words = query.lower().split()

    # Short queries need rewriting
    if len(words) < 8:
        return True

    # Queries with context pronouns need rewriting
    if _CONTEXT_PRONOUNS.intersection(set(words)):
        return True

    return False


def rewrite_query(query: str, llm=None) -> str:
    """
    Rewrite a query to be more clear and standalone.

    Args:
        query: The original user query.
        llm: A LangChain LLM instance. If None, returns the original query.

    Returns:
        The rewritten query, or the original if rewriting is not needed.
    """
    if not _needs_rewrite(query):
        logger.debug(f"Query does not need rewriting: {query[:60]}...")
        return query

    if llm is None:
        logger.warning("No LLM provided for query rewriting — returning original")
        return query

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate(
            template=REWRITER_PROMPT,
            input_variables=["question"],
        )
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": query}).strip()

        # Sanity check: if the LLM returns something too short or empty,
        # fall back to original
        if len(rewritten) < 5:
            logger.warning("Rewritten query too short — using original")
            return query

        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.error(f"Query rewriting failed: {e} — using original")
        return query
