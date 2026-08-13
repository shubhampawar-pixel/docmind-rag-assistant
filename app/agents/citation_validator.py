"""
Citation Validator Agent — Checks if LLM answers are grounded
in the retrieved source chunks.

Returns GROUNDED or NOT_GROUNDED with a one-sentence explanation.
If NOT_GROUNDED, the Streamlit UI shows a warning banner.
"""

from loguru import logger


VALIDATOR_PROMPT = """You are a fact-checking agent. You are given an answer and the source
chunks that were provided as context to generate it.

Answer: {answer}

Source chunks: {chunks}

Does the answer appear to be grounded in the source chunks?
Reply with: GROUNDED or NOT_GROUNDED, then one sentence explanation."""


def validate_citation(answer: str, chunks: str, llm=None) -> dict:
    """
    Validate whether the LLM answer is grounded in the source chunks.

    Args:
        answer: The LLM-generated answer.
        chunks: The concatenated source chunks provided as context.
        llm: A LangChain LLM instance.

    Returns:
        Dict with "status" (GROUNDED/NOT_GROUNDED) and "explanation".
    """
    if llm is None:
        logger.warning("No LLM provided for citation validation")
        return {
            "status": "UNKNOWN",
            "explanation": "Citation validation skipped (no LLM available).",
        }

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate(
            template=VALIDATOR_PROMPT,
            input_variables=["answer", "chunks"],
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "answer": answer,
            "chunks": chunks,
        }).strip()

        # Parse the result
        if result.upper().startswith("GROUNDED"):
            status = "GROUNDED"
        elif result.upper().startswith("NOT_GROUNDED"):
            status = "NOT_GROUNDED"
        else:
            # Try to detect from the full response
            if "NOT_GROUNDED" in result.upper():
                status = "NOT_GROUNDED"
            elif "GROUNDED" in result.upper():
                status = "GROUNDED"
            else:
                status = "UNKNOWN"

        # Extract explanation (everything after the status word)
        explanation = result
        for prefix in ["GROUNDED", "NOT_GROUNDED"]:
            if explanation.upper().startswith(prefix):
                explanation = explanation[len(prefix):].strip()
                # Remove leading punctuation
                if explanation and explanation[0] in ".,;:-":
                    explanation = explanation[1:].strip()
                break

        logger.info(f"Citation validation: {status}")
        return {
            "status": status,
            "explanation": explanation or "No explanation provided.",
        }

    except Exception as e:
        logger.error(f"Citation validation failed: {e}")
        return {
            "status": "UNKNOWN",
            "explanation": f"Validation error: {str(e)}",
        }
