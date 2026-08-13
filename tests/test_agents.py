"""
Tests for the agents.

Tests the query rewriter (short queries get expanded) and
the citation validator (grounded vs. not grounded detection).
These tests don't require a running LLM — they test the logic
and heuristics directly.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.query_rewriter import rewrite_query, _needs_rewrite
from app.agents.citation_validator import validate_citation


# ─── Query Rewriter Tests ───────────────────────────────────────────────────

def test_short_query_needs_rewrite():
    """Short queries (under 8 words) should be flagged for rewriting."""
    assert _needs_rewrite("explain that") is True
    assert _needs_rewrite("what is BERT") is True
    assert _needs_rewrite("how does it work") is True


def test_long_query_no_rewrite_needed():
    """Long queries without pronouns should not need rewriting."""
    query = (
        "What are the key differences between supervised and "
        "unsupervised learning approaches in modern machine learning"
    )
    assert _needs_rewrite(query) is False


def test_pronoun_query_needs_rewrite():
    """Queries with context pronouns should be flagged for rewriting."""
    assert _needs_rewrite("Can you explain that concept in more detail please") is True
    assert _needs_rewrite("What does it mean when the model converges slowly") is True
    assert _needs_rewrite("Tell me more about this architecture pattern used here") is True


def test_rewrite_without_llm_returns_original():
    """Without an LLM, rewrite should return the original query."""
    query = "explain that"
    result = rewrite_query(query, llm=None)
    assert result == query, "Should return original when no LLM is provided"


def test_rewrite_preserves_good_query():
    """A well-formed query without pronouns should pass through unchanged."""
    query = (
        "What are the advantages of using attention mechanisms "
        "in transformer architectures for sequence modeling tasks"
    )
    result = rewrite_query(query, llm=None)
    assert result == query


# ─── Citation Validator Tests ────────────────────────────────────────────────

def test_citation_validator_without_llm():
    """Without an LLM, validator should return UNKNOWN status."""
    result = validate_citation(
        answer="BERT is a transformer model.",
        chunks="BERT uses bidirectional training.",
        llm=None,
    )
    assert result["status"] == "UNKNOWN"
    assert "explanation" in result


def test_citation_validator_returns_dict():
    """Validator should always return a dict with status and explanation."""
    result = validate_citation(
        answer="Some answer",
        chunks="Some context",
        llm=None,
    )
    assert isinstance(result, dict)
    assert "status" in result
    assert "explanation" in result
    assert result["status"] in ("GROUNDED", "NOT_GROUNDED", "UNKNOWN")
