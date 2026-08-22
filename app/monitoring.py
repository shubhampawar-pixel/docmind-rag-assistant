"""
DocMind FastAPI Monitoring — Prometheus Metrics & Telemetry Exporter.

Tracks API latency (P50/P95/P99), request counts, token usage, and error metrics.
"""

import time
from typing import Callable

from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Prometheus Metrics Definitions
REQUEST_COUNT = Counter(
    "docmind_requests_total",
    "Total HTTP requests to DocMind API",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "docmind_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"]
)

QUERY_COUNT = Counter(
    "docmind_rag_queries_total",
    "Total RAG search queries executed",
    ["provider"]
)

GROUNDING_STATUS_COUNT = Counter(
    "docmind_grounding_status_total",
    "Citation validator grounding classification counts",
    ["status"]
)


def get_metrics():
    """Generate latest Prometheus metrics payload."""
    return generate_latest(), CONTENT_TYPE_LATEST
