"""
================================================================================
Skint Labs — Structured Observability Module
================================================================================
Shared across Jerry, WonderwallAi, and GiLLBoT.

Provides:
  - configure_logging()  — structlog + stdlib, JSON in prod, console in dev
  - bind_context()       — async-safe context binding (session_id, request_id, etc.)
  - log_decision()       — intent/decision logging primitive
  - log_llm_call()       — LLM call instrumentation
  - init_sentry()        — Sentry error tracking with FastAPI integration

Design principle: JSON to stdout. Railway captures it. No Prometheus,
no Grafana, no OTEL collector. Just structured, searchable, auditable logs.
================================================================================
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars  # noqa: F401

# ---------------------------------------------------------------------------
# PII redaction processor
# ---------------------------------------------------------------------------
_PII_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD_REDACTED]"),
]


def _redact_pii_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Scrub email, phone, SSN, and card patterns from log values."""
    for key, value in event_dict.items():
        if isinstance(value, str) and key not in ("event", "level", "timestamp", "logger"):
            for pattern, replacement in _PII_PATTERNS:
                value = pattern.sub(replacement, value)
            event_dict[key] = value
    return event_dict


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
def configure_logging(
    service_name: str = "skintlabs",
    environment: str = "development",
    log_level: str = "INFO",
    log_format: str = "json",
) -> None:
    """
    Configure structlog for the application.

    - Production (log_format="json"): JSON lines to stdout — Railway-friendly.
    - Development (log_format="console"): Coloured, human-readable output.

    This also patches stdlib logging so existing `logging.getLogger()` calls
    emit through structlog processors (same JSON/console format).
    """
    is_json = log_format == "json" or (log_format == "auto" and environment == "production")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _redact_pii_processor,
    ]

    if is_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Wire stdlib logging through structlog's formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "stripe"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.get_logger("observability").info(
        "logging_configured",
        service=service_name,
        environment=environment,
        format=log_format,
        level=log_level,
    )


# ---------------------------------------------------------------------------
# Context binding (async-safe via structlog.contextvars)
# ---------------------------------------------------------------------------
def bind_context(**kwargs: Any) -> None:
    """
    Bind contextual fields to the current async context.
    All subsequent log calls in this context will include these fields.

    Example:
        bind_context(session_id="abc123", store_id=42, client_ip="1.2.3.4")
        logger.info("message received")  # auto-includes session_id, store_id, client_ip
    """
    bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context (call at end of request/connection)."""
    clear_contextvars()


# ---------------------------------------------------------------------------
# Decision logging primitive
# ---------------------------------------------------------------------------
_decision_logger = structlog.get_logger("agent.decision")


def log_decision(
    decision_type: str,
    *,
    input_summary: str = "",
    options_considered: Optional[list[str]] = None,
    chosen: Any = None,
    reason: str = "",
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Log an agent decision — the intent logging primitive.

    This captures WHY a decision was made, not just what happened.
    Every AI decision point should call this.

    Args:
        decision_type: Category of decision (e.g., "intent_classification",
                       "product_search", "firewall_inbound", "escalation")
        input_summary: Truncated input that triggered this decision
        options_considered: What alternatives were available
        chosen: What was selected
        reason: WHY this was chosen (the key insight for auditability)
        confidence: Confidence score (0.0–1.0) if applicable
        latency_ms: How long this decision took
        metadata: Any additional context (scores, counts, etc.)
    """
    event_data: dict[str, Any] = {
        "decision_type": decision_type,
    }
    if input_summary:
        event_data["input_summary"] = input_summary[:200]
    if options_considered is not None:
        event_data["options_considered"] = options_considered
    if chosen is not None:
        event_data["chosen"] = chosen
    if reason:
        event_data["reason"] = reason
    if confidence is not None:
        event_data["confidence"] = round(confidence, 4)
    if latency_ms is not None:
        event_data["latency_ms"] = round(latency_ms, 2)
    if metadata:
        event_data["metadata"] = metadata

    _decision_logger.info("agent_decision", **event_data)


# ---------------------------------------------------------------------------
# LLM call instrumentation
# ---------------------------------------------------------------------------
_llm_logger = structlog.get_logger("agent.llm")


def log_llm_call(
    model: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    prompt_summary: str = "",
    completion_summary: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0.0,
    is_retry: bool = False,
    error: Optional[str] = None,
) -> None:
    """
    Log an LLM API call with token usage, latency, and summaries.

    Prompt/completion summaries should be first/last N chars — never the
    full prompt (privacy, log size).
    """
    event_data: dict[str, Any] = {
        "model": model,
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_in + tokens_out,
        },
        "latency_ms": round(latency_ms, 2),
    }
    if temperature is not None:
        event_data["temperature"] = temperature
    if max_tokens is not None:
        event_data["max_tokens"] = max_tokens
    if prompt_summary:
        event_data["prompt_summary"] = prompt_summary[:160]
    if completion_summary:
        event_data["completion_summary"] = completion_summary[:200]
    if is_retry:
        event_data["is_retry"] = True
    if error:
        event_data["error"] = error

    log_fn = _llm_logger.error if error else _llm_logger.info
    log_fn("llm_call", **event_data)


# ---------------------------------------------------------------------------
# Timer context manager (for wrapping code blocks)
# ---------------------------------------------------------------------------
class Timer:
    """Simple perf_counter timer for measuring latency."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000


# ---------------------------------------------------------------------------
# Sentry initialisation
# ---------------------------------------------------------------------------
def init_sentry(
    dsn: str,
    environment: str = "development",
    service_name: str = "skintlabs",
    traces_sample_rate: float = 0.1,
) -> None:
    """
    Initialise Sentry error tracking with FastAPI integration.
    No-op if dsn is empty.
    """
    if not dsn:
        structlog.get_logger("observability").info(
            "sentry_skipped", reason="no DSN configured"
        )
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=f"{service_name}@1.0.0",
            traces_sample_rate=traces_sample_rate,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
            ],
            # Don't send PII to Sentry
            send_default_pii=False,
        )
        structlog.get_logger("observability").info(
            "sentry_initialized",
            service=service_name,
            environment=environment,
        )
    except ImportError:
        structlog.get_logger("observability").warning(
            "sentry_unavailable", reason="sentry-sdk not installed"
        )
