"""Security headers and request logging middleware."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from server.observability import bind_context, clear_context

logger = logging.getLogger("wonderwallai.server.middleware")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds hardened security headers to all responses (matching Jerry's hardening)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), interest-cohort=()"
        )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing, status, and a unique request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:12]

        # Bind request_id into async context — all downstream logs inherit it
        bind_context(request_id=request_id)
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(duration_ms, 2),
            },
        )
        response.headers["X-Request-ID"] = request_id
        clear_context()
        return response
