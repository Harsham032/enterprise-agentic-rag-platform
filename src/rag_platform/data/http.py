"""HTTP plumbing shared by the external document sources.

Both SEC EDGAR and NCBI publish request-rate ceilings and require identifying
request headers. Exceeding either gets a client blocked, so rate limiting and
descriptive user agents are enforced here rather than left to call sites.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..errors import DataAcquisitionError
from ..logging_utils import get_logger

logger = get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    """Thread-safe minimum-interval limiter.

    A plain sleep-until-next-slot limiter is enough here: both upstreams specify
    a sustained request ceiling rather than a burst allowance.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class RetryableStatusError(DataAcquisitionError):
    """Raised for upstream status codes that are worth retrying."""


class HttpSource:
    """Base class for rate-limited JSON/XML HTTP clients."""

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        requests_per_second: float,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(requests_per_second)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers=headers, timeout=timeout, follow_redirects=True
        )

    def __enter__(self) -> HttpSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @retry(
        retry=retry_if_exception_type((RetryableStatusError, httpx.TransportError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        reraise=True,
    )
    def _request(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Issue a rate-limited GET with bounded exponential-backoff retries."""
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        self._limiter.acquire()
        try:
            response = self._client.get(url, params=params)
        except httpx.TransportError as exc:
            logger.warning("http_transport_error", url=url, error=str(exc))
            raise
        if response.status_code in RETRYABLE_STATUS:
            logger.warning("http_retryable_status", url=url, status=response.status_code)
            raise RetryableStatusError(f"{response.status_code} from {url}")
        if response.status_code >= 400:
            raise DataAcquisitionError(f"{response.status_code} from {url}: {response.text[:200]}")
        return response

    def _request_or_fail(self, path: str, params: dict[str, Any] | None) -> httpx.Response:
        """Run :meth:`_request`, converting an exhausted retry into a domain error.

        Tenacity re-raises the underlying transport exception once it gives up.
        Letting that escape would make every caller import httpx to handle a
        network outage, so it is translated at this boundary instead.
        """
        try:
            return self._request(path, params)
        except httpx.TransportError as exc:
            raise DataAcquisitionError(
                f"could not reach {path}: {exc}. Check network access to the upstream host "
                "and any egress policy in front of it."
            ) from exc

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self._request_or_fail(path, params)
        try:
            return response.json()
        except ValueError as exc:
            raise DataAcquisitionError(f"expected JSON from {response.url}") from exc

    def get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        return self._request_or_fail(path, params).text
