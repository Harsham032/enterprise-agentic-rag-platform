"""Optional language-model backends for answer synthesis and query planning.

The platform's default path is extractive and needs no model: answers are
assembled from sentences lifted verbatim out of retrieved chunks. That choice is
deliberate. It makes every claim trivially attributable, it makes groundedness
measurable rather than asserted, and it means the evaluation harness runs with
no credentials and no external dependency.

The backends below exist for deployments that want abstractive phrasing. They
are wired and unit-tested against fakes, but the benchmark in ``docs/results.md``
was produced with the extractive path only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..errors import GenerationError
from ..logging_utils import get_logger

logger = get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = (
    "You answer questions strictly from the numbered evidence passages supplied. "
    "Cite every claim with the bracketed passage number it came from, for example [2]. "
    "If the evidence does not answer the question, say so plainly instead of guessing."
)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal completion interface used by the planner and synthesiser."""

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        """Return a completion for ``prompt``."""


class AnthropicClient:
    """Completion backend backed by the Claude API."""

    def __init__(self, api_key: str, *, model: str = "claude-sonnet-5") -> None:
        if not api_key:
            raise GenerationError("an API key is required for this generation backend")
        try:
            import anthropic
        except ImportError as exc:
            raise GenerationError("the anthropic package is not installed") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIChatClient:
    """Completion backend backed by a chat-completions endpoint."""

    def __init__(self, api_key: str, *, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise GenerationError("an API key is required for this generation backend")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError("the openai package is not installed") from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


def build_llm_client(
    backend: str, *, anthropic_api_key: str = "", openai_api_key: str = ""
) -> LLMClient | None:
    """Return a client for ``backend``, or ``None`` for the extractive path."""
    if backend == "extractive":
        return None
    if backend == "anthropic":
        return AnthropicClient(anthropic_api_key)
    if backend == "openai":
        return OpenAIChatClient(openai_api_key)
    raise GenerationError(f"unknown generation backend: {backend}")
