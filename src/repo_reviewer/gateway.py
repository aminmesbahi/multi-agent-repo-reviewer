"""The only place real generative inference happens: an OpenAI-compatible client pointed at
the LiteLLM Proxy. Agents depend on the ``ChatClient`` protocol, not on this class, so tests
and MOCK mode inject a fake without touching the agent code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import openai

from repo_reviewer.config import GatewaySettings

log = logging.getLogger("repo_reviewer.gateway")


class GatewayError(Exception):
    """The gateway could not produce a usable reply. The message is actionable."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as the model produced it


@dataclass(frozen=True)
class ModelReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        timeout_s: float | None = None,
    ) -> ModelReply: ...


class LiteLLMGateway:
    """OpenAI-compatible chat client for the LiteLLM Proxy (``/v1/chat/completions``)."""

    def __init__(self, settings: GatewaySettings, http_client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
            http_client=http_client,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        timeout_s: float | None = None,
    ) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if timeout_s is not None:
            kwargs["timeout"] = min(timeout_s, self.settings.timeout_s)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            raise GatewayError(
                f"cannot reach the LiteLLM Proxy at {self.settings.base_url}. "
                "Is it running? (docker compose up -d litellm)"
            ) from exc
        except openai.APITimeoutError as exc:
            raise GatewayError(f"model call timed out after {self.settings.timeout_s}s") from exc
        except openai.AuthenticationError as exc:
            raise GatewayError("gateway rejected LITELLM_MASTER_KEY (401)") from exc
        except openai.APIStatusError as exc:
            raise GatewayError(_status_message(exc, f"model '{self.settings.model}'")) from exc
        if not response.choices:
            raise GatewayError("gateway returned no choices")
        message = response.choices[0].message
        calls = [
            ToolCall(call.id, call.function.name, call.function.arguments or "")
            for call in (message.tool_calls or [])
            if call.type == "function"
        ]
        return ModelReply(content=message.content, tool_calls=calls)

    def list_models(self) -> list[str]:
        """Model aliases the proxy exposes (used by the smoke test)."""
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except openai.APIConnectionError as exc:
            raise GatewayError(
                f"cannot reach the LiteLLM Proxy at {self.settings.base_url}"
            ) from exc
        except openai.APIStatusError as exc:
            raise GatewayError(_status_message(exc, "/models")) from exc


def _status_message(exc: openai.APIStatusError, what: str) -> str:
    """A database-free LiteLLM Proxy reports a wrong master key as HTTP 400 'No connected db'."""
    if "No connected db" in str(exc) or exc.status_code in (401, 403):
        return (
            f"gateway rejected the request for {what} (HTTP {exc.status_code}): "
            "check that LITELLM_MASTER_KEY matches the proxy's key"
        )
    return f"gateway returned HTTP {exc.status_code} for {what}"
