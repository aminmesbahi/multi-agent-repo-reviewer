"""Runtime configuration, read from environment variables (see ``.env.example``).

No credential has a default value: the gateway key must come from the environment.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from repo_reviewer.models import Mode

DEFAULT_GATEWAY_URL = "http://127.0.0.1:4000/v1"
MODEL_ALIASES = {Mode.LOCAL: "reviewer-local", Mode.DEEPSEEK: "reviewer-deepseek"}


class ConfigError(Exception):
    """Invalid or missing configuration. The message is meant to be shown to the user."""


@dataclass(frozen=True)
class GatewaySettings:
    """How to reach the LiteLLM Proxy for *generative* (tool-calling) inference."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0
    max_retries: int = 1
    max_tokens: int = 1_024
    temperature: float = 0.0


@dataclass(frozen=True)
class DecisionSettings:
    """How to reach a decision model. Deliberately separate from ``GatewaySettings``."""

    api_base: str
    api_key: str
    model: str
    timeout_s: float = 15.0
    allow_remote: bool = False


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def gateway_settings(mode: Mode, env: Mapping[str, str] | None = None) -> GatewaySettings:
    """Settings for LOCAL or DEEPSEEK mode. MOCK never talks to a gateway."""
    env = os.environ if env is None else env
    if mode is Mode.MOCK:
        raise ConfigError("MOCK mode does not use the gateway")
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key:
        raise ConfigError(
            "LITELLM_MASTER_KEY is not set. Copy .env.example to .env, set a key "
            "(it must match the one the LiteLLM Proxy was started with) and export it."
        )
    return GatewaySettings(
        base_url=env.get("REVIEWER_GATEWAY_URL", DEFAULT_GATEWAY_URL),
        api_key=key,
        model=env.get("REVIEWER_MODEL", MODEL_ALIASES[mode]),
        timeout_s=_float(env, "REVIEWER_TIMEOUT_S", 120.0),
        max_retries=_int(env, "REVIEWER_MAX_RETRIES", 1),
        max_tokens=_int(env, "REVIEWER_MAX_TOKENS", 1_024),
        temperature=_float(env, "REVIEWER_TEMPERATURE", 0.0),
    )


def is_local_host(host: str | None) -> bool:
    """True for loopback addresses and the container-to-host alias. Names are not resolved."""
    if not host:
        return False
    if host in {"localhost", "host.docker.internal"} or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def decision_settings(env: Mapping[str, str] | None = None) -> DecisionSettings:
    """Settings for the (experimental) decision-model backend.

    Refuses non-local endpoints unless ``DECISION_ALLOW_REMOTE=1``, because the request
    text and project facts would otherwise leave the machine.
    """
    env = os.environ if env is None else env
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key:
        raise ConfigError("LITELLM_MASTER_KEY is not set (needed to call the decision endpoint).")
    api_base = env.get("DECISION_API_BASE", "http://127.0.0.1:4000").rstrip("/")
    allow_remote = env.get("DECISION_ALLOW_REMOTE", "") == "1"
    host = urlparse(api_base).hostname
    if not allow_remote and not is_local_host(host):
        raise ConfigError(
            f"DECISION_API_BASE host {host!r} is not local. Request text would be sent to it. "
            "Set DECISION_ALLOW_REMOTE=1 only if you accept that."
        )
    return DecisionSettings(
        api_base=api_base,
        api_key=key,
        model=env.get("DECISION_MODEL", "router-decision"),
        timeout_s=_float(env, "DECISION_TIMEOUT_S", 15.0),
        allow_remote=allow_remote,
    )
