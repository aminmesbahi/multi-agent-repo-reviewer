"""LiteLLMGateway against a fake HTTP transport. This checks request/response translation of
the OpenAI-compatible protocol; it does not (and cannot) check a real LiteLLM Proxy or model."""

from __future__ import annotations

import json

import httpx
import pytest

from repo_reviewer.config import ConfigError, GatewaySettings, gateway_settings
from repo_reviewer.gateway import GatewayError, LiteLLMGateway
from repo_reviewer.models import Mode

SETTINGS = GatewaySettings("http://127.0.0.1:4000/v1", "sk-test", "reviewer-local", max_retries=0)


def gateway(handler) -> LiteLLMGateway:
    return LiteLLMGateway(SETTINGS, httpx.Client(transport=httpx.MockTransport(handler)))


def completion(message: dict) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "reviewer-local",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
        ],
    }


def test_request_goes_to_the_proxy_with_alias_key_and_limits():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion({"content": "hi"}))

    reply = gateway(handler).complete([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert seen["url"] == "http://127.0.0.1:4000/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "reviewer-local"
    assert seen["body"]["max_tokens"] == 1024 and seen["body"]["temperature"] == 0.0
    assert reply.content == "hi" and reply.tool_calls == []


def test_tool_calls_are_translated():
    message = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }
        ],
    }
    reply = gateway(lambda r: httpx.Response(200, json=completion(message))).complete([], None)
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [
        ("call_1", "read_file", '{"path": "a.py"}')
    ]


def test_tools_omitted_when_none():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion({"content": "{}"}))

    gateway(handler).complete([], None)
    assert "tools" not in seen["body"]


@pytest.mark.parametrize(
    ("status", "needle"), [(401, "LITELLM_MASTER_KEY"), (404, "HTTP 404"), (500, "HTTP 500")]
)
def test_http_errors_become_actionable_gateway_errors(status, needle):
    gw = gateway(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(GatewayError, match=needle):
        gw.complete([], None)


def test_connection_error_mentions_how_to_start_the_proxy():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(GatewayError, match="docker compose up"):
        gateway(handler).complete([], None)


def test_no_choices_is_an_error():
    gw = gateway(lambda r: httpx.Response(200, json={**completion({}), "choices": []}))
    with pytest.raises(GatewayError, match="no choices"):
        gw.complete([], None)


def test_gateway_settings_from_env():
    env = {"LITELLM_MASTER_KEY": "sk-abc", "REVIEWER_TIMEOUT_S": "30", "REVIEWER_MAX_TOKENS": "256"}
    s = gateway_settings(Mode.DEEPSEEK, env)
    assert (s.model, s.timeout_s, s.max_tokens, s.api_key) == (
        "reviewer-deepseek",
        30.0,
        256,
        "sk-abc",
    )
    assert gateway_settings(Mode.LOCAL, env).model == "reviewer-local"
    assert gateway_settings(Mode.LOCAL, env).base_url == "http://127.0.0.1:4000/v1"


def test_gateway_settings_validate_input():
    with pytest.raises(ConfigError, match="LITELLM_MASTER_KEY"):
        gateway_settings(Mode.LOCAL, {})
    with pytest.raises(ConfigError, match="number"):
        gateway_settings(Mode.LOCAL, {"LITELLM_MASTER_KEY": "k", "REVIEWER_TIMEOUT_S": "soon"})
    with pytest.raises(ConfigError, match="MOCK"):
        gateway_settings(Mode.MOCK, {"LITELLM_MASTER_KEY": "k"})


def test_database_free_proxy_wrong_key_shows_as_400_and_is_explained():
    body = {"error": {"message": "Authentication Error, No connected db.", "code": "400"}}
    gw = gateway(lambda r: httpx.Response(400, json=body))
    with pytest.raises(GatewayError, match="LITELLM_MASTER_KEY"):
        gw.list_models()
    with pytest.raises(GatewayError, match="LITELLM_MASTER_KEY"):
        gw.complete([], None)
