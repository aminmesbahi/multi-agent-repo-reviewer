# Agent routing and decision models

Status: **experimental extension**. The default behaviour of the project is unchanged: with no
`--router` and no `--request`, the deterministic planner runs every applicable agent.

## 1. Three ways to decide "which agents run"

| | Deterministic rules | Generative LLM router | Decision-model (System-1 style) router |
|---|---|---|---|
| Implemented here | `RuleBasedRouter` (default) | **No** (see below) | `DecisionModelRouter` (optional) |
| Input | Request text (optional) + project facts | Free text prompt | Request text + coarse facts, as typed questions |
| Output | Fixed code path | Free text you must parse | Typed probabilities, one per question |
| Handles paraphrase / other languages | No (keywords) | Yes | Yes, to the extent the model does |
| Failure mode | Misses unforeseen wording | Malformed text, hallucinated agent names, prompt injection | Wrong probability; transport errors |
| Cost per call | None | Full chat completion | One small classification call |
| Reproducible | Yes | Not without pinned model/seed | Mostly, for a fixed model |
| Safe to trust? | By construction | Only after strict parsing | Only after strict validation |

"System-1 style" here means a small model that answers fixed, typed questions quickly instead of
reasoning in free text. The router is deliberately small: it only *proposes* a set of agents;
validation and the planner decide what actually runs. LangGraph and the tool-using agents are
unchanged.

A **generative LLM router** (prompt a chat model to name the agents) is intentionally not
implemented: it would add a second, less constrained use of the generative gateway for a decision
that three yes/no probabilities can make. If you add one, implement `AgentRouter` and reuse
`validate_scores`-style allowlist validation on whatever it returns.

## 2. Interfaces (`src/repo_reviewer/routing/`)

```python
class AgentRouter(Protocol):
    name: str

    def route(self, request: RoutingRequest) -> RoutingDecision: ...  # must not raise


@dataclass(frozen=True)
class RoutingRequest:  # everything a router may see
    text: str | None  # the --request string (bounded to 1000 chars)
    candidates: tuple[AgentName, ...]  # applicable agents only
    facts: dict[str, int | bool]  # python_files, manifests, has_readme


class RoutingDecision(BaseModel):  # validated on construction
    router: str
    source: "cli" | "rules" | "decision_model" | "fallback"
    agents: tuple[AgentName, ...]  # non-empty, unique, members of the allowlist enum
    reason: str
    scores: dict[AgentName, float] | None
    fallback_reason: str | None
    latency_ms
    input_tokens
    output_tokens
    cost_usd  # when measurable


class DecisionBackend(Protocol):  # adapter to ONE inference API
    name: str

    def score(self, query: DecisionQuery) -> BackendOutput: ...  # raw, unvalidated
```

Generative inference (`gateway.py`) and decision inference (`routing/backends.py`) share no code
and no client class; they only happen to be reachable through the same LiteLLM Proxy.

CLI: `--router deterministic|decision`, `--request TEXT`, `--decision-backend mock|systemone`,
`--threshold`. `--router decision` requires `--request` and cannot be combined with `--agents`
(an explicit list already is the decision).

## 3. Multi-label routing

A request can need several agents ("check the README and the pinned dependencies"). A
single-select `choice` question cannot express that, so the router asks **one independent
yes/no question per candidate agent** and selects every agent whose probability is at or above the
threshold. In LiteLLM's System One format that is the `noul` question type ("probability the
answer is yes").

## 4. Validation and the conservative fallback

All checks are in `routing/decision.py` and are independent of the backend:

| Condition | Result |
|---|---|
| Backend unreachable / non-200 / timeout | fallback |
| Response is not JSON / has no `answers` mapping | fallback |
| An agent id outside the `AgentName` allowlist (`shell`, `Code`, ...) | fallback (`UnsupportedAgent`) |
| Scored agents differ from the candidate set (missing or extra) | fallback |
| A score that is not a number, a bool, NaN/inf, or outside [0, 1] | fallback |
| No agent reaches the threshold (ambiguous / low confidence) | fallback |
| Any other exception in the backend | fallback (reason names the exception type only) |

**Fallback = run every applicable agent.** This is conservative in the direction that matters: a
router failure can only cost extra time and tokens; it can never hide a review. The reason is
recorded in `RoutingDecision.fallback_reason`, printed in the report and in JSON output.
Independently of all this, the planner intersects any decision with the applicable set.

## 5. Thresholds are illustrative

`DEFAULT_SELECT_THRESHOLD = 0.5` is the natural boundary for an independent yes/no probability and
**nothing more**. It is not calibrated. To choose a value for a real backend:

```bash
uv run python scripts/compare_routers.py --backend systemone --sweep
```

The sweep prints precision, recall, exact-match and the "missed agent" rate for thresholds
0.1-0.9 over `evals/routing_dataset.jsonl`. Because a missed agent is worse than an extra one,
prefer the highest threshold that keeps `missed_agent_rate` near zero, then validate on data the
dataset's author did not write. The dataset is 36 requests labelled by one annotator, including
ambiguous and non-English requests (ambiguous requests are labelled "all agents", the intended
conservative outcome). It is a starting point for a real evaluation, not a benchmark.

## 6. Comparison script

`scripts/compare_routers.py` compares three routers on the dataset: an all-agents baseline, the
keyword rules, and the decision router. It reports precision / recall / F1, exact match, missed-agent
rate, average agents run (a proxy for downstream cost), fallback rate, latency (mean and p95), and
tokens / cost *where the backend reports them*. With `--backend mock`, results demonstrate the
harness only: the mock scorer and the keyword rules were written by the same author as the
dataset, so their scores are optimistic and say nothing about a real model. Latency of the mock
and the rules is microseconds; the only meaningful latency and cost figures come from a real
backend. The rule router and baseline report no tokens or cost because they make no model call.

## 7. Is the LiteLLM Decisions API appropriate?

What was checked (LiteLLM 1.106.3 installed from PyPI, its source read, a live proxy started):

- **Routes exist.** `POST /v1/systemone` (alias `/systemone`, System One format) and `POST
  /v1/decisions` (alias `/decisions`, OpenAI Decisions format) are implemented in
  `litellm/proxy/decisions_endpoints/endpoints.py`. The docs say 1.104.2 or later; the docs and the
  release notes disagree on which exact release introduced them, so pin the proxy version.
- **Question types** (source, `litellm/types/decisions.py`): `noul` (answer `{"type":"noul","noul":<p>}`),
  `choice` (single label with `probabilities`), `score`, `box`. `choice` is single-select, hence the
  per-agent `noul` design above. Up to 128 questions per request. No streaming.
- **A request in this repository's format is accepted by a live 1.106.3 proxy**: schema validation
  passed and the proxy forwarded the body to the upstream's `/v1/systemone`. Against a *fake*
  `hosted_vllm` upstream (a keyword script, not a model) the whole chain worked: CLI adapter ->
  proxy -> upstream -> parsed `noul` scores, token usage from `usage`.
- **Compatibility with the project's local model: no.** `reviewer-local` is an Ollama generative
  model (`ollama_chat/qwen3:8b`). Asking the proxy to use it for a decision returned HTTP 400
  `Unknown Decisions provider 'ollama_chat'`. In 1.106.3 the supported decision providers are
  `openai, openrouter, azure_ai, perplexity, typesafe, strands_decider, cloudflare, databricks,
  hosted_vllm`. The docs also list Laya and Bespoke Nimble (Ollama) as self-hosted options, but
  they are not in 1.106.3's provider list, so they were not usable with that version.
- **Self-hosted option that exists in 1.106.3:** `hosted_vllm/<model>` pointing at a vLLM server
  that implements `/v1/systemone` (the docs say vLLM >= 0.32 with Qwen3-family models). It was
  **not** run here: no vLLM runtime, GPU or model was available, so whether a given small model
  gives usable probabilities is unknown.
- **Other provider choices send data off-machine** (OpenAI, Perplexity, Cloudflare, Databricks,
  Azure, OpenRouter, TypeSafe's proprietary Jev). They are not used or required here.
- **Gateway features** (keys, budgets, spend, logs) need the database that this repository's
  config deliberately omits. The cost header `x-litellm-response-cost` is read if present; for
  self-hosted vLLM it is absent, so cost shows as unavailable.

Verdict: the Decisions API is a reasonable *transport* for a self-hosted decision model, because it
gives typed probabilities, one gateway and one auth key. It is **not** a drop-in for the Ollama
model this project uses for generation, and it is new enough that formats have moved between
versions (the docs note System One was served at `/v1/decisions` in `1.106.0-dev.1`). Hence
`SystemOneBackend` is labelled experimental, and no integration with a real decision model is
claimed.

## 8. Privacy

- Only `RoutingRequest.text` (the `--request` string) and three coarse facts (`python_files`,
  `manifests`, `has_readme` as numbers/booleans) can reach a router. File names and contents never
  do. A test asserts the adapter's payload contains neither.
- `DECISION_API_BASE` must point at a local host (`127.0.0.1`, `localhost`, `::1`, `*.localhost`,
  `host.docker.internal`) unless `DECISION_ALLOW_REMOTE=1` is set; hostnames are not resolved, so
  this is a guard against mistakes, not a network control.
- The decision endpoint is separate from `REVIEWER_GATEWAY_URL`, and choosing `--mode deepseek`
  does not route decisions to DeepSeek.
- Nothing in this repository depends on Jev or any proprietary API.

## 9. Adding a backend

Implement `DecisionBackend` for your API's wire format and return the raw scores:

```python
class MyBackend:
    name = "my-backend"

    def score(self, query: DecisionQuery) -> BackendOutput:
        # query.request_text, query.facts, query.agents  (id -> description)
        raw = call_my_api(...)  # any protocol: HTTP, gRPC, in-process
        return BackendOutput(scores={"code": 0.9, "dependencies": 0.1, "documentation": 0.2})
```

Raise `BackendError` for transport problems. Do not validate: `DecisionModelRouter` does that, so
every backend gets identical safety behaviour. Wire it up in `cli.build_router` and add contract
tests with `httpx.MockTransport` like `tests/test_routing.py` does.
