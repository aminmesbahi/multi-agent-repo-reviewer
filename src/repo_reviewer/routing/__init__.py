"""Agent routing: deterministic rules (default) or an optional decision model."""

from repo_reviewer.routing.base import AgentRouter, RoutingRequest
from repo_reviewer.routing.decision import DecisionModelRouter
from repo_reviewer.routing.rules import RuleBasedRouter

__all__ = ["AgentRouter", "DecisionModelRouter", "RuleBasedRouter", "RoutingRequest"]
