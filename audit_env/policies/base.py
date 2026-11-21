"""Policy adapter interface: policies only see observations (dicts) and emit JSON actions."""

from __future__ import annotations

from typing import Any


class Policy:
    name = "base"
    needs_truth = False  # True only for the evaluator-side oracle

    def reset(self, obs: dict[str, Any], info: dict[str, Any], seed: int, truth: dict | None = None) -> None:
        self.episode_id = info["episode_id"]
        self.manifest = obs["manifest"]
        self.policy = obs["policy"]

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def submit(self, findings: list[dict]) -> dict[str, Any]:
        return {"name": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": findings[:30]}}}
