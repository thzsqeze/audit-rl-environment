"""Evaluator-only oracle: submits the closed truth with its first evidence set (checks max reward)."""

from __future__ import annotations

from audit_env.policies.base import Policy


class OraclePolicy(Policy):
    name = "oracle"
    needs_truth = True

    def reset(self, obs, info, seed, truth=None):
        super().reset(obs, info, seed, truth)
        if truth is None:
            raise ValueError("oracle requires evaluator-side truth")
        self.findings = [{
            "type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
            "currency": t["currency"], "evidence_ids": t["evidence_sets"][0], "explanation": "oracle",
        } for t in truth["issues"]]

    def act(self, obs):
        return self.submit(self.findings)
