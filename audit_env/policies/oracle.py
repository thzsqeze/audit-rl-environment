"""Evaluator-only oracle: submits correct findings that the evaluator built from the closed truth.

As in the jury protocol, the oracle never reads truth itself: the evaluator (runner.oracle_findings) passes the
findings in (`load_findings`) and the oracle submits them with the ordinary submit action. It checks the
maximum reward, not an agent's ability.
"""

from __future__ import annotations

from audit_env.policies.base import Policy


class OraclePolicy(Policy):
    name = "oracle"
    needs_truth = True  # the evaluator supplies findings derived from the truth

    def reset(self, obs, info, seed, truth=None):
        super().reset(obs, info, seed, truth)
        self.findings = None

    def load_findings(self, findings):
        if not isinstance(findings, list):
            raise ValueError("oracle requires evaluator-side findings")
        self.findings = findings

    def act(self, obs):
        if self.findings is None:
            raise ValueError("oracle requires evaluator-side findings")
        return self.submit(self.findings)
