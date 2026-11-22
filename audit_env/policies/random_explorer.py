"""Random explorer with a fixed seed: random tool calls, then a few random guesses."""

from __future__ import annotations

import random

from audit_env.config import ERROR_TYPES
from audit_env.policies.base import Policy

TABLES = ("ledger", "invoices", "payments", "fx")


class RandomExplorerPolicy(Policy):
    name = "random"

    def reset(self, obs, info, seed, truth=None):
        super().reset(obs, info, seed, truth)
        # fixed policy seed + the public episode id; never the generation seed of the episode
        self.rng = random.Random(f"random-explorer:{seed}:{self.episode_id}")
        self.budget = self.rng.randint(3, 40)
        self.seen: list[tuple[str, str]] = []  # (table, id)
        self.docs: list[str] = []
        self.n = 0

    def _absorb(self, obs):
        res = obs.get("result")
        if not isinstance(res, dict):
            return
        rows = res.get("records") or [res]
        for r in rows:
            for key, table in (("row_id", "ledger"), ("invoice_id", "invoices"), ("payment_id", "payments")):
                if r.get(key) and (key != "invoice_id" or "total_minor" in r):
                    self.seen.append((table, r[key]))
                    break
            if r.get("doc_id"):
                self.docs.append(r["doc_id"])

    def act(self, obs):
        self._absorb(obs)
        self.n += 1
        if self.n > self.budget or obs.get("remaining_steps", 0) <= 1:
            return self._guess()
        choice = self.rng.random()
        if choice < 0.5 or not self.seen:
            table = self.rng.choice(TABLES)
            return {"name": "list_records", "args": {"table": table, "limit": self.rng.randint(1, 50),
                                                     "cursor": f"{table}:{self.rng.randint(0, 120)}"}}
        if choice < 0.75:
            table, rid = self.rng.choice(self.seen)
            return {"name": "get_record", "args": {"table": table, "id": rid}}
        if choice < 0.9 and self.docs:
            return {"name": "get_document", "args": {"doc_id": self.rng.choice(self.docs)}}
        return {"name": "calculator", "args": {"expression": f"{self.rng.randint(1, 999)} * 1.05"}}

    def _guess(self):
        findings = []
        ids = [i for t, i in self.seen if t in ("invoices", "payments")]
        for _ in range(self.rng.randint(0, 2) if ids else 0):
            findings.append({
                "type": self.rng.choice(ERROR_TYPES), "affected_ids": [self.rng.choice(ids)],
                "claimed_amount_minor": self.rng.randint(1, 10**6), "currency": self.manifest["base_currency"],
                "evidence_ids": self.rng.sample(self.docs, min(2, len(self.docs))), "explanation": "random guess",
            })
        return self.submit(findings)
