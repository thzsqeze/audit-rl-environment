"""Attack baseline: read documents and flag every invoice/payment under every issue type (30 max)."""

from __future__ import annotations

from audit_env.config import ERROR_TYPES
from audit_env.policies.reader import TableReader


class FlagAllPolicy(TableReader):
    name = "flag_all"
    tables = ("invoices", "payments")

    def act(self, obs):
        action = self.read_step(obs)
        if action is not None:
            return action
        findings = []
        docs = [(i["invoice_id"], i) for i in self.data["invoices"]] + \
               [(p["payment_id"], p) for p in self.data["payments"]]
        for t in ERROR_TYPES:
            for rid, doc in docs:
                findings.append({
                    "type": t, "affected_ids": [rid],
                    "claimed_amount_minor": doc.get("total_minor", doc.get("amount_minor", 0)),
                    "currency": doc["currency"], "evidence_ids": [doc["doc_id"]], "explanation": "flag everything",
                })
        # interleave types so the 30-finding cap covers all of them
        findings.sort(key=lambda f: (f["affected_ids"][0], f["type"]))
        return self.submit(findings[:30])
