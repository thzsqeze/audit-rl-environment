"""Reconciliation baselines.

* `reconciler`    - the *simple* reconciliation algorithm required by the brief. It uses naive
                    heuristics: duplicates by (customer, currency, total) only, unapplied cash = any
                    referenced receipt with credits to 2200/2900, and UTC dates instead of the
                    company timezone. Distractors are designed to hurt exactly these shortcuts.
* `rules_auditor` - reads every table and applies the full public rules (audit_env.rules); it is the
                    reference solver showing every issue is derivable within the step budget.
"""

from __future__ import annotations

from collections import defaultdict

from audit_env.policies.reader import TableReader
from audit_env.rules import Books, detect_all, detect_cutoff, detect_fx


class RulesAuditorPolicy(TableReader):
    name = "rules_auditor"

    def act(self, obs):
        action = self.read_step(obs)
        if action is not None:
            return action
        public = {"manifest": self.manifest, "policy": self.policy, **self.data}
        try:
            findings = detect_all(public)
        except Exception:  # incomplete data under a tight budget: answer nothing rather than crash
            findings = []
        return self.submit(findings)


class ReconcilerPolicy(TableReader):
    name = "reconciler"

    def act(self, obs):
        action = self.read_step(obs)
        if action is not None:
            return action
        try:
            findings = self._naive()
        except Exception:
            findings = []
        return self.submit(findings)

    def _naive(self) -> list[dict]:
        manifest = dict(self.manifest, timezone="+00:00")  # naive: ignores the company timezone
        books = Books(manifest, self.policy, self.data["ledger"], self.data["invoices"], self.data["payments"],
                      self.data["fx"])
        findings = detect_fx(books) + detect_cutoff(books)
        booked = {r["invoice_id"] for r in books.ledger if r["account"] == "1100" and r["payment_id"] is None}
        groups = defaultdict(list)
        for inv in books.invoices:
            if inv["invoice_id"] in booked:
                groups[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(inv)
        for group in groups.values():
            for dup in group[1:]:
                findings.append(_f("duplicate_invoice", [group[0]["invoice_id"], dup["invoice_id"]],
                                   dup["total_minor"], dup["currency"], [group[0]["doc_id"], dup["doc_id"]]))
        parked = defaultdict(lambda: [0, []])
        for r in books.ledger:
            if r["payment_id"] and r["account"] in ("2200", "2900") and r["credit_minor"] > 0:
                parked[r["payment_id"]][0] += r["txn_amount_minor"]
                parked[r["payment_id"]][1].append(r["row_id"])
        for pid, (amount, rows) in parked.items():
            pay = books.pay.get(pid)
            if pay and pay["invoice_refs"]:
                findings.append(_f("payment_allocation", [pid, pay["invoice_refs"][0]["invoice_id"]], amount,
                                   pay["currency"], [pay["doc_id"], *rows]))
        return findings


def _f(type_, ids, amount, currency, evidence):
    return {"type": type_, "affected_ids": sorted(ids), "claimed_amount_minor": amount, "currency": currency,
            "evidence_ids": evidence[:10], "explanation": "naive reconciliation"}
