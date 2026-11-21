"""Audit rules over PUBLIC records only (manifest, policy, ledger, invoices, payments, fx).

Used for two purposes:
* independent check that every truth issue is derivable from accessible evidence and that clean
  books / distractors raise nothing (see validate.py);
* the "reconciler" baseline policy, which feeds it records collected through the tool interface.

Nothing here may import grader-side data.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

from audit_env.money import convert_minor, fx_id, local_date, parse_offset

AR, CASH, SUSPENSE, REVENUE = "1100", "1000", "2900", "4000"


class Books:
    """Indexed view of the public tables."""

    def __init__(self, manifest: dict, policy: dict, ledger: list[dict], invoices: list[dict],
                 payments: list[dict], fx: list[dict]):
        self.manifest, self.policy = manifest, policy
        self.ledger, self.invoices, self.payments = ledger, invoices, payments
        self.base = manifest["base_currency"]
        self.tz = parse_offset(manifest["timezone"])
        self.start = dt.date.fromisoformat(manifest["period_start"])
        self.end = dt.date.fromisoformat(manifest["period_end"])
        self.rounding = manifest.get("rounding", {}).get("mode", "half_up")
        self.inv = {i["invoice_id"]: i for i in invoices}
        self.pay = {p["payment_id"]: p for p in payments}
        self.rates: dict[str, list[tuple[dt.date, str]]] = defaultdict(list)
        for r in fx:
            if r["target_currency"] == self.base:
                self.rates[r["source_currency"]].append((dt.date.fromisoformat(r["date"]), r["rate"]))
        for v in self.rates.values():
            v.sort()

    def local(self, ts: str) -> dt.date:
        return local_date(ts, self.tz)

    def in_period(self, day: dt.date) -> bool:
        return self.start <= day <= self.end

    def rate(self, currency: str, day: dt.date) -> tuple[str, str | None]:
        """Policy rate and its fx evidence id (None for base currency)."""
        if currency == self.base:
            return "1", None
        best = None
        for d, r in self.rates.get(currency, []):
            if d <= day:
                best = (d, r)
            else:
                break
        if best is None:
            raise KeyError(f"no rate for {currency} on/before {day}")
        return best[1], fx_id(best[0], currency, self.base)

    def conv(self, amount: int, currency: str, rate: str) -> int:
        return convert_minor(amount, currency, self.base, rate, self.rounding)


def _finding(type_: str, affected: list[str], amount: int, currency: str, evidence: list[str], why: str) -> dict:
    return {"type": type_, "affected_ids": sorted(affected), "claimed_amount_minor": int(amount),
            "currency": currency, "evidence_ids": sorted(dict.fromkeys(e for e in evidence if e))[:10],
            "explanation": why[:1000]}


def _issuance_rows(books: Books) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in books.ledger:
        if r["invoice_id"] and r["payment_id"] is None and r["account"] in (AR, REVENUE):
            out[r["invoice_id"]].append(r)
    return out


def detect_duplicates(books: Books) -> list[dict]:
    issued = _issuance_rows(books)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for inv in books.invoices:
        if not any(r["account"] == AR for r in issued.get(inv["invoice_id"], [])):
            continue
        lines = tuple((li["description"], li["quantity"], li["unit_price_minor"], li["amount_minor"])
                      for li in inv["line_items"])
        groups[(inv["entity_id"], inv["currency"], inv["total_minor"], lines)].append(inv)
    out = []
    for group in groups.values():
        group.sort(key=lambda i: i["issued_at"])
        for dup in group[1:]:
            orig = group[0]
            ev = [orig["doc_id"], dup["doc_id"]]
            ev += [r["row_id"] for i in (orig, dup) for r in issued[i["invoice_id"]] if r["account"] == AR]
            out.append(_finding("duplicate_invoice", [orig["invoice_id"], dup["invoice_id"]], dup["total_minor"],
                                dup["currency"], ev, "Same customer, currency, line items and total booked twice."))
    return out


def detect_allocation(books: Books) -> list[dict]:
    applied: dict[tuple[str, str], int] = defaultdict(int)
    rows_by_pay: dict[str, list[dict]] = defaultdict(list)
    for r in books.ledger:
        if r["payment_id"]:
            rows_by_pay[r["payment_id"]].append(r)
            if r["account"] == AR and r["credit_minor"] > 0 and r["invoice_id"]:
                applied[(r["payment_id"], r["invoice_id"])] += r["txn_amount_minor"]
    out = []
    for pay in books.payments:
        if pay["direction"] != "in" or not pay["invoice_refs"]:
            continue
        pid = pay["payment_id"]
        for ref in pay["invoice_refs"]:
            iid = ref["invoice_id"]
            inv = books.inv.get(iid)
            if inv is None:
                continue
            by_others = sum(v for (p, i), v in applied.items() if i == iid and p != pid)
            expected = min(ref["amount_minor"], max(inv["total_minor"] - by_others, 0))
            shortfall = expected - applied[(pid, iid)]
            if shortfall > 0:
                ev = [pay["doc_id"]] + [r["row_id"] for r in rows_by_pay[pid]
                                        if r["account"] == SUSPENSE or
                                        (r["account"] == AR and r["invoice_id"] != iid)]
                out.append(_finding("payment_allocation", [pid, iid], shortfall, pay["currency"], ev,
                                    "Remittance allocates the receipt to an open invoice but the ledger "
                                    "did not apply it in full."))
    return out


def detect_fx(books: Books) -> list[dict]:
    tol = int(books.policy.get("amount_tolerance_minor", 1))
    out = []
    for r in books.ledger:
        if r["txn_currency"] == books.base:
            continue
        if r["account"] == AR and r["debit_minor"] > 0 and r["invoice_id"] and r["payment_id"] is None:
            doc = books.inv.get(r["invoice_id"])
            if doc is None:
                continue
            day, amount, target, booked = books.local(doc["issued_at"]), doc["total_minor"], r["invoice_id"], \
                r["debit_minor"]
        elif r["account"] == CASH and r["debit_minor"] > 0 and r["payment_id"]:
            doc = books.pay.get(r["payment_id"])
            if doc is None:
                continue
            day, amount, target, booked = books.local(doc["timestamp"]), doc["amount_minor"], r["payment_id"], \
                r["debit_minor"]
        else:
            continue
        try:
            rate, fx_ev = books.rate(r["txn_currency"], day)
        except KeyError:
            continue
        correct = books.conv(amount, r["txn_currency"], rate)
        if abs(booked - correct) > tol:
            out.append(_finding("fx_conversion", [target], abs(booked - correct), books.base,
                                [r["row_id"], doc["doc_id"], fx_ev],
                                f"Booked {booked} vs {correct} at policy rate {rate} for local date {day}."))
    return out


def detect_cutoff(books: Books) -> list[dict]:
    out = []
    for iid, rows in sorted(_issuance_rows(books).items()):
        inv = books.inv.get(iid)
        if inv is None:
            continue
        day = books.local(inv["issued_at"])
        if not books.in_period(day):
            ar = [r["row_id"] for r in rows if r["account"] == AR]
            out.append(_finding("period_cutoff", [iid], inv["total_minor"], inv["currency"], [inv["doc_id"]] + ar,
                                f"Invoice dated {day} (local) is outside the period but booked in it."))
    return out


DETECTORS = (detect_duplicates, detect_allocation, detect_fx, detect_cutoff)


def detect_all(public: dict[str, Any]) -> list[dict]:
    books = Books(public["manifest"], public["policy"], public["ledger"], public["invoices"],
                  public["payments"], public["fx"])
    findings: list[dict] = []
    for det in DETECTORS:
        findings += det(books)
    findings.sort(key=lambda f: (f["type"], f["affected_ids"]))
    return findings
