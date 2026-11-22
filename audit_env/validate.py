"""Self-checks of generated episodes (run by tests and `audit-env check`).

1. Accounting identities hold on the clean twin (books before injection) and every journal still
   balances after injection.
2. Clean twin: every ledger row is in the period and the public rules find nothing, i.e. the
   distractors are legitimately explainable from accessible records.
3. Injected books: the public rules re-derive exactly the declared truth (type, ids, amount, currency)
   and their evidence covers one of the declared evidence sets; every evidence id exists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from audit_env.money import parse_ts
from audit_env.rules import Books, detect_all


def accounting_errors(public: dict[str, Any], *, require_in_period: bool) -> list[str]:
    errors: list[str] = []
    m = public["manifest"]
    books = Books(m, public["policy"], public["ledger"], public["invoices"], public["payments"], public["fx"])
    journals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in public["ledger"]:
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0) or min(r["debit_minor"], r["credit_minor"]) < 0:
            errors.append(f"row {r['row_id']}: exactly one positive side required")
        if r["currency"] != m["base_currency"]:
            errors.append(f"row {r['row_id']}: ledger currency must be base")
        if r["invoice_id"] is not None and r["invoice_id"] not in books.inv:
            errors.append(f"row {r['row_id']}: dangling invoice_id")
        if r["payment_id"] is not None and r["payment_id"] not in books.pay:
            errors.append(f"row {r['row_id']}: dangling payment_id")
        parse_ts(r["timestamp"])
        if require_in_period and not books.in_period(books.local(r["timestamp"])):
            errors.append(f"row {r['row_id']}: outside period in clean books")
        journals[r["journal_id"]][0] += r["debit_minor"]
        journals[r["journal_id"]][1] += r["credit_minor"]
    for jid, (dr, cr) in journals.items():
        if dr != cr:
            errors.append(f"journal {jid}: debits {dr} != credits {cr}")
    for inv in public["invoices"]:
        parse_ts(inv["issued_at"])
        if sum(li["amount_minor"] for li in inv["line_items"]) != inv["total_minor"]:
            errors.append(f"invoice {inv['invoice_id']}: total != sum(lines)")
        for li in inv["line_items"]:
            if li["quantity"] * li["unit_price_minor"] != li["amount_minor"]:
                errors.append(f"invoice {inv['invoice_id']}: qty*price != amount")
    booked_payments = {r["payment_id"] for r in public["ledger"] if r["payment_id"]}
    for pay in public["payments"]:
        parse_ts(pay["timestamp"])
        # a receipt of the next period may be published (like next-period invoices) but never booked in clean books
        if require_in_period and not books.in_period(books.local(pay["timestamp"])) and \
                pay["payment_id"] in booked_payments:
            errors.append(f"payment {pay['payment_id']}: outside period but booked in clean books")
        if pay["invoice_refs"] is not None:
            if sum(x["amount_minor"] for x in pay["invoice_refs"]) != pay["amount_minor"]:
                errors.append(f"payment {pay['payment_id']}: refs != amount")
            for x in pay["invoice_refs"]:
                if x["invoice_id"] not in books.inv:
                    errors.append(f"payment {pay['payment_id']}: dangling ref")
    if require_in_period:
        # every fully remitted period invoice nets to zero in AR
        ar: dict[str, int] = defaultdict(int)
        for r in public["ledger"]:
            if r["account"] == "1100" and r["invoice_id"]:
                ar[r["invoice_id"]] += r["debit_minor"] - r["credit_minor"]
        remitted: dict[str, int] = defaultdict(int)
        for pay in public["payments"]:
            if not books.in_period(books.local(pay["timestamp"])):
                continue  # next-period receipts do not settle anything in this period
            for x in pay["invoice_refs"] or []:
                remitted[x["invoice_id"]] += x["amount_minor"]
        issued = {r["invoice_id"] for r in public["ledger"]
                  if r["account"] == "1100" and r["payment_id"] is None and r["invoice_id"]}
        for iid in issued:
            if remitted[iid] >= books.inv[iid]["total_minor"] and ar[iid] != 0:
                errors.append(f"invoice {iid}: fully paid but AR balance {ar[iid]}")
    return errors


def evidence_index(public: dict[str, Any]) -> set[str]:
    ids = {r["row_id"] for r in public["ledger"]}
    ids |= {d["doc_id"] for d in public["invoices"]} | {d["doc_id"] for d in public["payments"]}
    ids |= {f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in public["fx"]}
    return ids


def truth_errors(public: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    known = evidence_index(public)
    found = {(f["type"], tuple(sorted(f["affected_ids"]))): f for f in detect_all(public)}
    declared = {(t["type"], tuple(sorted(t["affected_ids"]))): t for t in truth["issues"]}
    for key in found.keys() - declared.keys():
        errors.append(f"undeclared discrepancy derivable from records: {key}")
    for key, t in declared.items():
        for ev in t["evidence_sets"]:
            missing = [e for e in ev if e not in known]
            if missing:
                errors.append(f"{t['issue_id']}: evidence ids not accessible: {missing}")
        f = found.get(key)
        if f is None:
            errors.append(f"{t['issue_id']}: declared discrepancy not derivable from records")
            continue
        if f["currency"] != t["currency"] or abs(f["claimed_amount_minor"] - t["expected_amount_minor"]) > \
                truth.get("amount_tolerance_minor", 0):
            errors.append(f"{t['issue_id']}: derived amount {f['claimed_amount_minor']} {f['currency']} != "
                          f"declared {t['expected_amount_minor']} {t['currency']}")
        if not any(set(ev) <= set(f["evidence_ids"]) for ev in t["evidence_sets"]):
            errors.append(f"{t['issue_id']}: derived evidence does not cover any declared evidence set")
    return errors


def validate_episode(episode) -> list[str]:
    errors = [f"clean: {e}" for e in accounting_errors(episode.clean_twin, require_in_period=True)]
    errors += [f"clean: rules raised {f['type']} {f['affected_ids']}" for f in detect_all(episode.clean_twin)]
    errors += [f"injected: {e}" for e in accounting_errors(episode.public(), require_in_period=False)]
    errors += [f"truth: {e}" for e in truth_errors(episode.public(), episode.truth)]
    return errors
