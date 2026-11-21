"""Benign transformations of an episode on files, as the jury performs them for paired checks.

* permute: shuffle the record order of ledger / invoices / payments / fx (never line_items);
* rename:  replace every surrogate id (row_id, journal_id, doc_id, invoice_id, payment_id, entity_id)
           by a fresh one, consistently in all records, references, truth and provenance.
FX keys (FX:<date>:<source>:<target>) are natural keys and are never renamed.
"""

from __future__ import annotations

import copy
import random
import string
from typing import Any

from audit_env.generator import Episode

ID_FIELDS = {
    "ledger": ("row_id", "journal_id", "entity_id", "invoice_id", "payment_id"),
    "invoices": ("doc_id", "invoice_id", "entity_id"),
    "payments": ("doc_id", "payment_id", "entity_id", "refund_of"),
}


def _collect_ids(ep: Episode) -> list[str]:
    ids: set[str] = set()
    for table, fields in ID_FIELDS.items():
        for rec in getattr(ep, table):
            ids.update(rec[f] for f in fields if rec.get(f) is not None)
    for pay in ep.payments:
        for ref in pay.get("invoice_refs") or []:
            ids.add(ref["invoice_id"])
    return sorted(ids)


def _replace(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, str):
        if obj in mapping:
            return mapping[obj]
        if ":" in obj:  # journal keys in provenance, e.g. "invoice:INV-XXXX"
            head, _, tail = obj.partition(":")
            if tail in mapping:
                return f"{head}:{mapping[tail]}"
        return obj
    if isinstance(obj, list):
        return [_replace(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {_replace(k, mapping) if isinstance(k, str) and k in mapping else k: _replace(v, mapping)
                for k, v in obj.items()}
    return obj


def transform(ep: Episode, seed: int, *, permute: bool = True, rename: bool = True) -> tuple[Episode, dict]:
    rng = random.Random(f"transform:{seed}")
    mapping: dict[str, str] = {}
    if rename:
        alphabet = string.ascii_lowercase + string.digits
        used: set[str] = set()
        for old in _collect_ids(ep):
            while True:
                new = f"{old.split('-')[0].lower()}_{''.join(rng.choice(alphabet) for _ in range(10))}"
                if new not in used:
                    break
            used.add(new)
            mapping[old] = new
    out = Episode(
        manifest=copy.deepcopy(ep.manifest), policy=copy.deepcopy(ep.policy),
        ledger=_replace(ep.ledger, mapping), invoices=_replace(ep.invoices, mapping),
        payments=_replace(ep.payments, mapping), fx=copy.deepcopy(ep.fx),
        truth=_replace(ep.truth, mapping), provenance=_replace(ep.provenance, mapping),
    )
    if permute:
        for table in ("ledger", "invoices", "payments", "fx"):
            rng.shuffle(getattr(out, table))
    return out, mapping
