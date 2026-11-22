"""Episode assembly: hidden history -> injected errors -> renamed public records + closed grader files."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audit_env.config import ERROR_TYPES, SCENARIOS, GeneratorConfig, load_config
from audit_env.generator.errors import INJECTORS, SCENARIO_BUILDERS, VARIANTS_ATTR, InjectionSkipped, Issue
from audit_env.generator.history import build_history, plan_bookings
from audit_env.generator.world import CHART_OF_ACCOUNTS, ROUNDING, World
from audit_env.money import CURRENCY_EXPONENT, fx_id, iso
from audit_env.rng import make_rng, stable_hash
from audit_env.version import ENV_VERSION, SCHEMA_VERSION

LEDGER_COLUMNS = [
    "row_id", "journal_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
    "entity_id", "invoice_id", "payment_id", "txn_currency", "txn_amount_minor", "fx_rate",
]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
AMOUNT_TOLERANCE_MINOR = 0  # exact integer comparison (judge default tau=0); expected amounts are exact
LEDGER_INT_COLUMNS = ("debit_minor", "credit_minor", "txn_amount_minor")
TABLES = ("ledger", "invoices", "payments", "fx")


@dataclass
class Episode:
    manifest: dict[str, Any]
    policy: dict[str, Any]
    ledger: list[dict[str, Any]]
    invoices: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    fx: list[dict[str, Any]]
    truth: dict[str, Any]  # closed
    provenance: dict[str, Any]  # closed
    clean_twin: dict[str, Any] | None = field(default=None, repr=False)  # closed: books before injection

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    def public(self) -> dict[str, Any]:
        return {"manifest": self.manifest, "policy": self.policy, "ledger": self.ledger,
                "invoices": self.invoices, "payments": self.payments, "fx": self.fx}

    def write(self, directory: str | Path, include_closed: bool = True) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "manifest.json", self.manifest)
        _write_json(out / "policy.json", self.policy)
        (out / "ledger.csv").write_text(ledger_to_csv(self.ledger), encoding="utf-8")
        (out / "fx.csv").write_text(_rows_to_csv(self.fx, FX_COLUMNS), encoding="utf-8")
        for name, rows in (("invoices", self.invoices), ("payments", self.payments)):
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
        if include_closed:  # closed grader files: never exposed to the policy process
            _write_json(out / "truth.json", self.truth)
            _write_json(out / "provenance.json", self.provenance)
        return out

    @classmethod
    def load(cls, directory: str | Path) -> Episode:
        """Load an episode directory exactly as written (or as transformed by the jury): same ids, same order."""
        src = Path(directory)

        def read_json(name: str) -> Any:
            with open(src / name, encoding="utf-8") as fh:
                return json.load(fh)

        def read_jsonl(name: str) -> list[dict]:
            with open(src / name, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]

        with open(src / "ledger.csv", encoding="utf-8", newline="") as fh:
            ledger = []
            for row in csv.DictReader(fh):
                rec: dict[str, Any] = {k: (None if v == "null" else v) for k, v in row.items()}
                for k in LEDGER_INT_COLUMNS:
                    rec[k] = int(rec[k])
                ledger.append(rec)
        with open(src / "fx.csv", encoding="utf-8", newline="") as fh:
            fx = [dict(r) for r in csv.DictReader(fh)]
        provenance = read_json("provenance.json") if (src / "provenance.json").exists() else {}
        return cls(manifest=read_json("manifest.json"), policy=read_json("policy.json"), ledger=ledger,
                   invoices=read_jsonl("invoices.jsonl"), payments=read_jsonl("payments.jsonl"), fx=fx,
                   truth=read_json("truth.json"), provenance=provenance)

    def evidence_ids(self) -> set[str]:
        """Every id the policy can reach through the tools (valid evidence_ids)."""
        ids = {r["row_id"] for r in self.ledger}
        ids |= {d["doc_id"] for d in self.invoices} | {d["doc_id"] for d in self.payments}
        ids |= {d["invoice_id"] for d in self.invoices} | {d["payment_id"] for d in self.payments}
        ids |= {fx_id(r["date"], r["source_currency"], r["target_currency"]) for r in self.fx}
        return ids | {"manifest", "policy"}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("null" if row[k] is None else row[k]) for k in columns})
    return buf.getvalue()


def ledger_to_csv(rows: list[dict]) -> str:
    return _rows_to_csv(rows, LEDGER_COLUMNS)


# ------------------------------------------------------------------------------------------ injection

def _inject(world: World, cfg: GeneratorConfig) -> tuple[list[Issue], list[dict]]:
    rng = world.rng
    log: list[dict] = []
    if rng.random() < cfg.clean_probability:
        log.append({"step": "clean_company", "detail": "no issues injected"})
        return [], log
    plan: list[str] = []
    for name in SCENARIOS:
        lo, hi = cfg.scenario_counts.get(name, [0, 0])
        plan += [f"scenario:{name}"] * rng.randint(lo, hi)
    for name in ERROR_TYPES:
        lo, hi = cfg.error_counts.get(name, [0, 0])
        plan += [f"type:{name}"] * rng.randint(lo, hi)
    issues: list[Issue] = []
    for item in plan:
        kind, name = item.split(":", 1)
        size = 2 if kind == "scenario" else 1
        if len(issues) + size > cfg.max_issues:
            log.append({"step": "skipped", "item": item, "reason": "max_issues"})
            continue
        try:
            if kind == "scenario":
                new = SCENARIO_BUILDERS[name](world)
            else:
                attr = VARIANTS_ATTR[name]
                variant = rng.choice(getattr(cfg, attr)) if attr else None
                new = INJECTORS[name](world, variant)
        except InjectionSkipped as exc:
            log.append({"step": "skipped", "item": item, "reason": f"no suitable target ({exc})"})
            continue
        issues += new
        log.append({"step": "injected", "item": item, "issues": len(new)})
    return issues, log


def _clean_twin(world: World) -> World:
    """Same documents, minus injected duplicates, booked by the clean booking rules."""
    twin = world.snapshot()
    twin.invoices = {k: v for k, v in twin.invoices.items() if "injected_duplicate" not in v.tags}
    plan_bookings(twin)
    return twin


# ------------------------------------------------------------------------------------------ rendering

_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in "O0I1")


class _Renamer:
    PREFIX = {"E": "CUS-", "I": "INV-", "P": "PMT-", "D": "DOC-", "R": "GL-", "J": "JE-"}

    def __init__(self, rng):
        self.rng = rng
        self.map: dict[str, str] = {}
        self.used: set[str] = set()

    def __call__(self, key: str | None) -> str | None:
        if key is None:
            return None
        if key not in self.map:
            while True:
                length = 8 if key[0] == "R" else 6
                token = self.PREFIX[key[0]] + "".join(self.rng.choice(_ALPHABET) for _ in range(length))
                if token not in self.used:
                    break
            self.used.add(token)
            self.map[key] = token
        return self.map[key]


def _render(world: World, rename: _Renamer, shuffle_rng=None) -> tuple[dict, dict[str, dict], dict[tuple, str]]:
    """Return public tables, journal row lookup and evidence resolution map."""
    journals = sorted(world.journals().values(), key=lambda j: (j.ts, j.key))
    ledger, row_lookup = [], {}
    n = 0
    for j_index, journal in enumerate(journals):
        jid = rename(f"J{j_index + 1:05d}")
        for row in journal.rows:
            n += 1
            rid = rename(f"R{n:06d}")
            row_lookup[(journal.key, row.role, row.inv)] = rid
            row_lookup.setdefault((journal.key, row.role, None), rid)
            ledger.append({
                "row_id": rid, "journal_id": jid, "timestamp": iso(journal.ts.astimezone(dt.UTC)),
                "account": row.account, "debit_minor": row.debit, "credit_minor": row.credit,
                "currency": world.base, "entity_id": rename(row.entity), "invoice_id": rename(row.inv),
                "payment_id": rename(row.pay), "txn_currency": row.txn_currency,
                "txn_amount_minor": row.txn_amount, "fx_rate": row.fx_rate,
            })
    ledger.sort(key=lambda r: (r["timestamp"], r["journal_id"], r["row_id"]))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(ledger)

    invoices = []
    for inv in world.invoices.values():
        invoices.append({
            "doc_id": rename(inv.doc), "invoice_id": rename(inv.key), "entity_id": rename(inv.entity),
            "issued_at": iso(inv.issued_at), "due_at": iso(inv.due_at), "currency": inv.currency,
            "total_minor": inv.total, "line_items": inv.line_items,
        })
    payments = []
    for pay in world.payments.values():
        # out-of-period receipts are published only when they are part of the history (next_period_receipt)
        if not world.in_period(world.local(pay.ts)) and "next_period_receipt" not in pay.tags:
            continue
        payments.append({
            "doc_id": rename(pay.doc), "payment_id": rename(pay.key), "entity_id": rename(pay.entity),
            "timestamp": iso(pay.ts), "currency": pay.currency, "amount_minor": pay.amount,
            "direction": pay.direction,
            "invoice_refs": None if pay.refs is None else
            [{"invoice_id": rename(k), "amount_minor": a} for k, a in pay.refs],
            "refund_of": rename(pay.refund_of),
        })
    invoices.sort(key=lambda r: r["doc_id"])
    payments.sort(key=lambda r: r["doc_id"])
    fx_rows = [{"date": d.isoformat(), "source_currency": c, "target_currency": world.base, "rate": r}
               for (d, c), r in sorted(world.fx.items())]
    return {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx_rows}, row_lookup, {}


def _public_row(row: dict, rename: _Renamer) -> dict:
    return {"role": row["role"], "account": row["account"], "debit_minor": row["debit"],
            "credit_minor": row["credit"], "entity_id": rename(row["entity"]), "invoice_id": rename(row["inv"]),
            "payment_id": rename(row["pay"]), "txn_currency": row["txn_currency"],
            "txn_amount_minor": row["txn_amount"], "fx_rate": row["fx_rate"]}


def _public_journal_key(key: str, rename: _Renamer) -> str:
    kind, internal = key.split(":", 1)
    return f"{'invoice' if kind == 'inv' else 'payment'}:{rename(internal)}"


def _public_mutation(mutation: dict, rename: _Renamer) -> dict:
    """Causal change with public ids: bookings before/after the injection, plus scalar details."""
    out: dict[str, Any] = {}
    for key, value in mutation.items():
        if key in ("journals_before", "journals_after"):
            out[key] = {_public_journal_key(k, rename): (None if rows is None else
                                                         [_public_row(r, rename) for r in rows])
                        for k, rows in value.items()}
        elif key == "duplicate_journal":
            out[key] = [_public_row(r, rename) for r in value]
        elif key in ("copied_from", "new_document"):
            out[key] = rename(value)
        else:
            out[key] = value
    return out


def _resolve_evidence(world: World, rename: _Renamer, row_lookup: dict, ref: tuple) -> str:
    if ref[0] == "doc":
        obj = world.invoices.get(ref[1]) or world.payments[ref[1]]
        return rename(obj.doc)
    if ref[0] == "row":
        return row_lookup[(ref[1], ref[2], ref[3])]
    if ref[0] == "fx":
        return fx_id(ref[2], ref[1], world.base)
    raise ValueError(ref)


def _policy(world: World) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "period": {
            "start": world.period_start.isoformat(), "end": world.period_end.isoformat(), "inclusive": True,
            "basis": "Business dates are local dates in manifest.timezone. A document belongs to the period "
                     "iff the local date of its own timestamp (invoice issued_at, payment timestamp) is within "
                     "[start, end]. Only period documents may be booked in this period's ledger.",
        },
        "fx": {
            "quote": "fx.csv rate = units of target_currency per 1 unit of source_currency",
            "rate_date": "local date of the source document (invoice issued_at / payment timestamp)",
            "fallback": "if no rate is published on that date, use the latest rate dated before it",
            "invoices": "AR and revenue are booked at the invoice-date rate",
            "receipts": "cash is booked at the payment-date rate; AR is credited at the invoice-date rate; "
                        "differences go to account 7900",
        },
        "rounding": {"mode": ROUNDING, "unit": "minor",
                     "formula": "round(amount_minor * rate * 10**(exp_target - exp_source))"},
        "receipts": {
            "application": "A receipt is applied to the invoices in its invoice_refs, each up to that "
                           "invoice's open balance. Excess over open balances goes to 2200 and may be refunded. "
                           "Receipts with invoice_refs=null cannot be identified and stay in 2900.",
        },
        "accounts": CHART_OF_ACCOUNTS,
        "null_convention": "missing references are explicit null (JSON null / 'null' in CSV)",
        "finding_conventions": {
            "duplicate_invoice": {
                "meaning": "the same invoice (same customer, currency, line items and total) issued and "
                           "booked twice",
                "affected_ids": "both invoice_ids", "amount": "invoice total in invoice currency"},
            "payment_allocation": {
                "meaning": "a receipt's remittance (invoice_refs) was not applied, or only partly applied, to an "
                           "invoice that still had an open balance",
                "affected_ids": "[payment_id, invoice_id]",
                "amount": "remitted amount that should have been applied to that invoice but was not, in "
                          "payment currency"},
            "fx_conversion": {
                "meaning": "a foreign-currency document was converted with a rate other than the policy rate",
                "affected_ids": "[invoice_id] for invoice postings, [payment_id] for receipt postings",
                "amount": "absolute difference between booked and correct base amount, in base currency"},
            "period_cutoff": {
                "meaning": "an invoice or a receipt dated outside the period was booked in this period",
                "affected_ids": "[invoice_id] for an invoice, [payment_id] for a receipt",
                "amount": "invoice total / receipt amount, in the document currency"},
        },
        "evidence_ids": "ledger row_id, document doc_id or fx id 'FX:<date>:<source>:<target>'",
        "amount_tolerance_minor": AMOUNT_TOLERANCE_MINOR,
    }


def generate_episode(seed: int, config: Any = None) -> Episode:
    cfg = load_config(config)
    fingerprint = cfg.fingerprint()
    world = World(cfg, make_rng("history", seed, fingerprint))
    build_history(world)
    issues, log = _inject(world, cfg)
    twin = _clean_twin(world)

    episode_id = "ep-" + stable_hash([seed, fingerprint, cfg.id_salt, ENV_VERSION], 10)
    rename = _Renamer(make_rng("ids", seed, fingerprint, cfg.id_salt))
    shuffle_rng = make_rng("order", seed, fingerprint, cfg.id_salt) if cfg.shuffle_rows else None
    tables, row_lookup, _ = _render(world, rename, shuffle_rng)

    truth_issues = []
    for issue in issues:
        affected = sorted(rename(k) for k in issue.affected)
        evidence = [sorted(_resolve_evidence(world, rename, row_lookup, ref) for ref in ev) for ev in issue.evidence]
        truth_issues.append({
            "type": issue.type, "affected_ids": affected, "expected_amount_minor": issue.amount,
            "currency": issue.currency, "evidence_sets": evidence, "variant": issue.variant,
            "scenario": issue.scenario, "_mutation": issue.mutation,
        })
    truth_issues.sort(key=lambda t: (t["type"], t["affected_ids"]))
    provenance_issues = []
    for i, t in enumerate(truth_issues):
        t["issue_id"] = f"ISS-{i + 1:02d}"
        provenance_issues.append({"issue_id": t["issue_id"], "variant": t["variant"], "scenario": t["scenario"],
                                  "causal_change": _public_mutation(t.pop("_mutation"), rename)})

    injected_keys = {k for issue in issues for k in issue.affected}
    currencies = sorted({world.base, *world.foreign})
    manifest = {
        "schema_version": SCHEMA_VERSION, "env_version": ENV_VERSION, "episode_id": episode_id,
        "base_currency": world.base, "period_start": world.period_start.isoformat(),
        "period_end": world.period_end.isoformat(), "timezone": world.tz_name,
        "record_counts": {"ledger": len(tables["ledger"]), "invoices": len(tables["invoices"]),
                          "payments": len(tables["payments"]), "fx": len(tables["fx"]),
                          "documents": len(tables["invoices"]) + len(tables["payments"])},
        "currency_exponents": {c: CURRENCY_EXPONENT[c] for c in currencies},
        "rounding": {"mode": ROUNDING, "unit": "minor"},
        "formats": {"timestamps": "ISO-8601 with UTC offset", "money": "integer minor units + currency code",
                    "fx_rate": "decimal string"},
    }
    truth = {"episode_id": episode_id, "amount_tolerance_minor": AMOUNT_TOLERANCE_MINOR, "issues": truth_issues}
    provenance = {
        "episode_id": episode_id, "seed": seed, "config_id": cfg.config_id, "config_fingerprint": fingerprint,
        "config": cfg.to_dict(), "env_version": ENV_VERSION, "generation_log": log, "issues": provenance_issues,
        # correct-but-unusual documents only: a candidate that an injector turned into an error is not a distractor
        "distractors": sorted({f"{t}:{rename(k)}" for k, o in {**world.invoices, **world.payments}.items()
                               for t in o.tags if t != "injected_duplicate" and k not in injected_keys}),
    }
    # clean twin: same documents/ids, fresh row/journal ids (they are different rows)
    twin_rename = _Renamer(make_rng("twin-ids", seed, fingerprint, cfg.id_salt))
    twin_rename.map = {k: v for k, v in rename.map.items() if k[0] in "EIPD"}
    twin_rename.used = set(rename.used)
    twin_tables, _, _ = _render(twin, twin_rename)
    clean_twin = {"manifest": manifest, "policy": _policy(world), **twin_tables}
    provenance["clean_books"] = {
        "description": "Ledger of the same documents booked by the clean rules before any injection "
                       "(row/journal ids are distinct from the published ledger).",
        "ledger": twin_tables["ledger"],
    }
    return Episode(manifest=manifest, policy=_policy(world), clean_twin=clean_twin, truth=truth,
                   provenance=provenance, **tables)
