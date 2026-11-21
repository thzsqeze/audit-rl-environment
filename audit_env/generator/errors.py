"""Error injectors. Each injector changes the hidden history in a causally recorded way and returns
truth issues expressed with *internal* keys; ids are renamed later in episode assembly.

Evidence references are symbolic until assembly:
  ("doc", <invoice/payment key>)            -> doc_id of that document
  ("row", <journal key>, <role>, <inv key>) -> row_id of the ledger row with that role (and invoice)
  ("fx", <currency>, <date>)                -> fx.csv row id
"""

from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from audit_env.generator.world import (
    ACC_SUSPENSE,
    Invoice,
    InvoiceBooking,
    Payment,
    ReceiptBooking,
    World,
)

MIN_FX_DELTA_MINOR = 5  # injected FX misstatements are always well above the matching tolerance


class InjectionSkipped(Exception):
    """No suitable target exists in this history; the generator simply injects fewer issues."""


@dataclass
class Issue:
    type: str
    affected: list[str]  # internal keys
    amount: int
    currency: str
    evidence: list[list[tuple]]
    variant: str
    scenario: str | None = None
    mutation: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------ helpers

def _journal_snapshot(world: World, key: str) -> list[dict] | None:
    booking = world.bookings.get(key)
    if booking is None:
        return None
    return [dict(vars(r)) for r in world.build_journal(key, booking).rows]


def _record(world: World, keys: list[str], change) -> dict[str, Any]:
    before = {k: _journal_snapshot(world, k) for k in keys}
    change()
    after = {k: _journal_snapshot(world, k) for k in keys}
    return {"journals_before": before, "journals_after": after}


def _free_period_invoices(world: World, *, currency: str | None = None, foreign: bool | None = None,
                          max_day: dt.date | None = None) -> list[Invoice]:
    out = []
    for inv in world.invoices.values():
        if inv.kind != "period" or inv.key in world.reserved or inv.tags:
            continue
        if currency and inv.currency != currency:
            continue
        if foreign is not None and (inv.currency != world.base) != foreign:
            continue
        if max_day and world.local(inv.issued_at) > max_day:
            continue
        out.append(inv)
    return sorted(out, key=lambda i: i.key)


def _single_receipt(world: World, inv: Invoice) -> Payment | None:
    """The only in-period payment of `inv`, if it is a plain full settlement."""
    pays = [p for p in world.payments.values() if p.refs and any(k == inv.key for k, _ in p.refs)]
    if len(pays) != 1:
        return None
    pay = pays[0]
    if (pay.key in world.reserved or pay.tags or len(pay.refs) != 1 or pay.amount != inv.total
            or f"pay:{pay.key}" not in world.bookings):
        return None
    return pay


def _paid_invoices(world: World, foreign: bool | None = None) -> list[tuple[Invoice, Payment]]:
    out = []
    for inv in _free_period_invoices(world, foreign=foreign):
        pay = _single_receipt(world, inv)
        if pay is not None:
            out.append((inv, pay))
    return out


def _wrong_rate_date(world: World, currency: str, ts: dt.datetime, variant: str) -> dt.date | None:
    """A rate date that a careless bookkeeper could use, giving a different effective rate."""
    correct_rate, correct_day = world.rate_on(currency, world.local(ts))
    if variant == "utc_date":
        candidates = [ts.astimezone(dt.UTC).date()]
    else:  # stale_rate
        candidates = [correct_day - dt.timedelta(days=d) for d in world.rng.sample(range(3, 15), 6)]
    for day in candidates:
        try:
            rate, _ = world.rate_on(currency, day)
        except KeyError:
            continue
        if rate != correct_rate:
            return day
    return None


def _duplicate_invoice_doc(world: World, inv: Invoice, latest: dt.datetime | None = None) -> Invoice:
    rng = world.rng
    shift = dt.timedelta(days=rng.randint(0, 2), hours=rng.randint(1, 6), minutes=rng.randint(0, 59))
    issued = inv.issued_at + shift
    if latest is not None and issued >= latest:
        issued = inv.issued_at + (latest - inv.issued_at) / 2
    end_local = world.local_dt(world.period_end, 22)
    if issued > end_local:
        issued = inv.issued_at + dt.timedelta(minutes=rng.randint(5, 50))
    dup = Invoice(
        key=world.new_key("I"), doc=world.new_key("D"), entity=inv.entity, issued_at=issued,
        due_at=issued + (inv.due_at - inv.issued_at), currency=inv.currency,
        line_items=copy.deepcopy(inv.line_items), total=inv.total, kind="period", tags={"injected_duplicate"},
    )
    world.invoices[dup.key] = dup
    return dup


# ------------------------------------------------------------------------------------------ injectors

def inject_duplicate_invoice(world: World, variant: str | None = None) -> list[Issue]:
    cands = _free_period_invoices(world, max_day=world.period_end - dt.timedelta(days=3))
    if not cands:
        raise InjectionSkipped("duplicate_invoice")
    inv = world.rng.choice(cands)
    holder: dict[str, Invoice] = {}

    def change() -> None:
        dup = _duplicate_invoice_doc(world, inv)
        holder["dup"] = dup
        world.bookings[f"inv:{dup.key}"] = InvoiceBooking(dup.key)

    # record against the not-yet-existing journal key after creation
    change()
    dup = holder["dup"]
    mutation = {"journals_before": {f"inv:{dup.key}": None},
                "journals_after": {f"inv:{dup.key}": _journal_snapshot(world, f"inv:{dup.key}")},
                "copied_from": inv.key, "new_document": dup.key}
    world.reserved.update({inv.key, dup.key})
    return [_duplicate_issue(inv, dup, variant or "copy", mutation=mutation)]


def _duplicate_issue(inv: Invoice, dup: Invoice, variant: str, scenario: str | None = None,
                     mutation: dict | None = None) -> Issue:
    pair = sorted([inv, dup], key=lambda i: i.key)
    docs = [("doc", pair[0].key), ("doc", pair[1].key)]
    return Issue(
        type="duplicate_invoice", affected=[pair[0].key, pair[1].key], amount=inv.total, currency=inv.currency,
        evidence=[
            docs + [("row", f"inv:{i.key}", "ar", i.key) for i in pair],
            docs + [("row", f"inv:{i.key}", "revenue", i.key) for i in pair],
        ],
        variant=variant, scenario=scenario, mutation=mutation or {},
    )


def inject_payment_allocation(world: World, variant: str) -> list[Issue]:
    cands = _paid_invoices(world)
    if not cands:
        raise InjectionSkipped("payment_allocation")
    inv, pay = world.rng.choice(cands)
    key = f"pay:{pay.key}"
    applied = 0 if variant == "missing" else max(1, pay.amount * world.rng.randint(25, 80) // 100)
    unapplied = pay.amount - applied

    def change() -> None:
        apps = []
        if applied:
            apps = [(inv.key, applied, world.conv(applied, inv.currency, world.invoice_rate(inv)))]
        world.bookings[key] = ReceiptBooking(pay.key, apps, ACC_SUSPENSE, unapplied)

    mutation = _record(world, [key], change)
    world.reserved.update({inv.key, pay.key})
    return [Issue(
        type="payment_allocation", affected=[pay.key, inv.key], amount=unapplied, currency=pay.currency,
        evidence=[[("doc", pay.key), ("row", key, "suspense", None)]], variant=variant, mutation=mutation,
    )]


def inject_fx_conversion(world: World, variant: str) -> list[Issue]:
    try:
        return _inject_fx_conversion(world, variant)
    except InjectionSkipped:
        if variant == "stale_rate" or "stale_rate" not in world.cfg.fx_variants:
            raise
        return _inject_fx_conversion(world, "stale_rate")


def _inject_fx_conversion(world: World, variant: str) -> list[Issue]:
    rng = world.rng
    targets: list[tuple[str, Any]] = [("inv", i) for i in _free_period_invoices(world, foreign=True)]
    targets += [("pay", p) for _, p in _paid_invoices(world, foreign=True)]
    rng.shuffle(targets)
    for kind, obj in targets:
        ts = obj.issued_at if kind == "inv" else obj.ts
        wrong = _wrong_rate_date(world, obj.currency, ts, variant)
        if wrong is None:
            continue
        amount = obj.total if kind == "inv" else obj.amount
        correct_rate, correct_day = world.rate_on(obj.currency, world.local(ts))
        wrong_rate, _ = world.rate_on(obj.currency, wrong)
        delta = abs(world.conv(amount, obj.currency, wrong_rate) - world.conv(amount, obj.currency, correct_rate))
        if delta < MIN_FX_DELTA_MINOR:
            continue
        key = f"{kind}:{obj.key}"

        def change(key=key, wrong=wrong) -> None:
            booking = copy.deepcopy(world.bookings[key])
            booking.rate_date = wrong
            world.bookings[key] = booking

        mutation = _record(world, [key], change)
        mutation["correct_rate"], mutation["used_rate"] = correct_rate, wrong_rate
        fx_ev = ("fx", obj.currency, correct_day)
        if kind == "inv":
            evidence = [[("row", key, "ar", obj.key), ("doc", obj.key), fx_ev],
                        [("row", key, "revenue", obj.key), ("doc", obj.key), fx_ev]]
            world.reserved.add(obj.key)
        else:
            evidence = [[("row", key, "cash", None), ("doc", obj.key), fx_ev]]
            world.reserved.update({obj.key, obj.refs[0][0]})
        return [Issue(type="fx_conversion", affected=[obj.key], amount=delta, currency=world.base,
                      evidence=evidence, variant=variant, mutation=mutation)]
    raise InjectionSkipped("fx_conversion")


def inject_period_cutoff(world: World, variant: str) -> list[Issue]:
    try:
        return _inject_period_cutoff(world, variant)
    except InjectionSkipped:
        if variant == "backdated" or "backdated" not in world.cfg.cutoff_variants:
            raise
        return _inject_period_cutoff(world, "backdated")


def _inject_period_cutoff(world: World, variant: str) -> list[Issue]:
    rng = world.rng
    nxt = [i for i in world.invoices.values() if i.kind == "next" and i.key not in world.reserved]
    if variant == "utc_boundary":
        nxt = [i for i in nxt if "utc_boundary_candidate" in i.tags]
    else:
        nxt = [i for i in nxt if not i.tags]
    if not nxt:
        raise InjectionSkipped("period_cutoff")
    inv = rng.choice(sorted(nxt, key=lambda i: i.key))
    key = f"inv:{inv.key}"
    if variant == "utc_boundary":
        posting_ts = inv.issued_at  # same instant; booked by its UTC date, which is still in the period
    else:
        posting_ts = world.local_dt(world.period_end, rng.randint(15, 20), rng.randint(0, 59))

    def change() -> None:
        world.bookings[key] = InvoiceBooking(inv.key, ts=posting_ts)

    mutation = _record(world, [key], change)
    world.reserved.add(inv.key)
    return [Issue(
        type="period_cutoff", affected=[inv.key], amount=inv.total, currency=inv.currency,
        evidence=[[("doc", inv.key), ("row", key, "ar", inv.key)],
                  [("doc", inv.key), ("row", key, "revenue", inv.key)]],
        variant=variant, mutation=mutation,
    )]


# ------------------------------------------------------------------------------------------ scenarios

def scenario_partial_fx_payment_boundary(world: World) -> list[Issue]:
    """Partial application of a foreign-currency receipt dated on the period boundary, converted
    with the rate of its UTC date (or a stale rate) instead of its local date."""
    rng = world.rng
    offset_h = world.tz.utcoffset(None).total_seconds() / 3600
    cands = [(i, p) for i, p in _paid_invoices(world, foreign=True)
             if world.local(i.issued_at) <= world.period_end - dt.timedelta(days=3)]
    rng.shuffle(cands)
    for inv, pay in cands:
        if offset_h < 0:
            boundary = world.local_dt(world.period_end, rng.randint(24 - int(-offset_h), 23), rng.randint(0, 59))
        elif offset_h > 0:
            boundary = world.local_dt(world.period_end, rng.randint(0, max(0, int(offset_h) - 1)),
                                      rng.randint(0, 59))
        else:
            boundary = world.local_dt(world.period_end, rng.randint(9, 17), rng.randint(0, 59))
        new_ts = boundary.astimezone(dt.UTC)
        if new_ts <= inv.issued_at:
            continue
        old_ts = pay.ts
        pay.ts = new_ts  # history decision: this receipt arrives on the last local day of the period
        wrong = _wrong_rate_date(world, pay.currency, new_ts, "utc_date") or \
            _wrong_rate_date(world, pay.currency, new_ts, "stale_rate")
        if wrong is None:
            pay.ts = old_ts
            continue
        correct_rate, correct_day = world.rate_on(pay.currency, world.local(new_ts))
        wrong_rate, _ = world.rate_on(pay.currency, wrong)
        delta = abs(world.conv(pay.amount, pay.currency, wrong_rate) -
                    world.conv(pay.amount, pay.currency, correct_rate))
        if delta < MIN_FX_DELTA_MINOR:
            pay.ts = old_ts
            continue
        key = f"pay:{pay.key}"
        applied = max(1, pay.amount * rng.randint(30, 75) // 100)
        unapplied = pay.amount - applied
        # clean booking for the moved receipt is still a full settlement (recorded as "before")
        world.bookings[key] = ReceiptBooking(pay.key, [(inv.key, pay.amount, world.ar_base(inv))])

        def change() -> None:
            apps = [(inv.key, applied, world.conv(applied, inv.currency, world.invoice_rate(inv)))]  # noqa: B023 - called immediately
            world.bookings[key] = ReceiptBooking(pay.key, apps, ACC_SUSPENSE, unapplied, rate_date=wrong)  # noqa: B023

        mutation = _record(world, [key], change)
        mutation.update({"moved_receipt_from": old_ts.isoformat(), "correct_rate": correct_rate,
                         "used_rate": wrong_rate})
        world.reserved.update({inv.key, pay.key})
        name = "partial_fx_payment_boundary"
        return [
            Issue(type="payment_allocation", affected=[pay.key, inv.key], amount=unapplied, currency=pay.currency,
                  evidence=[[("doc", pay.key), ("row", key, "suspense", None)]], variant="partial",
                  scenario=name, mutation=mutation),
            Issue(type="fx_conversion", affected=[pay.key], amount=delta, currency=world.base,
                  evidence=[[("row", key, "cash", None), ("doc", pay.key), ("fx", pay.currency, correct_day)]],
                  variant="utc_date" if wrong == new_ts.date() else "stale_rate", scenario=name,
                  mutation={"see": "payment_allocation issue of the same scenario"}),
        ]
    raise InjectionSkipped("partial_fx_payment_boundary")


def scenario_duplicate_with_misapplied_payment(world: World) -> list[Issue]:
    """Invoice duplicated; the customer's payment for the original is applied to the duplicate."""
    rng = world.rng
    cands = [(i, p) for i, p in _paid_invoices(world)
             if p.ts - i.issued_at > dt.timedelta(hours=6)]
    if not cands:
        raise InjectionSkipped("duplicate_with_misapplied_payment")
    inv, pay = rng.choice(cands)
    dup = _duplicate_invoice_doc(world, inv, latest=pay.ts)
    world.bookings[f"inv:{dup.key}"] = InvoiceBooking(dup.key)
    key = f"pay:{pay.key}"

    def change() -> None:
        world.bookings[key] = ReceiptBooking(pay.key, [(dup.key, pay.amount, world.ar_base(dup))])

    mutation = _record(world, [key], change)
    mutation.update({"copied_from": inv.key, "new_document": dup.key,
                     "duplicate_journal": _journal_snapshot(world, f"inv:{dup.key}")})
    world.reserved.update({inv.key, dup.key, pay.key})
    name = "duplicate_with_misapplied_payment"
    return [
        _duplicate_issue(inv, dup, "copy", scenario=name, mutation=mutation),
        Issue(type="payment_allocation", affected=[pay.key, inv.key], amount=pay.amount, currency=pay.currency,
              evidence=[[("doc", pay.key), ("row", key, "ar", dup.key)]], variant="misapplied",
              scenario=name, mutation={"see": "duplicate_invoice issue of the same scenario"}),
    ]


INJECTORS = {
    "duplicate_invoice": inject_duplicate_invoice,
    "payment_allocation": inject_payment_allocation,
    "fx_conversion": inject_fx_conversion,
    "period_cutoff": inject_period_cutoff,
}
SCENARIO_BUILDERS = {
    "partial_fx_payment_boundary": scenario_partial_fx_payment_boundary,
    "duplicate_with_misapplied_payment": scenario_duplicate_with_misapplied_payment,
}
VARIANTS_ATTR = {
    "duplicate_invoice": None,
    "payment_allocation": "allocation_variants",
    "fx_conversion": "fx_variants",
    "period_cutoff": "cutoff_variants",
}
