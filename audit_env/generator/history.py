"""Build a consistent hidden history: company, FX table, invoices, payments, distractors, clean bookings."""

from __future__ import annotations

import calendar
import datetime as dt
from collections import defaultdict
from decimal import Decimal

from audit_env.generator.world import (
    ACC_CUSTOMER_CREDIT,
    ACC_SUSPENSE,
    Invoice,
    InvoiceBooking,
    Payment,
    ReceiptBooking,
    RefundBooking,
    World,
)
from audit_env.money import USD_VALUE, exponent, format_rate, parse_offset

SERVICES = [
    "Consulting services", "Software licence", "Maintenance and support", "Cloud hosting",
    "Implementation works", "Training session", "Data migration", "Hardware rental",
    "Logistics services", "Marketing campaign", "Audit preparation support", "Spare parts",
]


def _pick_range(world: World, bounds: list[int]) -> int:
    return world.rng.randint(bounds[0], bounds[1])


def _rand_local(world: World, day: dt.date, hours: tuple[int, int] = (8, 19)) -> dt.datetime:
    rng = world.rng
    return world.local_dt(day, rng.randint(*hours), rng.randint(0, 59), rng.randint(0, 59))


def _period_days(world: World) -> list[dt.date]:
    n = (world.period_end - world.period_start).days + 1
    return [world.period_start + dt.timedelta(days=i) for i in range(n)]


def setup_company(world: World) -> None:
    cfg, rng = world.cfg, world.rng
    world.base = rng.choice(cfg.base_currencies)
    pool = [c for c in cfg.foreign_currencies if c != world.base]
    k = min(len(pool), _pick_range(world, cfg.n_foreign_currencies))
    world.foreign = sorted(rng.sample(pool, k))
    world.tz_name = rng.choice(cfg.timezones)
    world.tz = parse_offset(world.tz_name)
    month = rng.choice(cfg.months)
    world.period_start = dt.date(cfg.year, month, 1)
    world.period_end = dt.date(cfg.year, month, calendar.monthrange(cfg.year, month)[1])
    world.entities = [world.new_key("E") for _ in range(_pick_range(world, cfg.n_customers))]
    _build_fx(world)


def _build_fx(world: World) -> None:
    """Business-day rates (Mon-Fri) from 60 days before the period to 15 days after it."""
    rng = world.rng
    start = world.period_start - dt.timedelta(days=60)
    end = world.period_end + dt.timedelta(days=15)
    for cur in world.foreign:
        level = USD_VALUE[cur] / USD_VALUE[world.base] * Decimal(str(round(rng.uniform(0.95, 1.05), 4)))
        day = start
        while day <= end:
            if day.weekday() < 5:
                level *= Decimal(str(round(1 + rng.gauss(0, 0.004), 6)))
                world.fx[(day, cur)] = format_rate(level)
            day += dt.timedelta(days=1)


def _unit_price(world: World, currency: str) -> int:
    usd = Decimal(world.rng.choice([25, 40, 75, 120, 180, 250, 400, 650, 900, 1500, 2400]))
    usd *= Decimal(str(round(world.rng.uniform(0.8, 1.2), 2)))
    major = usd / USD_VALUE[currency]
    # round to a "price-list looking" number of minor units
    minor = int(major * (Decimal(10) ** exponent(currency)))
    step = 10 ** max(exponent(currency) - 1, 0) if currency != "JPY" else 10
    return max(step, minor - minor % step)


def _line_items(world: World, currency: str, ref: str, description: str | None = None) -> list[dict]:
    rng = world.rng
    items = []
    for i in range(rng.randint(1, 3)):
        qty = rng.randint(1, 12)
        unit = _unit_price(world, currency)
        desc = description or rng.choice(SERVICES)
        items.append({
            "line_no": i + 1,
            "description": f"{desc} ({ref})",
            "quantity": qty,
            "unit_price_minor": unit,
            "amount_minor": qty * unit,
        })
    return items


def _new_invoice(world: World, kind: str, issued: dt.datetime, *, entity: str | None = None,
                 currency: str | None = None, items: list[dict] | None = None) -> Invoice:
    rng = world.rng
    if currency is None:
        currency = rng.choice(world.foreign) if world.foreign and rng.random() < world.cfg.foreign_share \
            else world.base
    entity = entity or rng.choice(world.entities)
    key = world.new_key("I")
    ref = f"PO-{rng.randint(10000, 99999)}-{key[-3:]}"
    items = items or _line_items(world, currency, ref)
    inv = Invoice(
        key=key, doc=world.new_key("D"), entity=entity, issued_at=issued,
        due_at=issued + dt.timedelta(days=rng.choice([14, 30, 45])), currency=currency,
        line_items=items, total=sum(li["amount_minor"] for li in items), kind=kind,
    )
    world.invoices[key] = inv
    return inv


def _new_payment(world: World, inv: Invoice | None, ts: dt.datetime, amount: int, *, currency: str | None = None,
                 refs: list[tuple[str, int]] | None = None, entity: str | None = None,
                 direction: str = "in", refund_of: str | None = None) -> Payment:
    key = world.new_key("P")
    pay = Payment(
        key=key, doc=world.new_key("D"), entity=entity if entity is not None else (inv.entity if inv else None),
        ts=ts.astimezone(dt.UTC), currency=currency or (inv.currency if inv else world.base),
        amount=amount, direction=direction, refs=refs, refund_of=refund_of,
    )
    world.payments[key] = pay
    return pay


def _payment_time(world: World, after: dt.datetime, min_days: int = 1, max_days: int = 25) -> dt.datetime | None:
    """Random in-period local business-hours timestamp after `after`, or None if it falls outside."""
    day = world.local(after) + dt.timedelta(days=world.rng.randint(min_days, max_days))
    if not world.in_period(day):
        return None
    return _rand_local(world, day, (9, 17))


def build_documents(world: World) -> None:
    cfg, rng = world.cfg, world.rng
    days = _period_days(world)
    # opening invoices: issued last period, open at period start, settled during this period
    for _ in range(_pick_range(world, cfg.n_opening_invoices)):
        day = world.period_start - dt.timedelta(days=rng.randint(5, 40))
        _new_invoice(world, "opening", _rand_local(world, day))
    for _ in range(_pick_range(world, cfg.n_period_invoices)):
        _new_invoice(world, "period", _rand_local(world, rng.choice(days[:-1])))
    for _ in range(_pick_range(world, cfg.n_next_period_invoices)):
        day = world.period_end + dt.timedelta(days=rng.randint(1, 5))
        _new_invoice(world, "next", _rand_local(world, day))
    _add_document_distractors(world)


def _add_document_distractors(world: World) -> None:
    cfg, rng = world.cfg, world.rng
    counts = {k: _pick_range(world, v) for k, v in cfg.distractor_counts.items()}
    # recurring invoices: same customer, same lines and amounts, different service week
    for _ in range(counts.get("recurring_invoice", 0)):
        entity = rng.choice(world.entities)
        currency = world.base
        contract = f"CT-{rng.randint(1000, 9999)}"
        qty, unit = rng.randint(1, 4), _unit_price(world, currency)
        start = world.period_start + dt.timedelta(days=rng.randint(0, 3))
        for week in range(4):
            day = start + dt.timedelta(days=7 * week)
            if not world.in_period(day):
                break
            iso_week = day.isocalendar()
            items = [{
                "line_no": 1,
                "description": f"Support subscription, week {iso_week[0]}-W{iso_week[1]:02d} ({contract})",
                "quantity": qty, "unit_price_minor": unit, "amount_minor": qty * unit,
            }]
            inv = _new_invoice(world, "period", _rand_local(world, day), entity=entity, currency=currency,
                               items=items)
            inv.tags.add("recurring_invoice")
            world.reserved.add(inv.key)
    # boundary timestamps: local date and UTC date fall into different periods
    offset_hours = world.tz.utcoffset(None).total_seconds() / 3600
    for _ in range(counts.get("boundary_timestamp", 0)):
        if offset_hours > 0:
            hours = (0, max(0, int(offset_hours) - 1))
            # in-period invoice whose UTC instant is still in the previous month
            inv = _new_invoice(world, "period", _rand_local(world, world.period_start, hours))
            # next-period invoice whose UTC instant is still inside this period (correctly NOT booked)
            nxt = _new_invoice(world, "next", _rand_local(world, world.period_end + dt.timedelta(days=1), hours))
            nxt.tags.add("boundary_timestamp")
        elif offset_hours < 0:
            hours = (24 - int(-offset_hours), 23)
            inv = _new_invoice(world, "period", _rand_local(world, world.period_end, hours))
        else:
            continue
        inv.tags.add("boundary_timestamp")
    # near duplicates: same customer, currency, quantities, prices and total as an existing invoice, but a
    # different purchase order (PO) in the line descriptions - a separate legitimate order, not a duplicate
    last_day = world.period_end - dt.timedelta(days=2)
    for _ in range(counts.get("near_duplicate", 0)):
        pool = sorted((i for i in world.invoices.values() if i.kind == "period" and not i.tags
                       and i.key not in world.reserved and world.local(i.issued_at) <= last_day),
                      key=lambda i: i.key)
        if not pool:
            break
        src = rng.choice(pool)
        ref = f"PO-{rng.randint(10000, 99999)}-{rng.randint(100, 999)}"
        while any(ref in li["description"] for li in src.line_items):
            ref = f"PO-{rng.randint(10000, 99999)}-{rng.randint(100, 999)}"
        items = [{**li, "description": f"{li['description'].rsplit(' (', 1)[0]} ({ref})"} for li in src.line_items]
        day = world.local(src.issued_at) + dt.timedelta(days=rng.randint(1, 2))
        near = _new_invoice(world, "period", _rand_local(world, day), entity=src.entity, currency=src.currency,
                            items=items)
        near.tags.add("near_duplicate")
        world.reserved.update({src.key, near.key})
    # utc-boundary candidates for cut-off errors (next-period invoices issued just after midnight)
    if offset_hours > 0:
        hours = (0, max(0, int(offset_hours) - 1))
        nxt = _new_invoice(world, "next", _rand_local(world, world.period_end + dt.timedelta(days=1), hours))
        nxt.tags.add("utc_boundary_candidate")


def build_payments(world: World) -> None:
    """Settlement plan for opening/period invoices plus payment distractors."""
    cfg, rng = world.cfg, world.rng
    counts = {k: _pick_range(world, v) for k, v in cfg.distractor_counts.items()}
    payable = [inv for inv in world.invoices.values() if inv.kind in ("opening", "period")]
    payable.sort(key=lambda i: i.key)

    # multi-invoice payments: two same-customer same-currency invoices settled together
    singles = [i for i in payable if i.kind == "period" and i.key not in world.reserved]
    for _ in range(counts.get("multi_invoice_payment", 0)):
        by_group: dict[tuple[str, str], list[Invoice]] = defaultdict(list)
        for inv in singles:
            by_group[(inv.entity, inv.currency)].append(inv)
        groups = [g for g in by_group.values() if len(g) >= 2]
        if not groups:
            break
        a, b = rng.sample(rng.choice(groups), 2)
        ts = _payment_time(world, max(a.issued_at, b.issued_at), 1, 10)
        if ts is None:
            continue
        pay = _new_payment(world, a, ts, a.total + b.total, refs=[(a.key, a.total), (b.key, b.total)])
        pay.tags.add("multi_invoice_payment")
        for inv in (a, b):
            singles.remove(inv)
            world.reserved.add(inv.key)
        world.reserved.add(pay.key)

    # split payments
    early = world.period_end - dt.timedelta(days=10)
    placed, attempts = 0, 0
    while placed < counts.get("split_payment", 0) and attempts < 30:
        attempts += 1
        cands = [i for i in singles if i.total >= 100 and world.local(i.issued_at) <= early]
        if not cands:
            break
        inv = rng.choice(cands)
        t1 = _payment_time(world, inv.issued_at, 1, 4)
        t2 = _payment_time(world, t1, 1, 4) if t1 else None
        if t1 is None or t2 is None:
            continue
        placed += 1
        first = inv.total * rng.randint(30, 70) // 100
        p1 = _new_payment(world, inv, t1, first, refs=[(inv.key, first)])
        p2 = _new_payment(world, inv, t2, inv.total - first, refs=[(inv.key, inv.total - first)])
        for x in (p1, p2):
            x.tags.add("split_payment")
            world.reserved.add(x.key)
        singles.remove(inv)
        world.reserved.add(inv.key)

    # double payment + refund (base currency only)
    placed, attempts = 0, 0
    while placed < counts.get("double_payment_refund", 0) and attempts < 30:
        attempts += 1
        cands = [i for i in singles if i.currency == world.base and world.local(i.issued_at) <= early]
        if not cands:
            break
        inv = rng.choice(cands)
        t1 = _payment_time(world, inv.issued_at, 1, 3)
        t2 = _payment_time(world, t1, 1, 3) if t1 else None
        t3 = _payment_time(world, t2, 1, 3) if t2 else None
        if t3 is None:
            continue
        placed += 1
        p1 = _new_payment(world, inv, t1, inv.total, refs=[(inv.key, inv.total)])
        p2 = _new_payment(world, inv, t2, inv.total, refs=[(inv.key, inv.total)])
        p3 = _new_payment(world, inv, t3, inv.total, refs=None, direction="out", refund_of=p2.key)
        for x in (p1, p2, p3):
            x.tags.add("double_payment_refund")
            world.reserved.add(x.key)
        singles.remove(inv)
        world.reserved.add(inv.key)

    # double payment, only part of the excess refunded: the rest correctly stays a customer credit in 2200
    placed, attempts = 0, 0
    while placed < counts.get("partial_refund", 0) and attempts < 30:
        attempts += 1
        cands = [i for i in singles if i.currency == world.base and world.local(i.issued_at) <= early]
        if not cands:
            break
        inv = rng.choice(cands)
        t1 = _payment_time(world, inv.issued_at, 1, 3)
        t2 = _payment_time(world, t1, 1, 3) if t1 else None
        t3 = _payment_time(world, t2, 1, 3) if t2 else None
        if t3 is None:
            continue
        placed += 1
        refunded = max(1, inv.total * rng.randint(20, 80) // 100)
        p1 = _new_payment(world, inv, t1, inv.total, refs=[(inv.key, inv.total)])
        p2 = _new_payment(world, inv, t2, inv.total, refs=[(inv.key, inv.total)])
        p3 = _new_payment(world, inv, t3, refunded, refs=None, direction="out", refund_of=p2.key)
        for x in (p1, p2, p3):
            x.tags.add("partial_refund")
            world.reserved.add(x.key)
        singles.remove(inv)
        world.reserved.add(inv.key)

    # unidentified receipts (no remittance) parked in suspense
    for _ in range(counts.get("unidentified_payment", 0)):
        day = rng.choice(_period_days(world))
        amount = _unit_price(world, world.base) * rng.randint(1, 5)
        pay = _new_payment(world, None, _rand_local(world, day, (9, 17)), amount, currency=world.base,
                           refs=None, entity=None)
        pay.tags.add("unidentified_payment")
        world.reserved.add(pay.key)

    # everything else: opening invoices are settled; period invoices are settled with p=0.7
    handled = {k for p in world.payments.values() if p.refs for k, _ in p.refs}
    for inv in payable:
        if inv.key in handled:
            continue
        if inv.kind == "period" and rng.random() > 0.7:
            continue
        if inv.kind == "opening":
            ts = _rand_local(world, rng.choice(_period_days(world)), (9, 17))
        else:
            ts = _payment_time(world, inv.issued_at, 2, 25)
        if ts is None:
            continue
        _new_payment(world, inv, ts, inv.total, refs=[(inv.key, inv.total)])

    # receipts dated in the first days of the next period for invoices still open at period end: the document is
    # published, but by policy it is NOT booked in this period (candidates for receipt cut-off errors)
    for _ in range(counts.get("next_period_receipt", 0)):
        handled = {k for p in world.payments.values() if p.refs for k, _ in p.refs}
        cands = [i for i in payable if i.kind == "period" and i.key not in handled and i.key not in world.reserved
                 and not i.tags]
        if not cands:
            break
        inv = rng.choice(cands)
        ts = _rand_local(world, world.period_end + dt.timedelta(days=rng.randint(1, 5)), (9, 17))
        pay = _new_payment(world, inv, ts, inv.total, refs=[(inv.key, inv.total)])
        pay.tags.add("next_period_receipt")
        world.reserved.update({inv.key, pay.key})


def plan_bookings(world: World) -> None:
    """Clean bookings for everything that belongs to the period (by company-local date)."""
    bookings = {}
    for inv in sorted(world.invoices.values(), key=lambda i: (i.issued_at, i.key)):
        if inv.kind == "period":
            bookings[f"inv:{inv.key}"] = InvoiceBooking(inv.key)
    applied: dict[str, int] = defaultdict(int)
    applied_base: dict[str, int] = defaultdict(int)
    for pay in sorted(world.payments.values(), key=lambda p: (p.ts, p.key)):
        if not world.in_period(world.local(pay.ts)):
            continue
        if pay.direction == "out":
            bookings[f"pay:{pay.key}"] = RefundBooking(pay.key)
            continue
        if pay.refs is None:
            bookings[f"pay:{pay.key}"] = ReceiptBooking(pay.key, [], ACC_SUSPENSE, pay.amount)
            continue
        apps, excess = [], 0
        for inv_key, amount in pay.refs:
            inv = world.invoices[inv_key]
            open_amount = inv.total - applied[inv_key]
            take = min(amount, open_amount)
            excess += amount - take
            if take <= 0:
                continue
            if take == open_amount:
                base = world.ar_base(inv) - applied_base[inv_key]
            else:
                base = world.conv(take, inv.currency, world.invoice_rate(inv))
            apps.append((inv_key, take, base))
            applied[inv_key] += take
            applied_base[inv_key] += base
        bookings[f"pay:{pay.key}"] = ReceiptBooking(
            pay.key, apps, ACC_CUSTOMER_CREDIT if excess else None, excess)
    world.bookings = bookings


def build_history(world: World) -> None:
    setup_company(world)
    build_documents(world)
    build_payments(world)
    plan_bookings(world)
