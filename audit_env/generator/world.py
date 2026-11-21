"""Hidden company history: documents, FX table and double-entry bookings.

The generator first builds a *consistent* history (documents + clean bookings), validates its
accounting identities, and only then lets error injectors replace individual bookings. Every
injected change is recorded in provenance with the booking before and after the change.
"""

from __future__ import annotations

import copy
import datetime as dt
import random
from dataclasses import dataclass, field

from audit_env.config import GeneratorConfig
from audit_env.money import convert_minor, fx_id

ACC_CASH = "1000"
ACC_AR = "1100"
ACC_CUSTOMER_CREDIT = "2200"
ACC_SUSPENSE = "2900"
ACC_REVENUE = "4000"
ACC_FX = "7900"

CHART_OF_ACCOUNTS = {
    ACC_CASH: "Cash at bank",
    ACC_AR: "Accounts receivable",
    ACC_CUSTOMER_CREDIT: "Customer credit balances (overpayments awaiting refund)",
    ACC_SUSPENSE: "Unapplied cash (receipts without usable remittance)",
    ACC_REVENUE: "Revenue",
    ACC_FX: "Realised FX gain/loss",
}

ROUNDING = "half_up"


@dataclass
class Invoice:
    key: str
    doc: str
    entity: str
    issued_at: dt.datetime  # aware, company offset
    due_at: dt.datetime
    currency: str
    line_items: list[dict]
    total: int
    kind: str  # opening | period | next
    tags: set[str] = field(default_factory=set)


@dataclass
class Payment:
    key: str
    doc: str
    entity: str | None
    ts: dt.datetime  # aware
    currency: str
    amount: int
    direction: str  # in | out
    refs: list[tuple[str, int]] | None  # remittance allocations (invoice key, amount)
    refund_of: str | None = None
    tags: set[str] = field(default_factory=set)


@dataclass
class Row:
    role: str  # ar | revenue | cash | fx | suspense | customer_credit
    account: str
    debit: int
    credit: int
    entity: str | None
    inv: str | None
    pay: str | None
    txn_currency: str
    txn_amount: int
    fx_rate: str


@dataclass
class Journal:
    key: str  # inv:<key> | pay:<key>
    ts: dt.datetime
    rows: list[Row]


@dataclass
class InvoiceBooking:
    inv: str
    ts: dt.datetime | None = None  # posting timestamp override (cut-off errors)
    rate_date: dt.date | None = None  # FX rate date override (FX errors)


@dataclass
class ReceiptBooking:
    pay: str
    apps: list[tuple[str, int, int]]  # (invoice key, txn amount, base amount) credited to AR
    excess_account: str | None = None
    excess: int = 0
    rate_date: dt.date | None = None


@dataclass
class RefundBooking:
    pay: str


Booking = InvoiceBooking | ReceiptBooking | RefundBooking


class World:
    """Mutable hidden state of one generated company/period."""

    def __init__(self, cfg: GeneratorConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.base: str = ""
        self.foreign: list[str] = []
        self.tz: dt.timezone = dt.UTC
        self.tz_name: str = "+00:00"
        self.period_start: dt.date = dt.date(2026, 1, 1)
        self.period_end: dt.date = dt.date(2026, 1, 31)
        self.fx: dict[tuple[dt.date, str], str] = {}
        self.entities: list[str] = []
        self.invoices: dict[str, Invoice] = {}
        self.payments: dict[str, Payment] = {}
        self.bookings: dict[str, Booking] = {}
        self.reserved: set[str] = set()  # invoice/payment keys already used by a distractor or issue
        self._counters: dict[str, int] = {}

    # ---------------------------------------------------------------- ids
    def new_key(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]:04d}"

    # ---------------------------------------------------------------- time
    def local(self, ts: dt.datetime) -> dt.date:
        return ts.astimezone(self.tz).date()

    def in_period(self, day: dt.date) -> bool:
        return self.period_start <= day <= self.period_end

    def local_dt(self, day: dt.date, hour: int, minute: int = 0, second: int = 0) -> dt.datetime:
        return dt.datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=self.tz)

    # ---------------------------------------------------------------- fx
    def rate_on(self, currency: str, day: dt.date) -> tuple[str, dt.date]:
        """Policy rate: latest published rate dated on or before `day` (company-local date)."""
        if currency == self.base:
            return "1", day
        probe = day
        for _ in range(15):
            rate = self.fx.get((probe, currency))
            if rate is not None:
                return rate, probe
            probe -= dt.timedelta(days=1)
        raise KeyError(f"no fx rate for {currency} near {day}")

    def fx_evidence(self, currency: str, day: dt.date) -> str | None:
        if currency == self.base:
            return None
        _, rate_day = self.rate_on(currency, day)
        return fx_id(rate_day, currency, self.base)

    def conv(self, amount: int, currency: str, rate: str) -> int:
        return convert_minor(amount, currency, self.base, rate, ROUNDING)

    def invoice_rate(self, inv: Invoice) -> str:
        return self.rate_on(inv.currency, self.local(inv.issued_at))[0]

    def ar_base(self, inv: Invoice) -> int:
        """Correct base-currency AR carrying amount of an invoice."""
        return self.conv(inv.total, inv.currency, self.invoice_rate(inv))

    # ---------------------------------------------------------------- bookings -> journals
    def build_journal(self, key: str, booking: Booking) -> Journal:
        if isinstance(booking, InvoiceBooking):
            return self._invoice_journal(key, booking)
        if isinstance(booking, ReceiptBooking):
            return self._receipt_journal(key, booking)
        return self._refund_journal(key, booking)

    def _invoice_journal(self, key: str, b: InvoiceBooking) -> Journal:
        inv = self.invoices[b.inv]
        rate_day = b.rate_date or self.local(inv.issued_at)
        rate, _ = self.rate_on(inv.currency, rate_day)
        base = self.conv(inv.total, inv.currency, rate)
        common = dict(entity=inv.entity, inv=inv.key, pay=None, txn_currency=inv.currency,
                      txn_amount=inv.total, fx_rate=rate)
        rows = [
            Row(role="ar", account=ACC_AR, debit=base, credit=0, **common),
            Row(role="revenue", account=ACC_REVENUE, debit=0, credit=base, **common),
        ]
        return Journal(key, b.ts or inv.issued_at, rows)

    def _receipt_journal(self, key: str, b: ReceiptBooking) -> Journal:
        pay = self.payments[b.pay]
        rate_day = b.rate_date or self.local(pay.ts)
        rate, _ = self.rate_on(pay.currency, rate_day)
        cash = self.conv(pay.amount, pay.currency, rate)
        rows = [Row("cash", ACC_CASH, cash, 0, pay.entity, None, pay.key, pay.currency, pay.amount, rate)]
        credited = 0
        for inv_key, txn, base in b.apps:
            inv = self.invoices[inv_key]
            rows.append(Row("ar", ACC_AR, 0, base, inv.entity, inv_key, pay.key, pay.currency, txn,
                            self.invoice_rate(inv)))
            credited += base
        if b.excess:
            role = "suspense" if b.excess_account == ACC_SUSPENSE else "customer_credit"
            excess_base = self.conv(b.excess, pay.currency, rate)
            rows.append(Row(role, b.excess_account, 0, excess_base, pay.entity, None, pay.key,
                            pay.currency, b.excess, rate))
            credited += excess_base
        plug = cash - credited
        if plug:
            rows.append(Row("fx", ACC_FX, max(-plug, 0), max(plug, 0), pay.entity, None, pay.key,
                            self.base, abs(plug), "1"))
        return Journal(key, pay.ts, rows)

    def _refund_journal(self, key: str, b: RefundBooking) -> Journal:
        pay = self.payments[b.pay]
        rate, _ = self.rate_on(pay.currency, self.local(pay.ts))
        base = self.conv(pay.amount, pay.currency, rate)
        common = dict(entity=pay.entity, inv=None, pay=pay.key, txn_currency=pay.currency,
                      txn_amount=pay.amount, fx_rate=rate)
        rows = [
            Row(role="customer_credit", account=ACC_CUSTOMER_CREDIT, debit=base, credit=0, **common),
            Row(role="cash", account=ACC_CASH, debit=0, credit=base, **common),
        ]
        return Journal(key, pay.ts, rows)

    def journals(self) -> dict[str, Journal]:
        return {key: self.build_journal(key, b) for key, b in self.bookings.items()}

    def snapshot(self) -> World:
        clone = copy.copy(self)
        clone.invoices = copy.deepcopy(self.invoices)
        clone.payments = copy.deepcopy(self.payments)
        clone.bookings = copy.deepcopy(self.bookings)
        clone.reserved = set(self.reserved)
        return clone
