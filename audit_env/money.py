"""Money, FX and time helpers shared by the generator, rules and grader.

Conventions (see DESIGN.md):
* money is always an integer number of minor units plus an ISO-4217 code;
* FX rates are decimal strings meaning "1 unit of source = rate units of target";
* conversion: minor_dst = round(minor_src * rate * 10**(exp_dst - exp_src)) with policy rounding;
* timestamps are ISO-8601 with an explicit UTC offset; business dates are taken in the
  company timezone declared in manifest.json.
"""

from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, getcontext

getcontext().prec = 50

CURRENCY_EXPONENT: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "JPY": 0,
    "KWD": 3,
}

# Approximate value of one unit in USD; only used to make synthetic rates plausible.
USD_VALUE: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "KZT": Decimal("0.00205"),
    "JPY": Decimal("0.0067"),
    "KWD": Decimal("3.25"),
}

ROUNDING_MODES = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}


def exponent(currency: str) -> int:
    return CURRENCY_EXPONENT[currency]


def convert_minor(amount_minor: int, src: str, dst: str, rate: Decimal | str, rounding: str = "half_up") -> int:
    """Convert an integer minor amount from src to dst using `rate` (1 src = rate dst)."""
    if src == dst:
        return int(amount_minor)
    rate_d = Decimal(rate)
    scaled = Decimal(amount_minor) * rate_d * (Decimal(10) ** (exponent(dst) - exponent(src)))
    return int(scaled.quantize(Decimal(1), rounding=ROUNDING_MODES[rounding]))


def format_rate(value: Decimal, significant: int = 6) -> str:
    """Round to `significant` significant digits and render as plain decimal (no exponent)."""
    if value == 0:
        return "0"
    q = Decimal(1).scaleb(value.adjusted() - significant + 1)
    out = value.quantize(q, rounding=ROUND_HALF_EVEN)
    text = format(out, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_offset(offset: str) -> dt.timezone:
    """Parse '+05:00' / '-03:00' / '+05:30' into a fixed-offset timezone."""
    sign = 1 if offset[0] == "+" else -1
    hours, minutes = offset[1:].split(":")
    return dt.timezone(sign * dt.timedelta(hours=int(hours), minutes=int(minutes)))


def parse_ts(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("timestamp without offset")
    return value


def local_date(ts: str | dt.datetime, tz: dt.timezone) -> dt.date:
    value = parse_ts(ts) if isinstance(ts, str) else ts
    return value.astimezone(tz).date()


def iso(ts: dt.datetime) -> str:
    return ts.isoformat(timespec="seconds")


def fx_id(date: dt.date | str, source: str, target: str) -> str:
    """Stable identifier of an fx.csv row (used as evidence id and get_record key)."""
    return f"FX:{date}:{source}:{target}"
