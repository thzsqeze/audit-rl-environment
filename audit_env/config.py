"""Generator configuration: JSON-serialisable, versioned, validated (unknown keys are rejected)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from audit_env.rng import stable_hash

ERROR_TYPES: tuple[str, ...] = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
SCENARIOS: tuple[str, ...] = ("partial_fx_payment_boundary", "duplicate_with_misapplied_payment")
DISTRACTORS: tuple[str, ...] = (
    "recurring_invoice",  # legit repeated amounts with different service periods
    "split_payment",  # one invoice settled by two payments
    "multi_invoice_payment",  # one payment settling two invoices
    "double_payment_refund",  # customer paid twice, excess parked in 2200 and refunded
    "unidentified_payment",  # payment without remittance, correctly parked in 2900
    "boundary_timestamp",  # documents whose UTC date and local date fall in different periods
)
FX_VARIANTS: tuple[str, ...] = ("utc_date", "stale_rate")
ALLOCATION_VARIANTS: tuple[str, ...] = ("missing", "partial")
CUTOFF_VARIANTS: tuple[str, ...] = ("backdated", "utc_boundary")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _range(lo: int, hi: int) -> list[int]:
    return [lo, hi]


@dataclass
class GeneratorConfig:
    config_id: str = "custom"
    config_version: str = "1"
    # company shape
    base_currencies: list[str] = field(default_factory=lambda: ["USD", "EUR", "KZT", "GBP"])
    foreign_currencies: list[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY", "KWD", "KZT"])
    n_foreign_currencies: list[int] = field(default_factory=lambda: _range(1, 2))
    foreign_share: float = 0.35
    timezones: list[str] = field(default_factory=lambda: ["+05:00", "-05:00", "+09:00", "+01:00", "-03:00", "+05:30"])
    year: int = 2026
    months: list[int] = field(default_factory=lambda: list(range(1, 13)))
    n_customers: list[int] = field(default_factory=lambda: _range(6, 14))
    n_period_invoices: list[int] = field(default_factory=lambda: _range(30, 60))
    n_opening_invoices: list[int] = field(default_factory=lambda: _range(2, 6))
    n_next_period_invoices: list[int] = field(default_factory=lambda: _range(2, 4))
    # issues: inclusive [min, max] count per type / scenario
    error_counts: dict[str, list[int]] = field(default_factory=lambda: {t: _range(0, 2) for t in ERROR_TYPES})
    scenario_counts: dict[str, list[int]] = field(default_factory=lambda: {s: _range(0, 1) for s in SCENARIOS})
    clean_probability: float = 0.2
    max_issues: int = 12
    fx_variants: list[str] = field(default_factory=lambda: list(FX_VARIANTS))
    allocation_variants: list[str] = field(default_factory=lambda: list(ALLOCATION_VARIANTS))
    cutoff_variants: list[str] = field(default_factory=lambda: list(CUTOFF_VARIANTS))
    # distractors (correct but suspicious-looking cases)
    distractor_counts: dict[str, list[int]] = field(
        default_factory=lambda: {
            "recurring_invoice": _range(1, 2),
            "split_payment": _range(1, 3),
            "multi_invoice_payment": _range(0, 2),
            "double_payment_refund": _range(0, 1),
            "unidentified_payment": _range(0, 1),
            "boundary_timestamp": _range(1, 2),
        }
    )
    # benign transformations (paired checks): same hidden history, different ids / order
    id_salt: str = ""
    shuffle_rows: bool = False

    def validate(self) -> GeneratorConfig:
        from audit_env.money import CURRENCY_EXPONENT

        def check_range(name: str, value: list[int], lo: int = 0) -> None:
            if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value)):
                raise ValueError(f"{name} must be [min, max] integers")
            if not lo <= value[0] <= value[1]:
                raise ValueError(f"{name} must satisfy {lo} <= min <= max")

        for cur in self.base_currencies + self.foreign_currencies:
            if cur not in CURRENCY_EXPONENT:
                raise ValueError(f"unsupported currency {cur}")
        for name in ("n_foreign_currencies", "n_customers", "n_period_invoices", "n_opening_invoices",
                     "n_next_period_invoices"):
            check_range(name, getattr(self, name))
        if self.n_period_invoices[0] < 15 or self.n_customers[0] < 3:
            raise ValueError("need at least 15 period invoices and 3 customers")
        for table, allowed in ((self.error_counts, ERROR_TYPES), (self.scenario_counts, SCENARIOS),
                               (self.distractor_counts, DISTRACTORS)):
            for key, value in table.items():
                if key not in allowed:
                    raise ValueError(f"unknown key {key}")
                check_range(key, value)
        for values, allowed in ((self.fx_variants, FX_VARIANTS), (self.allocation_variants, ALLOCATION_VARIANTS),
                                (self.cutoff_variants, CUTOFF_VARIANTS)):
            if not values or any(v not in allowed for v in values):
                raise ValueError(f"variants must be a non-empty subset of {allowed}")
        if not 0 <= self.clean_probability <= 1 or not 0 <= self.foreign_share <= 1:
            raise ValueError("probabilities must be within [0, 1]")
        if not 0 <= self.max_issues <= 12:
            raise ValueError("max_issues must be within [0, 12]")
        for tz in self.timezones:
            if len(tz) != 6 or tz[0] not in "+-" or tz[3] != ":":
                raise ValueError("timezones are fixed offsets like +05:00")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of everything that affects the hidden history (excludes benign transformations)."""
        data = self.to_dict()
        data.pop("id_salt")
        data.pop("shuffle_rows")
        return stable_hash(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratorConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data).validate()


def load_config(spec: str | dict[str, Any] | GeneratorConfig | None) -> GeneratorConfig:
    """Accept a preset name (configs/<name>.json), a path, a dict or a GeneratorConfig."""
    if spec is None:
        return GeneratorConfig().validate()
    if isinstance(spec, GeneratorConfig):
        return spec.validate()
    if isinstance(spec, dict):
        base = spec.get("extends")
        data = dict(spec)
        if base:
            data.pop("extends")
            merged = load_config(base).to_dict()
            merged.update(data)
            data = merged
        return GeneratorConfig.from_dict(data)
    path = Path(spec)
    if not path.suffix:
        path = CONFIG_DIR / f"{spec}.json"
    with open(path, encoding="utf-8") as fh:
        return load_config(json.load(fh))
