"""Deterministic, process-independent RNG derivation (never relies on hash() or set order)."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def derive_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def make_rng(*parts: Any) -> random.Random:
    return random.Random(derive_seed(*parts))


def stable_hash(obj: Any, length: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
