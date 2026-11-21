"""Submission schema, normalisation and one-to-one matching against the closed truth.

Published rules (also in DESIGN.md):

Normalisation
  type      -> stripped, lower-case; must be one of the four error types
  ids       -> stripped strings; duplicates inside one list are removed; compared as sets
  currency  -> stripped, upper-case ISO-like code (3 letters)
  amount    -> integer minor units (bool / float / string are schema errors)

Eligibility of finding f for truth issue t (all must hold)
  f.type == t.type
  set(f.affected_ids) == set(t.affected_ids)            (canonical ids, exact set)
  f.currency == t.currency
  |f.claimed_amount_minor - t.expected_amount_minor| <= amount_tolerance_minor  (default 1)
  some evidence set E of t satisfies E <= set(f.evidence_ids)

Matching
  maximum-cardinality bipartite matching (Kuhn's augmenting paths); findings are visited in
  submission order and issues in issue_id order, so the result is deterministic. Each truth issue
  can absorb at most one finding: repeated findings are false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from audit_env.config import ERROR_TYPES

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_AFFECTED_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_ID_CHARS = 64
MAX_AMOUNT = 10**15
FINDING_KEYS = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}
_CURRENCY = re.compile(r"^[A-Z]{3}$")


class SubmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message[:200]


@dataclass(frozen=True)
class Finding:
    type: str
    affected_ids: frozenset[str]
    claimed_amount_minor: int
    currency: str
    evidence_ids: frozenset[str]
    explanation: str


def _id_list(value: Any, name: str, max_len: int, min_len: int) -> frozenset[str]:
    if not isinstance(value, list) or not min_len <= len(value) <= max_len:
        raise SubmissionError("invalid_submission", f"{name} must be a list of {min_len}..{max_len} ids")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > MAX_ID_CHARS:
            raise SubmissionError("invalid_submission", f"{name} items must be non-empty strings")
        out.append(item.strip())
    return frozenset(out)


def parse_submission(submission: Any, episode_id: str) -> list[Finding]:
    """Validate findings.json and return normalised findings. Raises SubmissionError."""
    if not isinstance(submission, dict) or set(submission) != {"episode_id", "findings"}:
        raise SubmissionError("invalid_submission", "submission must be {episode_id, findings}")
    if submission["episode_id"] != episode_id:
        raise SubmissionError("invalid_submission", "episode_id does not match the current episode")
    findings = submission["findings"]
    if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:
        raise SubmissionError("invalid_submission", f"findings must be a list of at most {MAX_FINDINGS}")
    out = []
    for raw in findings:
        if not isinstance(raw, dict) or set(raw) != FINDING_KEYS:
            raise SubmissionError("invalid_submission", f"each finding must have exactly {sorted(FINDING_KEYS)}")
        ftype = raw["type"].strip().lower() if isinstance(raw["type"], str) else None
        if ftype not in ERROR_TYPES:
            raise SubmissionError("invalid_submission", f"type must be one of {list(ERROR_TYPES)}")
        amount = raw["claimed_amount_minor"]
        if isinstance(amount, bool) or not isinstance(amount, int) or not 0 <= amount < MAX_AMOUNT:
            raise SubmissionError("invalid_submission", "claimed_amount_minor must be a non-negative integer")
        currency = raw["currency"].strip().upper() if isinstance(raw["currency"], str) else ""
        if not _CURRENCY.match(currency):
            raise SubmissionError("invalid_submission", "currency must be a 3-letter code")
        if not isinstance(raw["explanation"], str) or len(raw["explanation"]) > MAX_EXPLANATION_CHARS:
            raise SubmissionError("invalid_submission", f"explanation must be a string <= {MAX_EXPLANATION_CHARS}")
        out.append(Finding(
            type=ftype,
            affected_ids=_id_list(raw["affected_ids"], "affected_ids", MAX_AFFECTED_IDS, 1),
            claimed_amount_minor=amount,
            currency=currency,
            evidence_ids=_id_list(raw["evidence_ids"], "evidence_ids", MAX_EVIDENCE_IDS, 0),
            explanation=raw["explanation"],
        ))
    return out


def mismatch_reason(f: Finding, issue: dict[str, Any], tolerance: int) -> str | None:
    """None if eligible, otherwise the first failing criterion (diagnostics, grader-side only)."""
    if f.type != issue["type"]:
        return "type"
    if f.affected_ids != frozenset(issue["affected_ids"]):
        return "affected_ids"
    if f.currency != issue["currency"]:
        return "currency"
    if abs(f.claimed_amount_minor - issue["expected_amount_minor"]) > tolerance:
        return "amount"
    if not any(set(ev) <= f.evidence_ids for ev in issue["evidence_sets"]):
        return "evidence"
    return None


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[int, str]]  # (finding index, issue_id)
    finding_status: list[str] = field(default_factory=list)  # per finding: "tp" or reason
    missed: list[str] = field(default_factory=list)  # unmatched issue ids


def match(findings: list[Finding], truth: dict[str, Any]) -> MatchResult:
    issues = sorted(truth["issues"], key=lambda t: t["issue_id"])
    tol = int(truth.get("amount_tolerance_minor", 1))
    edges: list[list[int]] = []
    for f in findings:
        edges.append([j for j, t in enumerate(issues) if mismatch_reason(f, t, tol) is None])
    owner: list[int | None] = [None] * len(issues)  # issue index -> finding index

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] is None or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(findings)):
        augment(i, set())
    matched = {fi: issues[j]["issue_id"] for j, fi in enumerate(owner) if fi is not None}
    status = []
    for i, f in enumerate(findings):
        if i in matched:
            status.append("tp")
        elif edges[i]:
            status.append("duplicate")
        else:
            reasons = [mismatch_reason(f, t, tol) for t in issues]
            order = ["evidence", "amount", "currency", "affected_ids", "type"]  # closest first
            status.append(min(reasons, key=order.index) if reasons else "no_issue_in_books")
    tp = len(matched)
    return MatchResult(
        tp=tp, fp=len(findings) - tp, fn=len(issues) - tp,
        pairs=sorted(matched.items()), finding_status=status,
        missed=[t["issue_id"] for j, t in enumerate(issues) if owner[j] is None],
    )
