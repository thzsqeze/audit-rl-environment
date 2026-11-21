"""Submission schema and one-to-one matching against the closed truth.

Implements the published judge conventions ("Genesis FinTech: определения проверок", matcher section).

Structure (violations make the whole submit invalid -> reward -1)
  submission       object with "episode_id" (must equal the current episode) and "findings" (list, <= 30)
  finding          object with all of: type (str), affected_ids (list[str]), claimed_amount_minor (JSON
                   integer, not bool), currency (str), evidence_ids (list[str], <= 10 items counted
                   BEFORE de-duplication), explanation (str, <= 1000 Unicode code points)
  Limits are checked on the raw lists/text first; only then are lists turned into sets.

Normalisation (no other transformation is applied)
  ids / evidence ids  exact strings (case and whitespace significant); lists are treated as sets
  type, currency      exact string match with the documented values; an unknown type string is a
                      structurally valid finding that simply cannot match (FP)

Eligibility of finding f for truth issue t (an edge of the bipartite graph)
  f.type == t.type
  set(f.affected_ids) == set(t.affected_ids)        canonical ids per type: see policy.finding_conventions
  f.currency == t.currency
  |f.claimed_amount_minor - t.expected_amount_minor| <= tau   (tau = truth.amount_tolerance_minor, default 0)
  every evidence id of f exists in the episode's accessible records (else: not matchable)
  some declared evidence set E of t satisfies E <= set(f.evidence_ids)

Matching
  maximum-cardinality matching; among maximum matchings choose the lexicographically smallest
  sequence of (issue_id, finding_index) pairs, issues sorted by Unicode code points. Repeated findings
  stay separate vertices, so a repeat of a matched finding is a false positive.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
FINDING_KEYS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")


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


def _str_list(value: Any, name: str, max_len: int | None = None) -> frozenset[str]:
    if not isinstance(value, list):
        raise SubmissionError("invalid_submission", f"{name} must be a list of strings")
    if max_len is not None and len(value) > max_len:
        raise SubmissionError("invalid_submission", f"{name} allows at most {max_len} items")
    if not all(isinstance(item, str) for item in value):
        raise SubmissionError("invalid_submission", f"{name} items must be strings")
    return frozenset(value)


def parse_submission(submission: Any, episode_id: str) -> list[Finding]:
    """Validate findings.json and return findings (lists as sets). Raises SubmissionError."""
    if not isinstance(submission, dict) or "findings" not in submission or "episode_id" not in submission:
        raise SubmissionError("invalid_submission", "submission must be an object {episode_id, findings}")
    if submission["episode_id"] != episode_id:
        raise SubmissionError("invalid_submission", "episode_id does not match the current episode")
    findings = submission["findings"]
    if not isinstance(findings, list):
        raise SubmissionError("invalid_submission", "findings must be a list")
    if len(findings) > MAX_FINDINGS:
        raise SubmissionError("invalid_submission", f"at most {MAX_FINDINGS} findings are allowed")
    out = []
    for raw in findings:
        if not isinstance(raw, dict) or any(k not in raw for k in FINDING_KEYS):
            raise SubmissionError("invalid_submission", f"each finding requires {list(FINDING_KEYS)}")
        if not isinstance(raw["type"], str) or not isinstance(raw["currency"], str):
            raise SubmissionError("invalid_submission", "type and currency must be strings")
        amount = raw["claimed_amount_minor"]
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise SubmissionError("invalid_submission", "claimed_amount_minor must be a JSON integer")
        explanation = raw["explanation"]
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_CHARS:
            raise SubmissionError("invalid_submission",
                                  f"explanation must be a string of at most {MAX_EXPLANATION_CHARS} characters")
        out.append(Finding(
            type=raw["type"],
            affected_ids=_str_list(raw["affected_ids"], "affected_ids"),
            claimed_amount_minor=amount,
            currency=raw["currency"],
            evidence_ids=_str_list(raw["evidence_ids"], "evidence_ids", MAX_EVIDENCE_IDS),
            explanation=explanation,
        ))
    return out


def mismatch_reason(f: Finding, issue: dict[str, Any], tolerance: int, known_ids: set[str] | None) -> str | None:
    """None if the finding is an eligible match for the issue, otherwise the first failing criterion."""
    if f.type != issue["type"]:
        return "type"
    if f.affected_ids != frozenset(issue["affected_ids"]):
        return "affected_ids"
    if f.currency != issue["currency"]:
        return "currency"
    if abs(f.claimed_amount_minor - issue["expected_amount_minor"]) > tolerance:
        return "amount"
    if known_ids is not None and not f.evidence_ids <= known_ids:
        return "unknown_evidence"
    if not any(set(ev) <= f.evidence_ids for ev in issue["evidence_sets"]):
        return "evidence"
    return None


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[str, int]]  # (issue_id, finding index), sorted
    finding_status: list[str] = field(default_factory=list)  # per finding: "tp", "duplicate" or a reason
    missed: list[str] = field(default_factory=list)


def _max_matching(edges: dict[int, list[int]], issues: Iterable[int], findings_free: set[int]) -> int:
    """Kuhn's algorithm restricted to the given issue and finding vertices."""
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges.get(i, []):
            if j not in findings_free or j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in issues)


def match(findings: list[Finding], truth: dict[str, Any], known_ids: set[str] | None = None) -> MatchResult:
    issues = sorted(truth["issues"], key=lambda t: t["issue_id"])
    tol = int(truth.get("amount_tolerance_minor", 0))
    reasons = [[mismatch_reason(f, t, tol, known_ids) for t in issues] for f in findings]
    edges = {i: [j for j in range(len(findings)) if reasons[j][i] is None] for i in range(len(issues))}
    all_findings = set(range(len(findings)))
    size = _max_matching(edges, range(len(issues)), all_findings)

    # lexicographically smallest maximum matching: fix pairs greedily, keep feasibility of `size`
    pairs: list[tuple[int, int]] = []
    free = set(all_findings)
    for i in range(len(issues)):
        if len(pairs) == size:
            break
        for j in sorted(edges[i]):
            if j not in free:
                continue
            rest = _max_matching(edges, range(i + 1, len(issues)), free - {j})
            if len(pairs) + 1 + rest == size:
                pairs.append((i, j))
                free.discard(j)
                break
    matched = {j: issues[i]["issue_id"] for i, j in pairs}
    status = []
    order = ["unknown_evidence", "evidence", "amount", "currency", "affected_ids", "type"]  # closest first
    for j in range(len(findings)):
        if j in matched:
            status.append("tp")
        elif any(r is None for r in reasons[j]):
            status.append("duplicate")
        else:
            status.append(min(reasons[j], key=order.index) if issues else "no_issue_in_books")
    tp = len(pairs)
    return MatchResult(
        tp=tp, fp=len(findings) - tp, fn=len(issues) - tp,
        pairs=[(issues[i]["issue_id"], j) for i, j in pairs], finding_status=status,
        missed=[t["issue_id"] for i, t in enumerate(issues) if i not in {p[0] for p in pairs}],
    )
