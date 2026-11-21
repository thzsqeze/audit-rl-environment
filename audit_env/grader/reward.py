"""Metrics and the shared terminal reward (section 05 of the brief)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from audit_env.grader.matcher import SubmissionError, match, parse_submission

STEP_COST = 0.002


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if tp + fn == 0:  # empty truth
        f1 = 1.0 if fp == 0 else 0.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def terminal_reward(f1: float, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); not rounded (judge tolerance is 1e-9 absolute)."""
    return max(-1.0, min(1.0, 2.0 * f1 - 1.0 - STEP_COST * inspection_steps))


@dataclass
class GradeReport:
    episode_id: str
    valid: bool
    error: str | None
    tp: int | None
    fp: int | None
    fn: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    inspection_steps: int
    reward: float
    n_findings: int
    finding_status: list[str]
    matched: list[tuple[str, int]]
    missed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def judge_view(self) -> dict[str, Any]:
        """Shape required by judge_adapter.score_episode."""
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "return": self.reward}


def grade(submission: Any, truth: dict[str, Any], inspection_steps: int,
          known_ids: set[str] | None = None) -> GradeReport:
    """Grade a findings.json payload. `submission=None` means "no answer" (truncation): empty findings.

    `known_ids` is the set of accessible evidence ids of the episode; evidence outside it is unmatchable.
    An invalid structure yields reward -1 and null metrics.
    """
    episode_id = truth["episode_id"]
    if submission is None:
        submission = {"episode_id": episode_id, "findings": []}
    try:
        findings = parse_submission(submission, episode_id)
    except SubmissionError as exc:
        return GradeReport(episode_id, False, f"{exc.code}: {exc.message}", None, None, None, None, None, None,
                           inspection_steps, -1.0, 0, [], [], [t["issue_id"] for t in truth["issues"]])
    result = match(findings, truth, known_ids)
    m = metrics(result.tp, result.fp, result.fn)
    return GradeReport(
        episode_id=episode_id, valid=True, error=None, tp=result.tp, fp=result.fp, fn=result.fn,
        precision=m["precision"], recall=m["recall"], f1=m["f1"], inspection_steps=inspection_steps,
        reward=terminal_reward(m["f1"], inspection_steps), n_findings=len(findings),
        finding_status=result.finding_status, matched=result.pairs, missed=result.missed,
    )
