"""Experimental reward (T5): reported separately, shared reward untouched, section-05 properties still hold."""

import json
from pathlib import Path

import pytest

from audit_env.analyze import episode_index
from audit_env.generator import Episode, generate_episode
from audit_env.grader import experimental_reward, grade, metrics, terminal_reward
from tests.conftest import oracle_findings

REPO = Path(__file__).resolve().parent.parent


def test_formula_and_clean_books_fallback():
    assert experimental_reward(1.0, 50, 0) == terminal_reward(1.0, 50) == pytest.approx(0.9)
    assert experimental_reward(1.0, 50, 1) == terminal_reward(1.0, 50)
    assert experimental_reward(1.0, 50, 10) == pytest.approx(0.99)
    assert experimental_reward(0.0, 0, 5) == -1.0 and experimental_reward(1.0, 0, 5) == 1.0
    assert experimental_reward(0.0, 100, 1) == -1.0  # clipped


def test_shared_reward_unchanged_on_every_recorded_submission():
    """Regrade every submit recorded in artifacts/ and compare with the stored shared reward."""
    index = episode_index([REPO / "episodes"])
    checked = 0
    for split in ("train", "dev"):
        lines = [json.loads(x) for x in open(REPO / "artifacts" / split / "trajectories.jsonl", encoding="utf-8")]
        grades = [json.loads(x) for x in open(REPO / "artifacts" / split / "grades.jsonl", encoding="utf-8")]
        submits = {(x["policy"], x["episode_id"]): x["action"]["args"]["findings"] for x in lines
                   if isinstance(x["action"], dict) and x["action"].get("name") == "submit"}
        for g in grades:
            ep = Episode.load(index[g["episode_id"]])
            submission = submits.get((g["policy"], g["episode_id"]))
            report = grade(submission, ep.truth, g["inspection_steps"], ep.evidence_ids())
            assert report.reward == g["reward"], g  # exact, not approximate
            assert report.reward == terminal_reward(report.f1 or 0.0, g["inspection_steps"]) or not report.valid
            checked += 1
    assert checked >= 100


@pytest.fixture(scope="module")
def books():
    return generate_episode(9, "interacting"), generate_episode(3, "clean")


def _exp(ep, findings, steps=10):
    return grade({"episode_id": ep.episode_id, "findings": findings}, ep.truth, steps, ep.evidence_ids())


def test_full_correct_answer_beats_silence_on_erroneous_books(books):
    ep, _ = books
    assert _exp(ep, oracle_findings(ep.truth)).experimental_reward > _exp(ep, []).experimental_reward


def test_silence_beats_false_accusations_on_clean_books(books):
    _, clean = books
    inv = clean.invoices[0]
    accusation = [{"type": "duplicate_invoice", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": 1,
                   "currency": inv["currency"], "evidence_ids": [inv["doc_id"]], "explanation": "x"}]
    assert _exp(clean, []).experimental_reward > _exp(clean, accusation).experimental_reward


def test_duplicates_give_no_gain(books):
    ep, _ = books
    full = oracle_findings(ep.truth)
    assert _exp(ep, full + full[:1]).experimental_reward < _exp(ep, full).experimental_reward


def test_flagging_everything_scores_badly(books):
    ep, _ = books
    flagged = [{"type": t, "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                "currency": inv["currency"], "evidence_ids": [inv["doc_id"]], "explanation": "all"}
               for inv in ep.invoices[:7] for t in ("duplicate_invoice", "payment_allocation", "fx_conversion",
                                                    "period_cutoff")][:30]
    assert _exp(ep, flagged).experimental_reward < 0


def test_step_savings_only_slightly_separate_equal_answers(books):
    ep, _ = books
    full = oracle_findings(ep.truth)
    fast, slow = _exp(ep, full, 5).experimental_reward, _exp(ep, full, 95).experimental_reward
    assert fast > slow and fast - slow <= 0.18  # never more than the shared formula's 0.18


@pytest.mark.parametrize("n", [2, 7, 10, 30])
def test_complete_answer_beats_shorter_partial_answer(n):
    """DESIGN.md section 8: shared reward can prefer dropping a finding to save steps; experimental cannot."""
    full_f1 = metrics(n, 0, 0)["f1"]
    partial_f1 = metrics(n - 1, 0, 1)["f1"]
    for saved in range(0, 100):
        full_steps, partial_steps = 99, 99 - saved
        assert experimental_reward(full_f1, full_steps, n) > experimental_reward(partial_f1, partial_steps, n)
    if n >= 10:  # the limitation of the shared formula that motivated the experiment
        assert terminal_reward(full_f1, 99) < terminal_reward(partial_f1, 20)


def test_invalid_submission_is_minus_one_in_both_columns(books):
    ep, _ = books
    report = grade({"episode_id": ep.episode_id, "findings": "nope"}, ep.truth, 3, ep.evidence_ids())
    assert report.reward == report.experimental_reward == -1.0
