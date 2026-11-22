"""Measured failure analysis (T3): counts agree with the grader, attribution is right, output is stable."""

import json

import pytest

from audit_env.analyze import analyze, episode_index, episode_kind, render, write_report
from audit_env.generator import Episode, generate_episode
from audit_env.runner import run_episode


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis")
    eps = {"clean": generate_episode(3, "clean_recurring"), "mixed": generate_episode(9, "interacting")}
    for name, ep in eps.items():
        ep.write(root / "episodes" / name)
    run_dir = root / "run"
    run_dir.mkdir()
    with open(run_dir / "trajectories.jsonl", "w", encoding="utf-8") as tf, \
            open(run_dir / "grades.jsonl", "w", encoding="utf-8") as gf:
        for policy in ("reconciler", "flag_all", "oracle"):
            for name in eps:
                traj, report = run_episode(policy, 0, {"mode": "closed", "episode_dir": str(root / "episodes" / name)})
                for line in traj:
                    tf.write(json.dumps({"policy": policy, **line}) + "\n")
                gf.write(json.dumps(report) + "\n")
    return root, run_dir, eps


def test_counts_agree_with_grader(run):
    root, run_dir, _ = run
    result = analyze(run_dir, episode_index([root / "episodes"]))
    grades = [json.loads(line) for line in open(run_dir / "grades.jsonl", encoding="utf-8")]
    for policy in ("reconciler", "flag_all", "oracle"):
        mine = [g for g in grades if g["policy"] == policy]
        assert sum(n for (p, *_), n in result["fp"].items() if p == policy) == sum(g["fp"] for g in mine)
        assert sum(n for (p, *_), n in result["fn"].items() if p == policy) == sum(g["fn"] for g in mine)
        assert len(result["per_policy"][policy]) == 2
    assert not result["unknown_episodes"]
    assert not [k for k in result["fp"] if k[0] == "oracle"] and not [k for k in result["fn"] if k[0] == "oracle"]


def test_false_positive_on_a_distractor_is_attributed_to_it(run):
    root, run_dir, eps = run
    prov = Episode.load(root / "episodes" / "clean").provenance
    recurring = [d.split(":", 1)[1] for d in prov["distractors"] if d.startswith("recurring_invoice:")]
    assert recurring, "clean_recurring must contain recurring invoices"
    target = recurring[0]
    custom = root / "custom"
    custom.mkdir(exist_ok=True)
    ep_id = eps["clean"].episode_id
    finding = {"type": "duplicate_invoice", "affected_ids": [target], "claimed_amount_minor": 1, "currency": "KZT",
               "evidence_ids": [], "explanation": "x"}
    action = {"name": "submit", "args": {"findings": {"episode_id": ep_id, "findings": [finding]}}}
    (custom / "trajectories.jsonl").write_text(json.dumps({"policy": "p", "episode_id": ep_id, "step": 1,
                                                           "action": action, "observation": {}}) + "\n")
    (custom / "grades.jsonl").write_text(json.dumps({
        "policy": "p", "episode_id": ep_id, "finding_status": ["no_issue_in_books"], "missed": [], "tp": 0, "fp": 1,
        "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0, "steps": 1, "wall_time_s": 0.1,
        "truncated": False, "valid": True}) + "\n")
    result = analyze(custom, episode_index([root / "episodes"]))
    assert dict(result["fp"]) == {("p", "distractor:recurring_invoice", "duplicate_invoice"): 1}
    assert result["per_policy"]["p"][0]["kind"] == "clean"


def test_missed_issues_are_grouped_by_type_variant_scenario(run):
    root, run_dir, eps = run
    result = analyze(run_dir, episode_index([root / "episodes"]))
    truth = eps["mixed"].truth["issues"]
    flag_all_fn = {(t, v, s): n for (p, t, v, s), n in result["fn"].items() if p == "flag_all"}
    expected = {}
    for t in truth:
        key = (t["type"], t["variant"], t["scenario"] or "-")
        expected[key] = expected.get(key, 0) + 1
    assert flag_all_fn == expected  # flag_all never matches, so every issue is missed exactly once


def test_report_is_deterministic_and_complete(run, tmp_path):
    root, run_dir, _ = run
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    write_report(run_dir, [root / "episodes"], a, "T")
    write_report(run_dir, [root / "episodes"], b, "T")
    text = a.read_text(encoding="utf-8")
    assert text == b.read_text(encoding="utf-8")
    for section in ("## Policies", "## By episode kind", "## False positives by cause", "## Missed issues",
                    "## Actions"):
        assert section in text
    assert str(root) not in text and "seed" not in text.lower()


def test_episode_kind():
    assert episode_kind({"issues": []}) == "clean"
    assert episode_kind({"issues": [{"scenario": None}]}) == "single"
    assert episode_kind({"issues": [{"scenario": None}, {"scenario": "x"}]}) == "interacting"


def test_render_handles_empty_run():
    text = render({"per_policy": {}, "fp": {}, "fn": {}, "actions": {}, "errors": {}, "unknown_episodes": []}, "T")
    assert "No false positives." in text and "No missed issues." in text
