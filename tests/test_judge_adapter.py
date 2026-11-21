"""Judge protocol through judge_adapter.py: fixtures C (clean), E (one error), L (>=101 rows), U (unknown id)."""

import json

import pytest

import judge_adapter
from audit_env.generator import Episode
from audit_env.transform import transform
from tests.conftest import oracle_findings

FORBIDDEN = ("issue_id", "evidence_sets", "expected_amount", "provenance", "truth", "causal_change",
             "clean_books", '"seed"', '"variant"', '"scenario"')


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge")
    judge_adapter.generate_episode(101, {"team_config": "clean_basic"}, str(root / "C"))
    judge_adapter.generate_episode(102, "type_period_cutoff", str(root / "E"))
    return {k: str(root / k) for k in ("C", "E")}


def closed(d):
    return {"mode": "closed", "episode_dir": d}


def test_generate_writes_open_and_closed_files(dirs):
    import os
    files = set(os.listdir(dirs["E"]))
    assert {"manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv",
            "truth.json", "provenance.json"} <= files


def test_c01_c04_reset_and_first_step(dirs):
    env = judge_adapter.make_env()
    obs, info = env.reset(0, closed(dirs["C"]))
    json.dumps([obs, info])
    assert {"task", "manifest", "policy", "remaining_steps"} <= set(obs) and obs["remaining_steps"] == 100
    m, ep = obs["manifest"], Episode.load(dirs["C"])
    assert m["record_counts"]["ledger"] == len(ep.ledger) and m["record_counts"]["invoices"] == len(ep.invoices)
    obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger",
                                                                                          "cursor": None}})
    assert isinstance(terminated, bool) and isinstance(truncated, bool) and reward == 0
    assert set(obs) == {"result", "error", "remaining_steps"} and obs["remaining_steps"] == 99


def test_c05_c06_three_pages_no_gaps(dirs):
    env = judge_adapter.make_env()
    env.reset(0, closed(dirs["C"]))
    ep = Episode.load(dirs["C"])
    assert len(ep.ledger) >= 101
    ids, cursor, pages = [], None, 0
    while True:
        obs = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}})[0]
        pages += 1
        assert len(obs["result"]["records"]) <= 50
        ids += [r["row_id"] for r in obs["result"]["records"]]
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            break
    assert pages >= 3 and ids == [r["row_id"] for r in ep.ledger]


def test_c07_c14_records_documents_errors(dirs):
    env = judge_adapter.make_env()
    env.reset(0, closed(dirs["C"]))
    ep = Episode.load(dirs["C"])
    r1, d1, u = ep.ledger[0], ep.invoices[0], "U-" + "Z" * 12
    assert u not in ep.evidence_ids()
    assert env.step({"name": "get_record", "args": {"table": "ledger", "id": r1["row_id"]}})[0]["result"] == r1
    assert env.step({"name": "get_document", "args": {"doc_id": d1["doc_id"]}})[0]["result"] == d1
    assert env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})[0]["result"] == "0.3"
    bad = env.step({"name": "calculator", "args": {"expression": "open('truth.json')"}})[0]
    assert bad["error"]["code"] == "calculator_error" and bad["result"] is None
    for action in ({"name": "get_record", "args": {"table": "ledger", "id": u}},
                   {"name": "get_document", "args": {"doc_id": u}}, {"name": "shell", "args": {}}):
        obs = env.step(action)[0]
        assert obs["result"] is None and obs["error"]["code"] in ("not_found", "unknown_action")
    obs = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1["row_id"]}})[0]
    assert obs["result"] == r1 and obs["remaining_steps"] == 100 - 8


def test_c19_closed_mode_no_reward_or_diagnostics(dirs):
    env = judge_adapter.make_env()
    obs, info = env.reset(0, closed(dirs["E"]))
    for _ in range(3):
        assert env.step({"name": "list_records", "args": {"table": "invoices"}})[1] == 0
    findings = oracle_findings(Episode.load(dirs["E"]).truth)
    obs, reward, terminated, truncated, info = env.step(
        {"name": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": findings}}})
    assert reward == 0 and info == {} and obs["result"] == {"accepted": True} and terminated
    assert env.grader_report()["reward"] == pytest.approx(1 - 0.006)


def test_c20_no_closed_data_in_any_channel(dirs):
    for d in dirs.values():
        env = judge_adapter.make_env()
        obs, info = env.reset(0, closed(d))
        seen = [obs, info]
        for table in ("ledger", "invoices", "payments", "fx"):
            cursor = None
            while True:
                step = env.step({"name": "list_records", "args": {"table": table, "cursor": cursor}})
                seen.append(step)
                cursor = step[0]["result"]["next_cursor"]
                if cursor is None:
                    break
        seen.append(env.step({"name": "get_document", "args": {"doc_id": "truth.json"}}))
        text = json.dumps(seen)
        assert not [m for m in FORBIDDEN if m in text]
        assert d not in text


def test_score_episode_contract(dirs):
    ep = Episode.load(dirs["E"])
    good = {"episode_id": ep.episode_id, "findings": oracle_findings(ep.truth)}
    assert judge_adapter.score_episode(dirs["E"], good, 10) == {
        "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "return": pytest.approx(0.98)}
    bad = judge_adapter.score_episode(dirs["E"], {"episode_id": ep.episode_id, "findings": "x"}, 10)
    assert bad["return"] == -1.0 and bad["tp"] is None
    assert judge_adapter.score_episode(dirs["C"], {"episode_id": Episode.load(dirs["C"]).episode_id,
                                                   "findings": []}, 10)["return"] == pytest.approx(0.98)


@pytest.mark.parametrize("permute,rename", [(True, False), (False, True), (True, True)])
def test_paired_transformation_keeps_oracle_score(dirs, tmp_path, permute, rename):
    ep = Episode.load(dirs["E"])
    moved, mapping = transform(ep, seed=7, permute=permute, rename=rename)
    moved.write(tmp_path / "T")
    a = judge_adapter.score_episode(dirs["E"], {"episode_id": ep.episode_id, "findings": oracle_findings(ep.truth)}, 5)
    b = judge_adapter.score_episode(str(tmp_path / "T"), {"episode_id": ep.episode_id,
                                                          "findings": oracle_findings(moved.truth)}, 5)
    assert a == b and a["tp"] == 1
    if rename:
        assert not set(mapping) & Episode.load(tmp_path / "T").evidence_ids()
