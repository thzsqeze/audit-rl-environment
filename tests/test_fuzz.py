"""Action fuzzing (T8): random and malformed actions never crash the environment or leak closed data."""

import json
import random

import pytest

from audit_env.cli import LEAK_MARKERS
from audit_env.env import AuditEnv
from audit_env.schema import SchemaStore

N_ACTIONS = 2000
FUZZ_SEED = 2026
JUNK = [None, True, False, 0, -1, 1.5, 10**30, "", " ", "x" * 5000, "\u00b2", "\x00", "../truth.json",
        "/etc/passwd", "C:\\Windows\\win.ini", [], [1, 2], {}, {"a": {"b": {"c": [None]}}}, float("nan")]
TABLES = ["ledger", "invoices", "payments", "fx"]
EXPRESSIONS = ["0.1+0.2", "1/0", "(" * 40 + "1" + ")" * 40, "2*" * 200 + "2", "open('x')", "__import__('os')",
               "1e5", "9" * 400, "-(-(-1))", "1 +", "", "3 / 7", "7 - 10", "lambda: 1", "1;2", "0x10"]


def _ids(env):
    ep = env._episode  # noqa: SLF001 - the fuzzer mixes existing ids with junk
    return {
        "ledger": [r["row_id"] for r in ep.ledger], "invoices": [r["invoice_id"] for r in ep.invoices],
        "payments": [r["payment_id"] for r in ep.payments],
        "fx": [f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in ep.fx],
        "docs": [d["doc_id"] for d in ep.invoices + ep.payments] + ["manifest", "policy"],
    }


def random_action(rng: random.Random, ids: dict, episode_id: str):
    pick = rng.random
    junk = lambda: rng.choice(JUNK)  # noqa: E731
    roll = pick()
    if roll < 0.08:  # not an action envelope at all
        return rng.choice([None, 7, "list_records", [], {"name": 5}, {"args": {}}, {"name": "x", "args": {}, "z": 1}])
    if roll < 0.13:
        return {"name": rng.choice(["shell", "sql", "read_file", "LIST_RECORDS", "", "submit "]), "args": {}}
    name = rng.choice(["list_records", "get_record", "get_document", "calculator", "submit"])
    if pick() < 0.1:
        return {"name": name, "args": junk()}
    table = rng.choice(TABLES) if pick() < 0.8 else junk()
    if name == "list_records":
        args = {"table": table}
        if pick() < 0.6:
            args["cursor"] = rng.choice([None, f"{table}:{rng.randint(0, 400)}", f"{table}:-1", f"{table}:\u00b2",
                                         f"{table}:\u0663", "ledger:5", junk()])
        if pick() < 0.6:
            args["limit"] = rng.choice([1, 50, 51, 0, -3, True, 2.0, "10", rng.randint(1, 50)])
        if pick() < 0.05:
            args["extra"] = 1
        return {"name": name, "args": args}
    if name == "get_record":
        pool = ids.get(table, []) if isinstance(table, str) else []
        rid = rng.choice(pool) if pool and pick() < 0.6 else rng.choice(ids["docs"] + [junk()])
        args = {"table": table, "id": rid}
        if pick() < 0.05:
            args.pop("id")
        return {"name": name, "args": args}
    if name == "get_document":
        return {"name": name, "args": {"doc_id": rng.choice(ids["docs"]) if pick() < 0.6 else junk()}}
    if name == "calculator":
        return {"name": name, "args": {"expression": rng.choice(EXPRESSIONS) if pick() < 0.85 else junk()}}
    # submit: valid, invalid structure, or junk
    finding = {"type": rng.choice(["duplicate_invoice", "fx_conversion", "x", 1]),
               "affected_ids": rng.choice([[rng.choice(ids["invoices"])], [], "id", [None]]),
               "claimed_amount_minor": rng.choice([100, True, "100", 1.5, -5]), "currency": rng.choice(["KZT", 1]),
               "evidence_ids": rng.choice([[rng.choice(ids["docs"])], ["U"] * 11, None]),
               "explanation": rng.choice(["short", "e" * 1001])}
    findings = {"episode_id": rng.choice([episode_id, "other", None]),
                "findings": rng.choice([[finding], [finding] * 31, [], None, junk()])}
    return {"name": name, "args": {"findings": rng.choice([findings, junk()])}}


@pytest.mark.parametrize("mode", ["closed", "train"])
def test_two_thousand_random_actions_never_crash_or_leak(mode):
    rng = random.Random(f"{FUZZ_SEED}:{mode}")
    store = SchemaStore()
    env = AuditEnv(mode=mode)
    episode_seeds = iter(range(100, 10_000))
    stats = {"episodes": 0, "submitted": 0, "truncated": 0, "errors": 0}

    def new_episode():
        obs, info = env.reset(next(episode_seeds), rng.choice(["interacting", "clean", "default"]))
        assert not [m for m in LEAK_MARKERS if m in json.dumps(obs) + json.dumps(info)]
        stats["episodes"] += 1
        return obs, _ids(env), info["episode_id"]

    obs, ids, episode_id = new_episode()
    remaining = obs["remaining_steps"]
    for _ in range(N_ACTIONS):
        action = random_action(rng, ids, episode_id)
        obs, reward, terminated, truncated, info = env.step(action)  # must never raise
        text = json.dumps(obs, allow_nan=False) + json.dumps(info, allow_nan=False)
        assert not [m for m in LEAK_MARKERS if m in text], (action, text[:300])
        assert store.errors(obs, "observation_step") == [], (action, obs)
        assert isinstance(terminated, bool) and isinstance(truncated, bool) and not (terminated and truncated)
        assert obs["remaining_steps"] == remaining - 1, "every step, valid or not, costs exactly one step"
        remaining = obs["remaining_steps"]
        if obs["error"] is not None:
            stats["errors"] += 1
            assert len(obs["error"]["message"]) <= 200 and obs["result"] is None
        if mode == "closed":
            assert reward == 0.0 and info == {}
        elif not (terminated or truncated):
            assert reward == 0.0 and info == {}  # no shaping, no hints during the episode
        if remaining == 0 and not terminated:
            assert truncated, "100 actions without submit must truncate"
        if terminated or truncated:
            stats["submitted" if terminated else "truncated"] += 1
            report = env.grader_report()
            if truncated:
                assert report["n_findings"] == 0  # truncation grades an empty answer
            # after the end: no state change, no late findings
            late = env.step(random_action(rng, ids, episode_id))
            assert late[0]["error"]["code"] == "episode_finished" and late[1] == 0.0
            assert (late[2], late[3]) == (terminated, truncated) and env.grader_report() == report
            obs, ids, episode_id = new_episode()
            remaining = obs["remaining_steps"]
    # the fuzzer really exercised errors, submissions and truncations
    assert stats["episodes"] > 20 and stats["submitted"] > 10 and stats["errors"] > 500, stats


@pytest.mark.parametrize("cursor", ["ledger:\u00b2", "ledger:\u0663", "ledger:\uff11", "ledger:-1", "ledger:1.0",
                                    "ledger: 1", "ledger:", "invoices:0", "ledger:99999"])
def test_malformed_cursors_are_bounded_errors(cursor):
    env = AuditEnv(mode="closed")
    env.reset(3, "clean")
    obs, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": cursor}})
    assert obs["error"]["code"] == "invalid_cursor" and obs["remaining_steps"] == 99
    obs, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": "ledger:1", "limit": 1}})
    assert obs["error"] is None and len(obs["result"]["records"]) == 1
