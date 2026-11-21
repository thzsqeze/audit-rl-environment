import json

import pytest

from audit_env.env import MAX_STEPS, PAGE_LIMIT, AuditEnv
from tests.conftest import submit_action


def test_reset_returns_task_manifest_policy_budget(env):
    obs, info = env.reset(1, "default")
    assert set(obs) >= {"task", "manifest", "policy", "remaining_steps"}
    assert obs["remaining_steps"] == MAX_STEPS
    assert info["episode_id"] == obs["manifest"]["episode_id"]
    json.dumps([obs, info])


def test_step_contract_types(env):
    env.reset(1, "default")
    obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger"}})
    assert set(obs) == {"result", "error", "remaining_steps"} and obs["error"] is None
    assert reward == 0.0 and terminated is False and truncated is False and isinstance(info, dict)
    assert obs["remaining_steps"] == MAX_STEPS - 1
    assert set(obs["result"]) == {"records", "next_cursor"}


def test_pagination_is_capped_and_cursors_are_stable(env):
    env.reset(2, "default")
    seen, cursor = [], None
    while True:
        obs = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 50, "cursor": cursor}})[0]
        assert len(obs["result"]["records"]) <= PAGE_LIMIT
        seen += [r["row_id"] for r in obs["result"]["records"]]
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) > 100
    a = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": "ledger:50", "limit": 5}})[0]
    b = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": "ledger:50", "limit": 5}})[0]
    assert a["result"] == b["result"] and [r["row_id"] for r in a["result"]["records"]] == seen[50:55]
    bad = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 51}})[0]
    assert bad["result"] is None and bad["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize("action,code", [
    ({"name": "get_record", "args": {"table": "ledger", "id": "GL-NOPE"}}, "not_found"),
    ({"name": "get_document", "args": {"doc_id": "../truth.json"}}, "not_found"),
    ({"name": "rm", "args": {}}, "unknown_action"),
    ({"name": "list_records", "args": {"table": "truth"}}, "invalid_arguments"),
    ({"name": "list_records", "args": {"table": "ledger", "cursor": "invoices:0"}}, "invalid_cursor"),
    ({"name": "list_records", "args": {"table": "ledger", "sql": "select"}}, "invalid_arguments"),
    ({"name": "calculator", "args": {"expression": "__import__('os').system('id')"}}, "calculator_error"),
    ({"name": "calculator", "args": {"expression": "open('/etc/passwd').read()"}}, "calculator_error"),
    ("not an object", "invalid_action"),
    ({"tool": "get_record", "args": {}}, "invalid_action"),
    ({"name": "get_record", "args": {"table": "ledger", "id": "x"}, "debug": True}, "invalid_action"),
])
def test_errors_are_bounded_and_consume_a_step(env, action, code):
    env.reset(1, "default")
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["result"] is None and obs["error"]["code"] == code
    assert len(obs["error"]["message"]) <= 200 and "/" not in obs["error"]["message"].replace("+ - * /", "")
    assert obs["remaining_steps"] == MAX_STEPS - 1
    assert reward == 0.0 and not terminated and not truncated and info == {}


def test_get_record_document_and_calculator(env, error_episode):
    env.reset(9, "interacting")
    row, inv, fx = error_episode.ledger[0], error_episode.invoices[0], error_episode.fx[0]
    assert env.step({"name": "get_record", "args": {"table": "ledger", "id": row["row_id"]}})[0]["result"] == row
    assert env.step({"name": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"] == inv
    fid = f"FX:{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"
    assert env.step({"name": "get_record", "args": {"table": "fx", "id": fid}})[0]["result"] == fx
    assert env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})[0]["result"] == "0.3"


def test_error_does_not_break_access(env):
    env.reset(1, "default")
    first = env.step({"name": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    env.step({"name": "get_record", "args": {"table": "invoices", "id": "nope"}})
    again = env.step({"name": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    assert again["result"] == first["result"] and again["remaining_steps"] == MAX_STEPS - 3


def test_observations_are_copies(env):
    env.reset(1, "default")
    obs = env.step({"name": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    obs["result"]["records"][0]["total_minor"] = -1
    again = env.step({"name": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    assert again["result"]["records"][0]["total_minor"] != -1


def test_submit_terminates_and_further_steps_are_rejected(env):
    obs, info = env.reset(3, "clean")
    obs, reward, terminated, truncated, info = env.step(submit_action(info["episode_id"], []))
    assert obs["result"] == {"accepted": True} and terminated and not truncated and reward == 1.0
    report = env.grader_report()
    obs, reward, terminated, truncated, _ = env.step(submit_action("x", []))
    assert obs["error"]["code"] == "episode_finished" and reward == 0.0 and terminated
    assert env.grader_report() == report


def test_submit_as_action_100_is_accepted(env):
    _, info = env.reset(9, "interacting")
    for _ in range(MAX_STEPS - 1):
        assert not env.step({"name": "calculator", "args": {"expression": "1"}})[3]
    obs, reward, terminated, truncated, _ = env.step(submit_action(info["episode_id"], []))
    assert terminated and not truncated and obs["result"] == {"accepted": True}
    assert env.grader_report()["inspection_steps"] == MAX_STEPS - 1


def test_step_limit_truncates_with_empty_answer(env):
    env.reset(9, "interacting")
    for i in range(MAX_STEPS):
        obs, reward, terminated, truncated, _ = env.step({"name": "calculator", "args": {"expression": "1+1"}})
        assert truncated == (i == MAX_STEPS - 1)
    assert not terminated and obs["remaining_steps"] == 0
    report = env.grader_report()
    assert report["n_findings"] == 0 and report["tp"] == 0 and report["end_reason"] == "step_limit"
    assert reward == -1.0
    env.step(submit_action(report["episode_id"], []))
    assert env.grader_report() == report


def test_time_limit_truncates():
    now = [0.0]
    env = AuditEnv(mode="train", clock=lambda: now[0])
    env.reset(1, "default")
    now[0] = 61.0
    obs, reward, terminated, truncated, _ = env.step({"name": "list_records", "args": {"table": "ledger"}})
    assert truncated and not terminated and obs["result"] is None and obs["error"]["code"] == "time_limit"
    assert env.grader_report()["n_findings"] == 0


def test_closed_mode_hides_reward_and_diagnostics():
    env = AuditEnv(mode="closed")
    obs, info = env.reset(9, "interacting")
    with pytest.raises(RuntimeError):
        env.grader_report()
    env.step({"name": "list_records", "args": {"table": "ledger"}})
    obs, reward, terminated, truncated, info = env.step(submit_action(info["episode_id"], []))
    assert reward == 0 and isinstance(reward, float) and info == {} and terminated
    assert obs == {"result": {"accepted": True}, "error": None, "remaining_steps": MAX_STEPS - 2}
    assert env.grader_report()["reward"] < 0


def test_reset_isolates_episodes(env):
    env.reset(1, "default")
    env.step({"name": "calculator", "args": {"expression": "1"}})
    obs, _ = env.reset(1, "default")
    assert obs["remaining_steps"] == MAX_STEPS


def test_same_actions_reproduce_transitions_and_rewards():
    actions = [{"name": "list_records", "args": {"table": "payments"}},
               {"name": "get_record", "args": {"table": "ledger", "id": "missing"}},
               {"name": "calculator", "args": {"expression": "7/3"}}]

    def play():
        env = AuditEnv(mode="train")
        obs, info = env.reset(11, "dev")
        out = [obs, info]
        for a in actions + [submit_action(info["episode_id"], [])]:
            out.append(env.step(a))
        return json.dumps(out, sort_keys=True)

    assert play() == play()
