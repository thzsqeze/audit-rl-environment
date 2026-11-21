import json

import pytest

from audit_env.env import MAX_STEPS, PAGE_LIMIT, AuditEnv


def test_reset_returns_task_manifest_policy_budget(env):
    obs, info = env.reset(1, "default")
    assert set(obs) >= {"task", "manifest", "policy", "tools", "steps_remaining"}
    assert obs["steps_remaining"] == MAX_STEPS
    assert info["episode_id"] == obs["manifest"]["episode_id"]
    json.dumps(obs)  # JSON-compatible


def test_step_contract_types(env):
    env.reset(1, "default")
    obs, reward, terminated, truncated, info = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert isinstance(obs, dict) and reward == 0.0 and terminated is False and truncated is False
    assert isinstance(info, dict)
    assert obs["ok"] and obs["steps_remaining"] == MAX_STEPS - 1


def test_pagination_is_capped_and_cursors_are_stable(env):
    env.reset(2, "default")
    seen, cursor = [], None
    while True:
        args = {"table": "ledger", "limit": 50}
        if cursor:
            args["cursor"] = cursor
        obs = env.step({"tool": "list_records", "args": args})[0]
        assert len(obs["result"]["rows"]) <= PAGE_LIMIT
        seen += [r["row_id"] for r in obs["result"]["rows"]]
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == obs["result"]["total"]
    again = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger:50", "limit": 5}})[0]
    assert [r["row_id"] for r in again["result"]["rows"]] == seen[50:55]
    bad = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})[0]
    assert not bad["ok"] and bad["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize("action,code", [
    ({"tool": "get_record", "args": {"table": "ledger", "id": "GL-NOPE"}}, "not_found"),
    ({"tool": "get_document", "args": {"doc_id": "../_grader/truth.json"}}, "not_found"),
    ({"tool": "rm", "args": {}}, "unknown_action"),
    ({"tool": "list_records", "args": {"table": "truth"}}, "invalid_arguments"),
    ({"tool": "list_records", "args": {"table": "ledger", "cursor": "invoices:0"}}, "invalid_cursor"),
    ({"tool": "list_records", "args": {"table": "ledger", "sql": "select"}}, "invalid_arguments"),
    ({"tool": "calculator", "args": {"expression": "__import__('os').system('id')"}}, "calculator_error"),
    ("not an object", "invalid_action"),
    ({"tool": "get_record", "args": {"table": "ledger", "id": "x"}, "debug": True}, "invalid_action"),
])
def test_errors_are_bounded_and_consume_a_step(env, action, code):
    env.reset(1, "default")
    obs, reward, terminated, truncated, _ = env.step(action)
    assert not obs["ok"] and obs["error"]["code"] == code
    assert len(obs["error"]["message"]) <= 200
    assert obs["steps_remaining"] == MAX_STEPS - 1
    assert reward == 0.0 and not terminated and not truncated
    assert "/" not in obs["error"]["message"].replace("+ - * /", "")


def test_get_record_document_and_calculator(env, error_episode):
    env.reset(9, "interacting")
    row = error_episode.ledger[0]
    inv = error_episode.invoices[0]
    assert env.step({"tool": "get_record", "args": {"table": "ledger", "id": row["row_id"]}})[0]["result"]["record"] == row
    assert env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"]["document"] == inv
    fx = error_episode.fx[0]
    fid = f"FX:{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"
    assert env.step({"tool": "get_record", "args": {"table": "fx", "id": fid}})[0]["result"]["record"] == fx
    assert env.step({"tool": "calculator", "args": {"expression": "2*(3+4)"}})[0]["result"]["value"] == "14"


def test_observations_are_copies(env):
    env.reset(1, "default")
    obs = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    obs["result"]["rows"][0]["total_minor"] = -1
    again = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]
    assert again["result"]["rows"][0]["total_minor"] != -1


def test_submit_terminates_and_further_steps_are_rejected(env):
    obs, info = env.reset(3, "clean")
    obs, reward, terminated, truncated, info = env.step(
        {"tool": "submit", "args": {"episode_id": info["episode_id"], "findings": []}})
    assert terminated and not truncated and reward == 1.0
    obs, reward, terminated, truncated, _ = env.step(
        {"tool": "submit", "args": {"episode_id": "x", "findings": []}})
    assert obs["error"]["code"] == "episode_finished" and reward == 0.0 and terminated


def test_step_limit_truncates_with_empty_answer():
    env = AuditEnv(mode="train")
    env.reset(9, "interacting")
    for i in range(MAX_STEPS):
        obs, reward, terminated, truncated, _ = env.step({"tool": "calculator", "args": {"expression": "1+1"}})
        if i < MAX_STEPS - 1:
            assert not truncated
    assert truncated and not terminated and obs["steps_remaining"] == 0
    report = env.grader_report()
    assert report["n_findings"] == 0 and report["tp"] == 0 and report["end_reason"] == "step_limit"
    assert reward == -1.0  # F1=0 on erroneous books, minus step cost, clipped


def test_time_limit_truncates():
    now = [0.0]
    env = AuditEnv(mode="train", clock=lambda: now[0])
    env.reset(1, "default")
    now[0] = 61.0
    obs, reward, terminated, truncated, _ = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert truncated and not terminated and obs["end_reason"] == "time_limit"
    assert "result" not in obs or "rows" not in obs.get("result", {})


def test_eval_mode_hides_reward_and_diagnostics():
    env = AuditEnv(mode="eval")
    obs, info = env.reset(9, "interacting")
    with pytest.raises(RuntimeError):
        env.grader_report()
    obs, reward, terminated, truncated, info = env.step(
        {"tool": "submit", "args": {"episode_id": info["episode_id"], "findings": []}})
    assert reward == 0.0 and info == {} and terminated
    assert "grade" not in json.dumps(obs) and "f1" not in json.dumps(obs)
    assert env.grader_report()["reward"] < 0


def test_reset_isolates_episodes(env):
    env.reset(1, "default")
    env.step({"tool": "calculator", "args": {"expression": "1"}})
    obs, _ = env.reset(1, "default")
    assert obs["steps_remaining"] == MAX_STEPS


def test_same_actions_reproduce_transitions_and_rewards():
    actions = [{"tool": "list_records", "args": {"table": "payments"}},
               {"tool": "get_record", "args": {"table": "ledger", "id": "missing"}},
               {"tool": "calculator", "args": {"expression": "7/3"}}]

    def play():
        env = AuditEnv(mode="train")
        obs, info = env.reset(11, "dev")
        out = [obs, info]
        for a in actions + [{"tool": "submit", "args": {"episode_id": info["episode_id"], "findings": []}}]:
            out.append(env.step(a))
        return json.dumps(out, sort_keys=True)

    assert play() == play()
