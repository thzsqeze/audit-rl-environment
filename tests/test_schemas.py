"""JSON schemas in schemas/ match the data, the environment and the matcher (T1)."""

import copy
import json
import re
from pathlib import Path

import pytest

from audit_env.env import AuditEnv
from audit_env.generator import Episode
from audit_env.grader.matcher import SubmissionError, parse_submission
from audit_env.runner import run_episode
from audit_env.schema import SCHEMA_DIR, SCHEMA_NAMES, SchemaStore

REPO = Path(__file__).resolve().parent.parent
EPISODE_DIRS = sorted((REPO / "episodes").glob("*/ep-*"))
TRAJECTORIES = sorted((REPO / "artifacts").glob("*/trajectories.jsonl"))


@pytest.fixture(scope="module")
def store():
    return SchemaStore()


def _assert_valid(store, instance, name):
    errors = store.errors(instance, name)
    assert not errors, f"{name}: {errors[:3]}"


def test_every_schema_loads_with_supported_keywords_only(store):
    files = {p.name for p in SCHEMA_DIR.glob("*.schema.json")}
    assert {f"{n}.schema.json" for n in SCHEMA_NAMES} <= files
    for name in files:
        schema = store.load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == name


def test_unsupported_keyword_is_rejected(tmp_path):
    (tmp_path / "x.schema.json").write_text(json.dumps({"type": "object", "patternProperties": {}}))
    with pytest.raises(ValueError, match="unsupported"):
        SchemaStore(tmp_path).load("x")


def test_validator_follows_json_types(store):
    assert store.errors(True, "fx_row")  # not an object
    row = {"date": "2026-01-02", "source_currency": "EUR", "target_currency": "KZT", "rate": "1.5"}
    _assert_valid(store, row, "fx_row")
    finding = {"type": "t", "affected_ids": [], "claimed_amount_minor": True, "currency": "KZT", "evidence_ids": [],
               "explanation": ""}
    assert store.errors({"episode_id": "e", "findings": [finding]}, "findings")  # bool is not an integer


def test_committed_episode_set_is_complete():
    assert len(list((REPO / "episodes" / "train").glob("ep-*"))) == 10
    assert len(list((REPO / "episodes" / "dev").glob("ep-*"))) == 10


@pytest.mark.parametrize("directory", EPISODE_DIRS, ids=lambda p: p.name)
def test_committed_episodes_match_schemas(store, directory):
    ep = Episode.load(directory)
    _assert_valid(store, ep.manifest, "manifest")
    _assert_valid(store, ep.policy, "policy")
    _assert_valid(store, ep.truth, "truth")
    for name, rows in (("ledger_row", ep.ledger), ("invoice", ep.invoices), ("payment", ep.payments),
                       ("fx_row", ep.fx)):
        for row in rows:
            _assert_valid(store, row, name)
    counts = ep.manifest["record_counts"]
    assert counts == {"ledger": len(ep.ledger), "invoices": len(ep.invoices), "payments": len(ep.payments),
                      "fx": len(ep.fx), "documents": len(ep.invoices) + len(ep.payments)}


@pytest.mark.parametrize("path", TRAJECTORIES, ids=lambda p: p.parent.name)
def test_recorded_trajectories_match_schemas(store, path):
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert lines
    for record in lines:
        _assert_valid(store, record, "trajectory_line")
        if record["action"] is not None:  # baseline policies only send well-formed actions
            _assert_valid(store, record["action"], "action")


def test_trajectory_artifacts_exist():
    assert {p.parent.name for p in TRAJECTORIES} == {"train", "dev"}


def test_live_observations_of_every_tool_match_schemas(store):
    env = AuditEnv(mode="train")
    obs, _ = env.reset(9, "interacting")
    _assert_valid(store, obs, "observation_reset")
    ep = env._episode  # noqa: SLF001 - the test picks existing ids from the loaded episode
    pay, fx = ep.payments[0], ep.fx[0]
    valid_actions = [
        *({"name": "list_records", "args": {"table": t, "cursor": None, "limit": 50}}
          for t in ("ledger", "invoices", "payments", "fx")),
        {"name": "get_record", "args": {"table": "ledger", "id": ep.ledger[0]["row_id"]}},
        {"name": "get_record", "args": {"table": "invoices", "id": ep.invoices[0]["invoice_id"]}},
        {"name": "get_record", "args": {"table": "payments", "id": pay["payment_id"]}},
        {"name": "get_record", "args": {"table": "fx",
                                        "id": f"FX:{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"}},
        {"name": "get_document", "args": {"doc_id": pay["doc_id"]}},
        {"name": "get_document", "args": {"doc_id": "manifest"}},
        {"name": "get_document", "args": {"doc_id": "policy"}},
        {"name": "calculator", "args": {"expression": "0.1+0.2"}},
        {"name": "calculator", "args": {"expression": "1-3"}},
        {"name": "calculator", "args": {"expression": "open('x')"}},
        {"name": "get_record", "args": {"table": "ledger", "id": "no-such-id"}},
    ]
    invalid_actions = [{"name": "nope", "args": {}}, "not an object"]
    end = [{"name": "submit", "args": {"findings": {"episode_id": ep.episode_id, "findings": []}}},
           {"name": "list_records", "args": {"table": "ledger"}}]  # after the end
    for action in valid_actions + invalid_actions + end:
        if action in valid_actions or action is end[0]:
            _assert_valid(store, action, "action")
        else:
            assert action is end[1] or store.errors(action, "action")
        obs, *_ = env.step(action)
        _assert_valid(store, obs, "observation_step")


def test_oracle_findings_match_findings_schema(store):
    traj, report = run_episode("oracle", 9, "interacting", "closed")
    submit = traj[-1]["action"]
    assert submit["args"]["findings"]["findings"], "interacting episode must have issues"
    _assert_valid(store, submit["args"]["findings"], "findings")
    _assert_valid(store, submit, "action")
    assert report["valid"] and report["fn"] == 0


def test_error_codes_in_schema_match_environment(store):
    source = (REPO / "audit_env" / "env.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'EnvError\(\s*"([a-z_]+)"', source))
    schema = store.load("observation_step")
    declared = set(schema["properties"]["error"]["anyOf"][1]["properties"]["code"]["enum"])
    assert codes and codes == declared


# ------------------------------------------------------------------ negative cases: schemas really reject

GOOD_FINDING = {"type": "duplicate_invoice", "affected_ids": ["R1", "R2"], "claimed_amount_minor": 10000,
                "currency": "KZT", "evidence_ids": ["D1", "D2"], "explanation": "x"}


def _submission(**changes):
    finding = dict(GOOD_FINDING)
    for key, value in changes.items():
        if value is KeyError:
            finding.pop(key)
        else:
            finding[key] = value
    return {"episode_id": "ep-1", "findings": [finding]}


FINDINGS_CASES = [
    ("valid", {"episode_id": "ep-1", "findings": [GOOD_FINDING]}, True),
    ("empty", {"episode_id": "ep-1", "findings": []}, True),
    ("unknown type is structurally valid", _submission(type="mystery"), True),
    ("extra key in finding", {"episode_id": "ep-1", "findings": [{**GOOD_FINDING, "note": 1}]}, True),
    ("30 findings", {"episode_id": "ep-1", "findings": [GOOD_FINDING] * 30}, True),
    ("10 evidence ids", _submission(evidence_ids=["D1"] * 10), True),
    ("1000 code points", _submission(explanation="é" * 1000), True),
    ("31 findings", {"episode_id": "ep-1", "findings": [GOOD_FINDING] * 31}, False),
    ("11 evidence ids", _submission(evidence_ids=["D1"] * 11), False),
    ("1001 chars", _submission(explanation="a" * 1001), False),
    ("amount bool", _submission(claimed_amount_minor=True), False),
    ("amount string", _submission(claimed_amount_minor="10000"), False),
    ("amount float", _submission(claimed_amount_minor=10000.5), False),
    ("missing explanation", _submission(explanation=KeyError), False),
    ("affected id not a string", _submission(affected_ids=[1]), False),
    ("findings not a list", {"episode_id": "ep-1", "findings": {}}, False),
    ("missing episode_id", {"findings": []}, False),
    ("not an object", [], False),
]


@pytest.mark.parametrize("label,submission,valid", FINDINGS_CASES, ids=[c[0] for c in FINDINGS_CASES])
def test_findings_schema_agrees_with_matcher(store, label, submission, valid):
    assert (store.errors(submission, "findings") == []) is valid, label
    try:
        parse_submission(copy.deepcopy(submission), "ep-1")
        matcher_valid = True
    except SubmissionError:
        matcher_valid = False
    assert matcher_valid is valid, label


LEDGER_ROW = {"row_id": "GL-1", "journal_id": "JE-1", "timestamp": "2026-10-01T13:03:38+00:00", "account": "1100",
              "debit_minor": 100, "credit_minor": 0, "currency": "KZT", "entity_id": None, "invoice_id": "INV-1",
              "payment_id": None, "txn_currency": "KZT", "txn_amount_minor": 100, "fx_rate": "1"}

BAD_INSTANCES = [
    ("ledger_row", {**LEDGER_ROW, "debit_minor": 1.5}),
    ("ledger_row", {**LEDGER_ROW, "timestamp": "2026-10-01T13:03:38"}),  # no offset
    ("ledger_row", {**LEDGER_ROW, "fx_rate": 1.0}),  # rate must be a decimal string
    ("ledger_row", {**LEDGER_ROW, "invoice_id": ""}),  # missing reference must be null, not ""
    ("ledger_row", {k: v for k, v in LEDGER_ROW.items() if k != "payment_id"}),
    ("ledger_row", {**LEDGER_ROW, "seed": 7}),
    ("fx_row", {"date": "2026-01-02", "source_currency": "EUR", "target_currency": "KZT", "rate": "1e3"}),
    ("action", {"name": "list_records", "args": {"table": "ledger", "limit": 51}}),
    ("action", {"name": "list_records", "args": {"table": "truth"}}),
    ("action", {"name": "shell", "args": {"cmd": "ls"}}),
    ("observation_step", {"result": None, "error": {"code": "not_found", "message": "x" * 201}, "remaining_steps": 1}),
    ("observation_step", {"result": None, "error": None, "remaining_steps": 1, "truth": []}),
    ("observation_step", {"result": {"records": [{}] * 51, "next_cursor": None}, "error": None, "remaining_steps": 1}),
    ("observation_step", {"result": {"records": [{**LEDGER_ROW, "truth": 1}], "next_cursor": None}, "error": None,
                          "remaining_steps": 1}),  # page records are validated too
    ("fx_row", {"date": "2026-01-02\n", "source_currency": "EUR", "target_currency": "KZT", "rate": "1.5"}),
    ("fx_row", {"date": "2026-01-02", "source_currency": "EUR\n", "target_currency": "KZT", "rate": "1.5\n"}),
    ("trajectory_line", {"episode_id": "e", "step": 1, "action": None, "reward": float("nan"), "terminated": False,
                         "truncated": False, "observation": {"result": None, "error": None, "remaining_steps": 1}}),
    ("trajectory_line", {"episode_id": "e", "step": 1, "action": None, "reward": float("inf"), "terminated": False,
                         "truncated": False, "observation": {"result": None, "error": None, "remaining_steps": 1}}),
    ("findings", {"episode_id": "e", "findings": [{**GOOD_FINDING, "evidence_ids": ["D1", 2]}]}),
    ("findings", {"episode_id": "e", "findings": [{**GOOD_FINDING, "affected_ids": "R1"}]}),
    ("action", {"name": "list_records", "args": {"table": "ledger", "limit": 0}}),
    ("action", {"name": "list_records", "args": {"table": "ledger", "limit": True}}),
    ("action", {"name": "calculator", "args": {"expression": "1" * 301}}),
    ("trajectory_line", {"episode_id": "e", "step": 1, "action": None, "reward": 0.0, "terminated": False,
                         "observation": {"result": None, "error": None, "remaining_steps": 1}}),  # no truncated
    ("truth", {"episode_id": "e", "amount_tolerance_minor": 0, "issues": [
        {"issue_id": "ISS-01", "type": "new_type", "variant": "copy", "scenario": None, "affected_ids": ["A"],
         "expected_amount_minor": 1, "currency": "KZT", "evidence_sets": [["D1"]]}]}),
]


@pytest.mark.parametrize("name,instance", BAD_INSTANCES, ids=[f"{n}-{i}" for i, (n, _) in enumerate(BAD_INSTANCES)])
def test_schemas_reject_malformed_instances(store, name, instance):
    assert store.errors(instance, name)


def test_good_ledger_row_is_accepted(store):
    _assert_valid(store, LEDGER_ROW, "ledger_row")
