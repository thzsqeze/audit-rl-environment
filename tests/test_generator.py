import json

import pytest

from audit_env.config import ERROR_TYPES, GeneratorConfig, load_config
from audit_env.generator import generate_episode
from audit_env.money import parse_ts
from audit_env.validate import validate_episode


@pytest.mark.parametrize("config", ["default", "train", "dev", "clean", "interacting", "smoke"])
def test_generated_episodes_pass_self_validation(config):
    for seed in range(12):
        ep = generate_episode(seed, config)
        assert validate_episode(ep) == [], (config, seed)


def test_same_seed_and_config_reproduce_exactly():
    a, b = generate_episode(42, "default"), generate_episode(42, "default")
    assert json.dumps([a.public(), a.truth], sort_keys=True) == json.dumps([b.public(), b.truth], sort_keys=True)
    assert generate_episode(43, "default").episode_id != a.episode_id


def test_clean_config_has_no_issues_but_has_distractors():
    for seed in range(5):
        ep = generate_episode(seed, "clean")
        assert ep.truth["issues"] == []
        assert ep.provenance["distractors"], "clean books still contain plausible look-alike cases"


def test_interacting_config_has_both_scenarios():
    ep = generate_episode(9, "interacting")
    scenarios = {t["scenario"] for t in ep.truth["issues"]}
    assert {"partial_fx_payment_boundary", "duplicate_with_misapplied_payment"} <= scenarios


def test_all_four_types_are_independently_configurable():
    for t in ERROR_TYPES:
        cfg = {"extends": "default", "clean_probability": 0.0, "scenario_counts": {},
               "error_counts": {x: ([1, 1] if x == t else [0, 0]) for x in ERROR_TYPES}}
        types = {i["type"] for s in range(4) for i in generate_episode(s, cfg).truth["issues"]}
        assert types == {t}


def test_episode_sizes_and_formats(error_episode):
    m = error_episode.manifest
    assert 20 <= m["record_counts"]["documents"] <= 200
    assert m["record_counts"]["ledger"] == len(error_episode.ledger) <= 2000
    assert len(error_episode.truth["issues"]) <= 12
    for row in error_episode.ledger:
        assert isinstance(row["debit_minor"], int) and isinstance(row["credit_minor"], int)
        assert parse_ts(row["timestamp"]).utcoffset() is not None
        assert isinstance(row["fx_rate"], str)
    for inv in error_episode.invoices:
        assert parse_ts(inv["issued_at"]).utcoffset() is not None
    for r in error_episode.fx:
        assert isinstance(r["rate"], str) and "e" not in r["rate"].lower()
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                "record_counts", "rounding"):
        assert key in m


def test_missing_references_are_explicit_null(error_episode):
    assert any(r["invoice_id"] is None for r in error_episode.ledger)
    for p in error_episode.payments:
        assert "invoice_refs" in p and "refund_of" in p


def test_public_records_do_not_contain_truth_or_seed(error_episode):
    text = json.dumps(error_episode.public())
    for marker in ("issue_id", "ISS-", "evidence_sets", "provenance", "injected", "seed", "variant"):
        assert marker not in text


def test_benign_transformation_preserves_answer_shape():
    base = load_config("interacting").to_dict()
    moved = dict(base, id_salt="other", shuffle_rows=True)
    a, b = generate_episode(5, base), generate_episode(5, moved)
    shape = lambda ep: sorted((t["type"], t["expected_amount_minor"], t["currency"]) for t in ep.truth["issues"])  # noqa: E731
    assert shape(a) == shape(b)
    assert not {r["row_id"] for r in a.ledger} & {r["row_id"] for r in b.ledger}
    assert validate_episode(b) == []


def test_config_rejects_unknown_keys_and_bad_ranges():
    with pytest.raises(ValueError):
        GeneratorConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        load_config({"extends": "default", "error_counts": {"duplicate_invoice": [3, 1]}})
    with pytest.raises(ValueError):
        load_config({"extends": "default", "max_issues": 13})


def test_truth_evidence_exists_and_is_minimal(error_episode):
    ids = {r["row_id"] for r in error_episode.ledger} | {d["doc_id"] for d in error_episode.invoices} | \
          {d["doc_id"] for d in error_episode.payments}
    for issue in error_episode.truth["issues"]:
        for ev in issue["evidence_sets"]:
            assert 1 <= len(ev) <= 10
            assert all(e in ids or e.startswith("FX:") for e in ev)
