"""New distractors, error variants and large presets (T6, T7), checked against the records themselves."""

import json
import time
from collections import defaultdict

import pytest

from audit_env.env import AuditEnv
from audit_env.generator import generate_episode
from audit_env.money import local_date, parse_offset
from audit_env.rules import Books, detect_all
from audit_env.runner import run_episode
from audit_env.validate import validate_episode

SEEDS = range(12)


def _distractor_ids(ep, kind):
    return [d.split(":", 1)[1] for d in ep.provenance["distractors"] if d.startswith(kind + ":")]


@pytest.mark.parametrize("preset", ["clean_near_duplicate", "clean_partial_refund", "clean_next_period_receipt",
                                    "type_fx_conversion_due_date", "type_period_cutoff_receipt", "large"])
def test_new_presets_are_valid(preset):
    for seed in SEEDS:
        ep = generate_episode(seed, preset)
        assert validate_episode(ep) == [], (preset, seed)


def test_near_duplicate_is_a_different_order_with_the_same_amount():
    seen = 0
    for seed in SEEDS:
        ep = generate_episode(seed, "clean_near_duplicate")
        inv = {i["invoice_id"]: i for i in ep.invoices}
        for near_id in _distractor_ids(ep, "near_duplicate"):
            near = inv[near_id]
            twins = [i for i in ep.invoices if i["invoice_id"] != near_id and i["entity_id"] == near["entity_id"]
                     and i["currency"] == near["currency"] and i["total_minor"] == near["total_minor"]
                     and [(x["quantity"], x["unit_price_minor"]) for x in i["line_items"]]
                     == [(x["quantity"], x["unit_price_minor"]) for x in near["line_items"]]]
            assert twins, "a same-customer, same-amount, same-lines invoice exists"
            for twin in twins:  # ...but under a different purchase order
                assert [x["description"] for x in twin["line_items"]] != [x["description"] for x in near["line_items"]]
            seen += 1
        assert detect_all(ep.public()) == []  # rules see a different order, not a duplicate
    assert seen >= len(SEEDS)


def test_partial_refund_leaves_a_customer_credit():
    seen = 0
    for seed in SEEDS:
        ep = generate_episode(seed, "clean_partial_refund")
        pays = {p["payment_id"]: p for p in ep.payments}
        for pid in _distractor_ids(ep, "partial_refund"):
            refund = pays[pid]
            if refund["direction"] != "out":
                continue
            receipt = pays[refund["refund_of"]]
            assert 0 < refund["amount_minor"] < receipt["amount_minor"]
            credit = defaultdict(int)
            for r in ep.ledger:
                if r["account"] == "2200" and r["payment_id"] in (pid, receipt["payment_id"]):
                    credit[r["payment_id"]] += r["credit_minor"] - r["debit_minor"]
            assert sum(credit.values()) > 0, "part of the overpayment stays in 2200"
            seen += 1
        assert detect_all(ep.public()) == []
    assert seen >= len(SEEDS)


def test_next_period_receipt_is_published_but_not_booked_in_clean_books():
    seen = 0
    for seed in SEEDS:
        ep = generate_episode(seed, "clean_next_period_receipt")
        tz = parse_offset(ep.manifest["timezone"])
        booked = {r["payment_id"] for r in ep.ledger if r["payment_id"]}
        for pid in _distractor_ids(ep, "next_period_receipt"):
            pay = next(p for p in ep.payments if p["payment_id"] == pid)
            assert local_date(pay["timestamp"], tz).isoformat() > ep.manifest["period_end"]
            assert pid not in booked
            seen += 1
        assert detect_all(ep.public()) == []
    assert seen >= len(SEEDS)


def test_due_date_rate_error_uses_the_due_date_rate():
    for seed in SEEDS:
        ep = generate_episode(seed, "type_fx_conversion_due_date")
        [issue] = ep.truth["issues"]
        assert issue["type"] == "fx_conversion" and issue["variant"] == "due_date_rate"
        books = Books(ep.manifest, ep.policy, ep.ledger, ep.invoices, ep.payments, ep.fx)
        inv = books.inv[issue["affected_ids"][0]]
        ar = next(r for r in ep.ledger if r["invoice_id"] == inv["invoice_id"] and r["account"] == "1100"
                  and r["payment_id"] is None)
        due_rate, _ = books.rate(inv["currency"], books.local(inv["due_at"]))
        issue_rate, _ = books.rate(inv["currency"], books.local(inv["issued_at"]))
        assert ar["fx_rate"] == due_rate != issue_rate
        assert ar["debit_minor"] == books.conv(inv["total_minor"], inv["currency"], due_rate)
        assert issue["expected_amount_minor"] == abs(ar["debit_minor"] -
                                                     books.conv(inv["total_minor"], inv["currency"], issue_rate))


def test_receipt_cutoff_books_a_next_period_receipt():
    for seed in SEEDS:
        ep = generate_episode(seed, "type_period_cutoff_receipt")
        [issue] = ep.truth["issues"]
        assert issue["type"] == "period_cutoff" and issue["variant"] == "receipt_next_period"
        pay = next(p for p in ep.payments if p["payment_id"] == issue["affected_ids"][0])
        tz = parse_offset(ep.manifest["timezone"])
        assert local_date(pay["timestamp"], tz).isoformat() > ep.manifest["period_end"]
        rows = [r for r in ep.ledger if r["payment_id"] == pay["payment_id"]]
        assert rows and all(local_date(r["timestamp"], tz).isoformat() <= ep.manifest["period_end"] for r in rows)
        assert (issue["expected_amount_minor"], issue["currency"]) == (pay["amount_minor"], pay["currency"])
        assert not [r for r in ep.clean_twin["ledger"] if r["payment_id"] == pay["payment_id"]]
        assert pay["payment_id"] not in _distractor_ids(ep, "next_period_receipt")  # it is the error, not a distractor


def test_large_preset_is_at_the_top_of_the_suggested_ranges():
    for seed in range(5):
        ep = generate_episode(seed, "large")
        counts = ep.manifest["record_counts"]
        assert counts["documents"] <= 200 and counts["ledger"] >= 200
        assert 8 <= len(ep.truth["issues"]) <= 12


def test_stress_preset_reaches_two_thousand_ledger_rows_within_limits():
    ep = generate_episode(0, "stress")
    assert validate_episode(ep) == []
    assert 1500 <= ep.manifest["record_counts"]["ledger"] <= 2100
    started = time.perf_counter()
    for policy in ("random", "reconciler", "flag_all", "oracle", "rules_auditor"):
        _, report = run_episode(policy, 0, "stress", "closed", isolated=True)
        assert report["wall_time_s"] < 10 and not report["truncated"]
    assert time.perf_counter() - started < 60


@pytest.mark.parametrize("preset", ["clean_near_duplicate", "clean_partial_refund", "clean_next_period_receipt",
                                    "type_fx_conversion_due_date", "type_period_cutoff_receipt", "large"])
def test_paired_transformation_keeps_oracle_answer_on_new_presets(preset, tmp_path):
    from audit_env.generator import Episode
    from audit_env.transform import transform
    from audit_env.validate import truth_errors

    keys = ("tp", "fp", "fn", "f1", "reward", "inspection_steps")
    for seed in range(3):
        original = generate_episode(seed, preset)
        original.write(tmp_path / f"o{seed}")
        moved, mapping = transform(Episode.load(tmp_path / f"o{seed}"), seed=seed, permute=True, rename=True)
        moved.write(tmp_path / f"t{seed}")
        assert truth_errors(moved.public(), moved.truth) == []
        a = run_episode("oracle", seed, {"mode": "closed", "episode_dir": str(tmp_path / f"o{seed}")})[1]
        b = run_episode("oracle", seed, {"mode": "closed", "episode_dir": str(tmp_path / f"t{seed}")})[1]
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
        assert mapping, "ids were renamed"


MAX_OBSERVATION_BYTES = 64 * 1024


@pytest.mark.parametrize("preset", ["large", "stress"])
def test_observation_size_is_bounded(preset):
    env = AuditEnv(mode="closed")
    obs, _ = env.reset(0, preset)
    largest = len(json.dumps(obs).encode())
    for table in ("ledger", "invoices", "payments", "fx"):
        cursor = None
        while True:
            obs, *_ = env.step({"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})
            largest = max(largest, len(json.dumps(obs).encode()))
            cursor = obs["result"]["next_cursor"] if obs["result"] else None
            if cursor is None or obs["remaining_steps"] <= 1:
                break
        if obs["remaining_steps"] <= 1:
            break
    assert largest <= MAX_OBSERVATION_BYTES, largest
