"""Command line: generate | validate | run | check | serve | demo.

`audit-env check` is the single verification command used in the container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from audit_env.config import CONFIG_DIR
from audit_env.env import AuditEnv
from audit_env.generator import generate_episode
from audit_env.policies import BASELINES
from audit_env.runner import run_episode, summarize
from audit_env.validate import validate_episode
from audit_env.version import ENV_VERSION, SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent
# Observation text that must never appear (grader-only field names and paths).
LEAK_MARKERS = ("issue_id", "ISS-", "evidence_sets", "expected_amount", "provenance", "truth", "clean_books",
                "causal_change", "injected", "\"variant\"", "\"scenario\"", "\"seed\"", "/workspaces", "/home/",
                "/app/", "/tmp/")


def load_split(name: str) -> list[dict[str, Any]]:
    with open(CONFIG_DIR / "splits.json", encoding="utf-8") as fh:
        return json.load(fh)[name]


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------------------------ commands

def cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    items = load_split(args.split) if args.split else [{"seed": s, "config": args.config} for s in args.seeds]
    index = []
    for item in items:
        ep = generate_episode(item["seed"], item["config"])
        errors = validate_episode(ep)
        if errors:
            print(f"seed {item['seed']}: INVALID {errors[:3]}", file=sys.stderr)
            return 1
        ep.write(out / ep.episode_id, include_closed=not args.public_only)
        index.append({"episode_id": ep.episode_id, "seed": item["seed"], "config": item["config"],
                      "issues": len(ep.truth["issues"]), "ledger_rows": len(ep.ledger),
                      "documents": ep.manifest["record_counts"]["documents"]})
        print(f"{ep.episode_id}: seed={item['seed']} issues={len(ep.truth['issues'])} rows={len(ep.ledger)}")
    (out / "index.json").write_text(json.dumps({"env_version": ENV_VERSION, "episodes": index}, indent=2) + "\n")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bad = 0
    for seed in args.seeds:
        errors = validate_episode(generate_episode(seed, args.config))
        if errors:
            bad += 1
            print(f"seed {seed}: {errors[:5]}")
    print(f"{len(args.seeds) - bad}/{len(args.seeds)} episodes valid")
    return 1 if bad else 0


def cmd_run(args: argparse.Namespace) -> int:
    items = load_split(args.split) if args.split else [{"seed": s, "config": args.config} for s in args.seeds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"env_version": ENV_VERSION, "mode": args.mode, "isolated": args.isolated,
                               "policies": {}}
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as traj_fh, \
            open(out / "grades.jsonl", "w", encoding="utf-8") as grade_fh:
        for name in args.policies.split(","):
            reports = []
            for item in items:
                traj, report = run_episode(name, item["seed"], item["config"], args.mode, isolated=args.isolated)
                reports.append(report)
                for line in traj:
                    traj_fh.write(json.dumps({"policy": name, **line}, sort_keys=True) + "\n")
                grade_fh.write(json.dumps(report, sort_keys=True) + "\n")
            summary["policies"][name] = summarize(reports)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["policies"], indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from audit_env.server import serve

    serve(mode=args.mode, grader_log=args.grader_log)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    env = AuditEnv(mode="train")
    for name in ("reconciler", "rules_auditor"):
        traj, report = run_episode(name, args.seed, args.config, "train", env=env)
        print(f"\n=== policy={name} episode={report['episode_id']} ===")
        for line in traj[1:]:
            a = line["action"]
            brief = {k: (v if k != "findings" else f"<{len(v['findings'])} findings>") for k, v in a["args"].items()}
            print(f"step {line['step']:>3}: {a['name']} {brief} -> reward={line['reward']} "
                  f"terminated={line['terminated']} truncated={line['truncated']}")
        print(f"P={report['precision']:.2f} R={report['recall']:.2f} F1={report['f1']:.2f} "
              f"return={report['reward']} steps={report['steps']} statuses={report['finding_status']}")
    return 0


# ------------------------------------------------------------------------------------------ check

def _check_episodes(results: dict) -> None:
    items = load_split("train") + load_split("dev")
    presets = sorted({c for _, c, _ in JURY_MATRIX} | {"default", "interacting"})
    items += [{"seed": s, "config": c} for c in presets for s in range(5000, 5005)]
    failures = []
    for item in items:
        errors = validate_episode(generate_episode(item["seed"], item["config"]))
        if errors:
            failures.append({"seed": item["seed"], "config": str(item["config"]), "errors": errors[:5]})
    results["episode_validation"] = {"checked": len(items), "failures": failures, "passed": not failures}


# Mirrors the published jury matrix D01-D20 (5 clean, 10 single-type, 5 interacting).
JURY_MATRIX = (
    [("D01", "clean_basic", "clean"), ("D02", "clean_split_payment", "clean"), ("D03", "clean_refund", "clean"),
     ("D04", "clean_recurring", "clean"), ("D05", "clean_boundary", "clean")]
    + [(f"D0{i}", "type_duplicate_invoice", "single") for i in (6, 7, 8)]
    + [("D09", "type_payment_allocation_missing", "single"), ("D10", "type_payment_allocation_partial", "single"),
       ("D11", "type_payment_allocation_missing", "single")]
    + [(f"D{i}", "type_fx_conversion", "single") for i in (12, 13)]
    + [(f"D{i}", "type_period_cutoff", "single") for i in (14, 15)]
    + [(f"D{i}", "scenario_partial_fx_payment_boundary", "interacting") for i in (16, 17, 18)]
    + [(f"D{i}", "scenario_duplicate_with_misapplied_payment", "interacting") for i in (19, 20)]
)
# P01-P05 permute, P06-P10 rename, P11-P20 both (published pair matrix)
PAIR_KIND = {i: ((True, False) if i <= 5 else (False, True) if i <= 10 else (True, True)) for i in range(1, 21)}


def _eval_items(base_seed: int = 7000) -> list[dict]:
    return [{"id": d, "seed": base_seed + i, "config": cfg, "kind": kind}
            for i, (d, cfg, kind) in enumerate(JURY_MATRIX, start=1)]


def _materialize(items: list[dict], root: Path) -> list[dict]:
    """Generate each episode into its own directory (judge protocol: episodes are loaded from files)."""
    out = []
    for item in items:
        directory = root / item["id"]
        generate_episode(item["seed"], item["config"]).write(directory)
        out.append({**item, "dir": str(directory), "env_config": {"mode": "closed", "episode_dir": str(directory)}})
    return out


def _check_baselines(results: dict, out: Path, items: list[dict]) -> None:
    per_policy, leaks = {}, []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as fh:
        for name in (*BASELINES, "rules_auditor"):
            reports = []
            for item in items:
                # closed run, policy in a separate locked-down process (jury "Граница доступа")
                traj, report = run_episode(name, item["seed"], item["env_config"], isolated=True)
                report["kind"] = item["kind"]
                reports.append(report)
                for line in traj:
                    text = json.dumps(line["observation"]) + json.dumps(line.get("info", {}))
                    hits = [m for m in LEAK_MARKERS if m in text]
                    if hits:
                        leaks.append({"policy": name, "episode": item["id"], "step": line["step"], "markers": hits})
                    if line["reward"] != 0.0:
                        leaks.append({"policy": name, "episode": item["id"], "step": line["step"],
                                      "markers": ["non-zero reward in closed mode"]})
                    fh.write(json.dumps({"policy": name, **line}, sort_keys=True) + "\n")
            per_policy[name] = {"all": summarize(reports),
                                **{k: summarize([r for r in reports if r["kind"] == k])
                                   for k in ("clean", "single", "interacting")}}
    oracle = per_policy["oracle"]["all"]
    checks = {
        "oracle_max_reward": oracle["mean_reward"] == 1.0 and oracle["fp"] == 0 and oracle["fn"] == 0,
        "flag_all_negative": per_policy["flag_all"]["all"]["mean_reward"] < 0,
        "random_below_oracle": per_policy["random"]["all"]["mean_reward"] < 0.5,
        "no_invalid_baseline_submissions": all(p["all"]["invalid_submissions"] == 0 for p in per_policy.values()),
        "within_time_limit": all(p["all"]["max_wall_time_s"] < 60 for p in per_policy.values()),
        "no_leaks_in_closed_mode": not leaks,
        "policies_ran_isolated": all(p["all"]["isolated"] == p["all"]["episodes"] for p in per_policy.values()),
        "no_policy_failures": all(p["all"]["truncated"] == 0 for p in per_policy.values()),
    }
    results["baselines"] = {"summary": per_policy, "checks": checks, "leaks": leaks[:20],
                            "passed": all(checks.values())}


def _check_schemas(results: dict, out: Path, items: list[dict]) -> None:
    """Validate the D01-D20 episode files and the closed-run trajectories against schemas/."""
    from audit_env.generator import Episode
    from audit_env.schema import SchemaStore

    store = SchemaStore()
    errors: list[str] = []

    def check(instance: Any, name: str, where: str) -> None:
        errs = store.errors(instance, name)
        if errs and len(errors) < 20:
            errors.append(f"{where}: {name}: {errs[0]}")

    checked = 0
    for item in items:
        ep = Episode.load(item["dir"])
        for name, value in (("manifest", ep.manifest), ("policy", ep.policy), ("truth", ep.truth)):
            check(value, name, item["id"])
        for name, rows in (("ledger_row", ep.ledger), ("invoice", ep.invoices), ("payment", ep.payments),
                           ("fx_row", ep.fx)):
            for row in rows:
                check(row, name, item["id"])
        checked += 1
    lines = 0
    with open(out / "trajectories.jsonl", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            record = json.loads(line)
            check(record, "trajectory_line", f"trajectories.jsonl:{n}")
            if record["action"] is not None:
                check(record["action"], "action", f"trajectories.jsonl:{n}")
            lines += 1
    results["schemas"] = {"episodes": checked, "trajectory_lines": lines, "errors": errors, "passed": not errors}


def _replay(config: Any, actions: list[dict]) -> str:
    env = AuditEnv(mode="train")
    out: list[Any] = list(env.reset(0, config))
    for a in actions:
        out.append(env.step(a))
    return _digest(out)


def _check_pairs(results: dict, items: list[dict], root: Path) -> None:
    from audit_env.generator import Episode
    from audit_env.transform import transform
    from audit_env.validate import truth_errors

    pairs = []
    for n, item in enumerate(items, start=1):
        permute, rename = PAIR_KIND[n]
        fails: list[str] = []
        # 1. same seed/config/version/actions reproduce records, transitions and rewards
        if _digest(generate_episode(item["seed"], item["config"]).public()) != \
                _digest(Episode.load(item["dir"]).public()):
            fails.append("regeneration differs from stored episode")
        traj, _ = run_episode("rules_auditor", item["seed"], {"mode": "train", "episode_dir": item["dir"]})
        actions = [line["action"] for line in traj[1:]]
        cfg = {"mode": "train", "episode_dir": item["dir"]}
        if _replay(cfg, actions) != _replay(cfg, actions):
            fails.append("replay differs")
        # 2-4. jury-style transformation on files
        original = Episode.load(item["dir"])
        moved, mapping = transform(original, seed=n, permute=permute, rename=rename)
        tdir = root / f"P{n:02d}"
        moved.write(tdir)
        if truth_errors(moved.public(), moved.truth):
            fails.append("transformed evidence not derivable/accessible")
        tcfg = {"mode": "closed", "episode_dir": str(tdir)}
        keys = ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps")
        for name in ("oracle", "rules_auditor"):
            r1 = run_episode(name, item["seed"], item["env_config"])
            r2 = run_episode(name, item["seed"], tcfg)
            if {k: r1[1][k] for k in keys} != {k: r2[1][k] for k in keys}:
                fails.append(f"{name} grade changed")
            if name == "oracle":
                inverse = {v: k for k, v in mapping.items()}
                back = [sorted((f["type"], tuple(sorted(inverse.get(x, x) for x in f["affected_ids"])),
                                f["claimed_amount_minor"], f["currency"],
                                tuple(sorted(inverse.get(x, x) for x in f["evidence_ids"])))
                               for f in r2[0][-1]["action"]["args"]["findings"]["findings"])]
                orig = [sorted((f["type"], tuple(sorted(f["affected_ids"])), f["claimed_amount_minor"], f["currency"],
                                tuple(sorted(f["evidence_ids"])))
                               for f in r1[0][-1]["action"]["args"]["findings"]["findings"])]
                if back != orig:
                    fails.append("oracle findings differ after inverse id mapping")
        pairs.append({"pair": f"P{n:02d}", "episode": item["id"], "permute": permute, "rename": rename,
                      "passed": not fails, "failures": fails})
    results["paired_checks"] = {"pairs": pairs, "passed_count": sum(p["passed"] for p in pairs),
                                "passed": all(p["passed"] for p in pairs)}


def _check_tests(results: dict) -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        results["tests"] = {"passed": None, "note": "pytest not installed; skipped"}
        return
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True,
                          text=True, cwd=REPO)
    tail = proc.stdout.strip().splitlines()[-1:] if proc.stdout else []
    results["tests"] = {"passed": proc.returncode == 0, "summary": tail}


def cmd_check(args: argparse.Namespace) -> int:
    import tempfile

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    with tempfile.TemporaryDirectory(prefix="audit-check-") as tmp:
        items = _materialize(_eval_items(), Path(tmp) / "episodes")
        results["jury_matrix"] = [{k: i[k] for k in ("id", "seed", "config", "kind")} for i in items]
        steps = [("tests", _check_tests), ("episode_validation", _check_episodes),
                 ("baselines", lambda r: _check_baselines(r, out, items)),
                 ("schemas", lambda r: _check_schemas(r, out, items)),
                 ("paired_checks", lambda r: _check_pairs(r, items, Path(tmp) / "pairs"))]
        if args.skip_tests:
            steps = steps[1:]
        for name, fn in steps:
            t0 = time.time()
            fn(results)
            status = results[name]["passed"]
            print(f"[{'PASS' if status else ('SKIP' if status is None else 'FAIL')}] {name} ({time.time() - t0:.1f}s)")
    results["elapsed_s"] = round(time.time() - started, 2)
    (out / "check_report.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("\nBaseline diagnostics, closed mode, jury-style matrix D01-D20 (5 clean, 10 single, 5 interacting):")
    print(f"{'policy':<14}{'P':>7}{'R':>7}{'F1':>7}{'return':>9}{'steps':>7}{'time_s':>8}")
    for name, s in results["baselines"]["summary"].items():
        a = s["all"]
        print(f"{name:<14}{a['mean_precision']:>7.3f}{a['mean_recall']:>7.3f}{a['mean_f1']:>7.3f}"
              f"{a['mean_reward']:>9.3f}{a['mean_steps']:>7.1f}{a['mean_wall_time_s']:>8.3f}")
    print(f"\npaired checks: {results['paired_checks']['passed_count']}/20 passed")
    print(f"report: {out / 'check_report.json'}")
    ok = all(results[n]["passed"] is not False for n, _ in steps)
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ main

def _seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit-env", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="write episodes (public files + truth.json/provenance.json)")
    p.add_argument("--split", choices=["train", "dev"])
    p.add_argument("--seeds", type=_seeds, default=[0])
    p.add_argument("--config", default="default")
    p.add_argument("--out", default="episodes")
    p.add_argument("--public-only", action="store_true", help="do not write closed grader files")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("validate", help="self-check generated episodes")
    p.add_argument("--seeds", type=_seeds, default=list(range(50)))
    p.add_argument("--config", default="default")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("run", help="run policies and record trajectories")
    p.add_argument("--policies", default=",".join(BASELINES))
    p.add_argument("--split", choices=["train", "dev"])
    p.add_argument("--seeds", type=_seeds, default=[0])
    p.add_argument("--config", default="default")
    p.add_argument("--mode", choices=["train", "closed"], default="train")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--isolated", action="store_true", help="run each policy in a separate locked-down process")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="single verification command (tests, data, reward, reproducibility)")
    p.add_argument("--out", default="reports")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("serve", help="JSON-lines stdio server (agent in a separate process)")
    p.add_argument("--mode", choices=["train", "closed"], default="closed")
    p.add_argument("--grader-log", default=None)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("demo", help="show a scripted policy interacting with one episode")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--config", default="interacting")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
