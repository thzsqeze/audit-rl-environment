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

from audit_env.config import CONFIG_DIR, load_config
from audit_env.env import AuditEnv
from audit_env.generator import generate_episode
from audit_env.policies import BASELINES
from audit_env.runner import run_episode, summarize
from audit_env.validate import validate_episode
from audit_env.version import ENV_VERSION, SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent
# Observation text that must never appear (grader-only field names and paths).
LEAK_MARKERS = ("issue_id", "ISS-", "evidence_sets", "expected_amount", "provenance", "_grader", "truth",
                "causal_change", "injected", "\"variant\"", "\"scenario\"", "/workspaces", "/home/", "/app/")


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
    summary: dict[str, Any] = {"env_version": ENV_VERSION, "mode": args.mode, "policies": {}}
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as traj_fh, \
            open(out / "grades.jsonl", "w", encoding="utf-8") as grade_fh:
        for name in args.policies.split(","):
            reports = []
            for item in items:
                traj, report = run_episode(name, item["seed"], item["config"], args.mode)
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
            brief = {k: (v if k != "findings" else f"<{len(v)} findings>") for k, v in a["args"].items()}
            print(f"step {line['step']:>3}: {a['tool']} {brief} -> reward={line['reward']} "
                  f"terminated={line['terminated']} truncated={line['truncated']}")
        print(f"P={report['precision']:.2f} R={report['recall']:.2f} F1={report['f1']:.2f} "
              f"return={report['reward']} steps={report['steps']} statuses={report['finding_status']}")
    return 0


# ------------------------------------------------------------------------------------------ check

def _check_episodes(results: dict) -> None:
    items = load_split("train") + load_split("dev")
    items += [{"seed": s, "config": c} for c in ("default", "clean", "interacting") for s in range(5000, 5010)]
    failures = []
    for item in items:
        errors = validate_episode(generate_episode(item["seed"], item["config"]))
        if errors:
            failures.append({"seed": item["seed"], "config": str(item["config"]), "errors": errors[:5]})
    results["episode_validation"] = {"checked": len(items), "failures": failures, "passed": not failures}


def _eval_items() -> list[dict]:
    # mirrors the jury protocol: 20 episodes, 5 clean and 5 with interacting errors
    return ([{"seed": s, "config": "default", "kind": "mixed"} for s in range(7000, 7010)]
            + [{"seed": s, "config": "clean", "kind": "clean"} for s in range(7100, 7105)]
            + [{"seed": s, "config": "interacting", "kind": "interacting"} for s in range(7200, 7205)])


def _check_baselines(results: dict, out: Path) -> None:
    items = _eval_items()
    per_policy, leaks = {}, []
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as fh:
        for name in (*BASELINES, "rules_auditor"):
            reports = []
            for item in items:
                traj, report = run_episode(name, item["seed"], item["config"], "eval")
                report["kind"] = item["kind"]
                reports.append(report)
                for line in traj:
                    text = json.dumps(line["observation"])
                    hits = [m for m in LEAK_MARKERS if m in text]
                    if hits:
                        leaks.append({"policy": name, "seed": item["seed"], "step": line["step"], "markers": hits})
                    fh.write(json.dumps({"policy": name, **line}, sort_keys=True) + "\n")
            per_policy[name] = {"all": summarize(reports),
                                **{k: summarize([r for r in reports if r["kind"] == k])
                                   for k in ("mixed", "clean", "interacting")}}
    oracle = per_policy["oracle"]["all"]
    checks = {
        "oracle_max_reward": oracle["mean_reward"] == 1.0 and oracle["fp"] == 0 and oracle["fn"] == 0,
        "flag_all_negative": per_policy["flag_all"]["all"]["mean_reward"] < 0,
        "random_below_oracle": per_policy["random"]["all"]["mean_reward"] < 0.5,
        "no_invalid_baseline_submissions": all(p["all"]["invalid_submissions"] == 0 for p in per_policy.values()),
        "within_time_limit": all(p["all"]["max_wall_time_s"] < 60 for p in per_policy.values()),
        "no_observation_leaks": not leaks,
    }
    results["baselines"] = {"summary": per_policy, "checks": checks, "leaks": leaks[:20],
                            "passed": all(checks.values())}


def _replay(seed: int, config: Any, actions: list[dict]) -> list[Any]:
    env = AuditEnv(mode="train")
    obs, info = env.reset(seed, config)
    out: list[Any] = [obs, info]
    for a in actions:
        out.append(env.step(a))
    return out


def _check_reproducibility(results: dict) -> None:
    failures = []
    for item in _eval_items():
        seed, config = item["seed"], item["config"]
        a, b = generate_episode(seed, config), generate_episode(seed, config)
        if _digest([a.public(), a.truth]) != _digest([b.public(), b.truth]):
            failures.append({"seed": seed, "check": "regeneration differs"})
        traj, _ = run_episode("reconciler", seed, config, "train")
        actions = [line["action"] for line in traj[1:]]
        if _digest(_replay(seed, config, actions)) != _digest(_replay(seed, config, actions)):
            failures.append({"seed": seed, "check": "replay differs"})
        # benign transformation: new ids + shuffled ledger order, same hidden history
        cfg = load_config(config).to_dict()
        cfg.update(id_salt="paired-check", shuffle_rows=True)
        c = generate_episode(seed, cfg)
        if not validate_episode(c) == []:
            failures.append({"seed": seed, "check": "transformed episode invalid"})
        shape = lambda ep: sorted((t["type"], t["expected_amount_minor"], t["currency"]) for t in ep.truth["issues"])  # noqa: E731
        if shape(a) != shape(c) or a.episode_id == c.episode_id:
            failures.append({"seed": seed, "check": "transformation changed the answer shape"})
        if set(r["row_id"] for r in a.ledger) & set(r["row_id"] for r in c.ledger):
            failures.append({"seed": seed, "check": "transformation kept row ids"})
        for name in ("oracle", "rules_auditor"):
            r1 = run_episode(name, seed, config, "eval")[1]["reward"]
            r2 = run_episode(name, seed, cfg, "eval")[1]["reward"]
            if name == "oracle" and r1 != r2:
                failures.append({"seed": seed, "check": f"{name} reward changed {r1} -> {r2}"})
    results["reproducibility"] = {"pairs": len(_eval_items()), "failures": failures, "passed": not failures}


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
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    steps = [("tests", _check_tests), ("episode_validation", _check_episodes),
             ("baselines", lambda r: _check_baselines(r, out)), ("reproducibility", _check_reproducibility)]
    if args.skip_tests:
        steps = steps[1:]
    for name, fn in steps:
        t0 = time.time()
        fn(results)
        status = results[name]["passed"]
        print(f"[{'PASS' if status else ('SKIP' if status is None else 'FAIL')}] {name} ({time.time() - t0:.1f}s)")
    results["elapsed_s"] = round(time.time() - started, 2)
    (out / "check_report.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("\nBaseline diagnostics (20 eval episodes: 10 mixed, 5 clean, 5 interacting):")
    print(f"{'policy':<14}{'P':>7}{'R':>7}{'F1':>7}{'return':>9}{'steps':>7}{'time_s':>8}")
    for name, s in results["baselines"]["summary"].items():
        a = s["all"]
        print(f"{name:<14}{a['mean_precision']:>7.3f}{a['mean_recall']:>7.3f}{a['mean_f1']:>7.3f}"
              f"{a['mean_reward']:>9.3f}{a['mean_steps']:>7.1f}{a['mean_wall_time_s']:>8.3f}")
    print(f"\nreport: {out / 'check_report.json'}")
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

    p = sub.add_parser("generate", help="write episodes (public files + _grader/) to a directory")
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
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="single verification command (tests, data, reward, reproducibility)")
    p.add_argument("--out", default="reports")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("serve", help="JSON-lines stdio server (agent in a separate process)")
    p.add_argument("--mode", choices=["train", "eval"], default="eval")
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
