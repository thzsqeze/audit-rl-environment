"""Run policies on episodes, record trajectories and aggregate evaluator-side diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from audit_env.env import AuditEnv
from audit_env.policies import POLICIES


def run_episode(policy_name: str, seed: int, config: Any = None, mode: str = "train",
                env: AuditEnv | None = None) -> tuple[list[dict], dict]:
    env = env or AuditEnv(mode=mode)
    policy = POLICIES[policy_name]()
    started = time.perf_counter()
    obs, info = env.reset(seed, config)
    # evaluator-only access to the truth, used exclusively by the oracle adapter
    truth = env._episode.truth if policy.needs_truth else None  # noqa: SLF001
    policy.reset(obs, info, seed, truth)
    episode_id = info["episode_id"]
    traj = [{"episode_id": episode_id, "step": 0, "action": None, "observation": obs, "reward": 0.0,
             "terminated": False, "truncated": False, "info": info}]
    step = 0
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1
        traj.append({"episode_id": episode_id, "step": step, "action": action, "observation": obs,
                     "reward": reward, "terminated": terminated, "truncated": truncated, "info": step_info})
        if terminated or truncated:
            break
    report = env.grader_report()
    report.update({"policy": policy_name, "seed": seed, "steps": step,
                   "wall_time_s": round(time.perf_counter() - started, 4)})
    return traj, report


def summarize(reports: list[dict]) -> dict[str, Any]:
    if not reports:
        return {}
    keys = ("precision", "recall", "f1", "reward", "steps", "wall_time_s")
    # invalid submissions have null metrics; they count as 0 in these diagnostic means
    out = {f"mean_{k}": round(mean((r[k] or 0) for r in reports), 4) for k in keys}
    out.update({
        "episodes": len(reports),
        "invalid_submissions": sum(not r["valid"] for r in reports),
        "truncated": sum(r["truncated"] for r in reports),
        "tp": sum(r["tp"] or 0 for r in reports), "fp": sum(r["fp"] or 0 for r in reports),
        "fn": sum(r["fn"] or 0 for r in reports),
        "max_wall_time_s": max(r["wall_time_s"] for r in reports),
    })
    return out


def run_suite(policies: list[str], seeds: list[int], config: Any = None, mode: str = "closed",
              out_dir: str | Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"mode": mode, "seeds": seeds, "policies": {}}
    out = Path(out_dir) if out_dir else None
    for name in policies:
        reports, lines = [], []
        for seed in seeds:
            traj, report = run_episode(name, seed, config, mode)
            reports.append(report)
            lines.extend(traj)
        summary["policies"][name] = summarize(reports)
        if out:
            pdir = out / name
            pdir.mkdir(parents=True, exist_ok=True)
            with open(pdir / "trajectories.jsonl", "w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(json.dumps(line, sort_keys=True) + "\n")
            with open(pdir / "grades.jsonl", "w", encoding="utf-8") as fh:
                for r in reports:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
