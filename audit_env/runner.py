"""Run policies on episodes, record trajectories and aggregate evaluator-side diagnostics.

Two ways to run a policy, with identical transitions and grades:

* in-process (default, fast, used by tests and the demo);
* isolated (`isolated=True`, `audit-env run --isolated`, the closed D01-D20 run of `audit-env check`): the
  policy runs in its own process (`audit_env.policy_worker`) with an empty working directory, a minimal
  environment and an audit hook that refuses file, process and network access. It receives only the
  observations (and, at reset, the public `info`); the environment, the episode files, the seed and the
  grader stay in this evaluator process. The oracle gets its findings from the evaluator, as in the jury
  protocol. A policy that hangs, crashes or answers garbage ends the episode as truncated with empty findings.

Policies never receive the generation seed of the episode: they get a separate `policy_seed` (default 0).
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

from audit_env.env import AuditEnv
from audit_env.policies import POLICIES

ISOLATION_GRACE_S = 0.25  # extra wait beyond the episode budget before a silent policy is stopped
MAX_REPLY_CHARS = 2_000_000  # one answer line; longer output is treated as a broken policy


class PolicyFailure(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def oracle_findings(truth: dict) -> list[dict]:
    """Evaluator side: correct findings built from the closed truth (first declared evidence set)."""
    return [{
        "type": t["type"], "affected_ids": list(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
        "currency": t["currency"], "evidence_ids": list(t["evidence_sets"][0]), "explanation": "oracle",
    } for t in truth["issues"]]


def policy_process_spec(policy_name: str, workdir: str) -> tuple[list[str], dict[str, str]]:
    """argv and environment of an isolated policy process: no episode path, seed or config anywhere."""
    if policy_name not in POLICIES:
        raise KeyError(policy_name)
    code_root = str(Path(__file__).resolve().parent.parent)  # location of the audit_env package (code only)
    bootstrap = (f"import sys; sys.path.insert(0, {code_root!r}); "
                 "from audit_env.policy_worker import main; sys.exit(main())")
    argv = [sys.executable, "-I", "-B", "-c", bootstrap, policy_name]
    env = {"PYTHONIOENCODING": "utf-8"}
    if os.name == "nt" and os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]  # required by the Windows CRT; not a data path
    return argv, env


class RemotePolicy:
    """Evaluator-side handle of a policy running in a separate process."""

    def __init__(self, policy_name: str):
        self.name = policy_name
        self.needs_truth = POLICIES[policy_name].needs_truth
        self.workdir = tempfile.mkdtemp(prefix="policy-")
        argv, env = policy_process_spec(policy_name, self.workdir)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=self.workdir, env=env, text=True, encoding="utf-8", bufsize=1)
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline(MAX_REPLY_CHARS)
            if not line:
                break
            if not line.endswith(chr(10)) and len(line) >= MAX_REPLY_CHARS:
                self._lines.put("too long")  # bounded memory: never buffer an endless line
                break
            self._lines.put(line)
        self._lines.put(None)

    def _write(self, data: str, errors: list) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            errors.append(exc)

    def _call(self, message: dict, timeout: float) -> dict:
        # a policy that stops reading would block a plain write once the pipe buffer is full, so the write
        # runs in a thread and shares the deadline with the answer
        deadline = time.monotonic() + timeout
        errors: list = []
        writer = threading.Thread(target=self._write, args=(json.dumps(message) + "\n", errors), daemon=True)
        writer.start()
        writer.join(timeout)
        if writer.is_alive():
            raise PolicyFailure("time_limit", "policy did not read its input in time")
        if errors:
            raise PolicyFailure("policy_error", "policy process is not accepting input")
        try:
            line = self._lines.get(timeout=max(deadline - time.monotonic(), 0.0))
        except queue.Empty:
            raise PolicyFailure("time_limit", "policy did not answer in time") from None
        if line is None:
            raise PolicyFailure("policy_error", "policy process exited")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            raise PolicyFailure("policy_error", "policy answered with invalid JSON") from None
        if not isinstance(reply, dict) or "error" in reply:
            detail = reply.get("error", "") if isinstance(reply, dict) else ""
            raise PolicyFailure("policy_error", str(detail)[:200])
        return reply

    def reset(self, obs: dict, info: dict, policy_seed: int, findings: list[dict] | None, timeout: float) -> None:
        message: dict[str, Any] = {"type": "reset", "observation": obs, "info": info, "policy_seed": policy_seed}
        if self.needs_truth:
            message["oracle_findings"] = findings
        self._call(message, timeout)

    def act(self, obs: dict, timeout: float) -> Any:
        reply = self._call({"type": "act", "observation": obs}, timeout)
        if "action" not in reply:
            raise PolicyFailure("policy_error", "reply without an action")
        return reply["action"]

    def close(self, failed: bool = False) -> None:
        """Stop the worker: EOF for a healthy one, kill for one that failed (it may not read its input)."""
        if self.proc.poll() is None:
            if failed:
                self.proc.kill()
            else:
                try:
                    if self.proc.stdin:
                        self.proc.stdin.close()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        try:  # drop unsent input of a killed worker (the pipe is broken, so this cannot block)
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        shutil.rmtree(self.workdir, ignore_errors=True)


def run_episode(policy_name: str, seed: int, config: Any = None, mode: str = "train",
                env: AuditEnv | None = None, isolated: bool = False,
                policy_seed: int = 0) -> tuple[list[dict], dict]:
    """Run one episode. `seed`/`config` go to the environment only; the policy gets `policy_seed`."""
    env = env or AuditEnv(mode=mode)
    remote = RemotePolicy(policy_name) if isolated else None
    policy = None if remote else POLICIES[policy_name]()
    started = time.perf_counter()
    failed = True  # until the episode loop has run
    try:
        obs, info = env.reset(seed, config)
        # evaluator-only: the oracle's findings come from the closed truth, never from a policy channel
        findings = oracle_findings(env._episode.truth) if POLICIES[policy_name].needs_truth else None  # noqa: SLF001
        episode_id = info["episode_id"]
        traj = [{"episode_id": episode_id, "step": 0, "action": None, "observation": obs, "reward": 0.0,
                 "terminated": False, "truncated": False, "info": info}]
        step = 0
        failed = False
        try:
            if remote:
                remote.reset(obs, info, policy_seed, findings, env.time_left() + ISOLATION_GRACE_S)
            else:
                policy.reset(obs, info, policy_seed, None)
                if policy.needs_truth:
                    policy.load_findings(findings)
            while True:
                action = remote.act(obs, env.time_left() + ISOLATION_GRACE_S) if remote else policy.act(obs)
                obs, reward, terminated, truncated, step_info = env.step(action)
                step += 1
                traj.append({"episode_id": episode_id, "step": step, "action": action, "observation": obs,
                             "reward": reward, "terminated": terminated, "truncated": truncated, "info": step_info})
                if terminated or truncated:
                    break
        except PolicyFailure as exc:
            failed = True
            obs, reward, terminated, truncated, step_info = env.abort(exc.reason)
            traj.append({"episode_id": episode_id, "step": step + 1, "action": None, "observation": obs,
                         "reward": reward, "terminated": terminated, "truncated": truncated, "info": step_info})
    finally:
        if remote:
            remote.close(failed=failed)
    report = env.grader_report()
    report.update({"policy": policy_name, "seed": seed, "steps": step, "isolated": isolated,
                   "wall_time_s": round(time.perf_counter() - started, 4)})
    return traj, report


def summarize(reports: list[dict]) -> dict[str, Any]:
    if not reports:
        return {}
    keys = ("precision", "recall", "f1", "reward", "experimental_reward", "steps", "wall_time_s")
    # invalid submissions have null metrics; they count as 0 in these diagnostic means
    out = {f"mean_{k}": round(mean((r[k] or 0) for r in reports), 4) for k in keys}
    out.update({
        "episodes": len(reports),
        "invalid_submissions": sum(not r["valid"] for r in reports),
        "truncated": sum(r["truncated"] for r in reports),
        "tp": sum(r["tp"] or 0 for r in reports), "fp": sum(r["fp"] or 0 for r in reports),
        "fn": sum(r["fn"] or 0 for r in reports),
        "max_wall_time_s": max(r["wall_time_s"] for r in reports),
        "isolated": sum(bool(r.get("isolated")) for r in reports),
    })
    return out


def run_suite(policies: list[str], seeds: list[int], config: Any = None, mode: str = "closed",
              out_dir: str | Path | None = None, isolated: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {"mode": mode, "seeds": seeds, "isolated": isolated, "policies": {}}
    out = Path(out_dir) if out_dir else None
    for name in policies:
        reports, lines = [], []
        for seed in seeds:
            traj, report = run_episode(name, seed, config, mode, isolated=isolated)
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
