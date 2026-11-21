"""Process-isolated environment: JSON-lines over stdin/stdout.

The agent runs in another process and never shares memory with the grader. Requests:
  {"cmd": "reset", "seed": 3, "config": "dev"}
  {"cmd": "step", "action": {"name": "list_records", "args": {"table": "ledger"}}}
Responses mirror the Python API: {"observation", "info"} or
{"observation", "reward", "terminated", "truncated", "info"}; malformed requests yield {"error": ...}.
The grade report is written only to the evaluator log file (--grader-log), never to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO

from audit_env.env import AuditEnv

MAX_REQUEST_BYTES = 1_000_000


def serve(mode: str = "closed", grader_log: str | None = None, stdin: IO[str] = sys.stdin,
          stdout: IO[str] = sys.stdout) -> None:
    env = AuditEnv(mode=mode)
    log = open(Path(grader_log), "a", encoding="utf-8") if grader_log else None  # noqa: SIM115
    try:
        for line in stdin:
            if not line.strip():
                continue
            response: dict
            try:
                if len(line) > MAX_REQUEST_BYTES:
                    raise ValueError("request too large")
                req = json.loads(line)
                if req.get("cmd") == "reset":
                    config = req.get("config")
                    if isinstance(config, str) and not config.replace("_", "").isalnum():
                        raise ValueError("config must be a preset name or an object, not a path")
                    obs, info = env.reset(int(req["seed"]), config)
                    response = {"observation": obs, "info": info}
                elif req.get("cmd") == "step":
                    obs, reward, terminated, truncated, info = env.step(req.get("action"))
                    response = {"observation": obs, "reward": reward, "terminated": terminated,
                                "truncated": truncated, "info": info}
                    if (terminated or truncated) and log:
                        log.write(json.dumps(env.grader_report(), sort_keys=True) + "\n")
                        log.flush()
                else:
                    response = {"error": "unknown cmd; use reset or step"}
            except RuntimeError:
                response = {"error": "call reset before step"}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                response = {"error": "malformed request"}
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
    finally:
        if log:
            log.close()
