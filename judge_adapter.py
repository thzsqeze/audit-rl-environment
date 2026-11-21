"""Judge adapter (Genesis FinTech check definitions, section "Единый способ подключения").

Entry points used by the jury inside the container (working directory: repository root):

    generate_episode(seed, config, output_dir)   -> writes manifest.json, policy.json, ledger.csv,
                                                    invoices.jsonl, payments.jsonl, fx.csv (open) and
                                                    truth.json, provenance.json (closed) into output_dir
    make_env()                                   -> object with reset(seed, config) / step(action)
    score_episode(episode_dir, findings, inspection_steps) -> {"tp","fp","fn","precision","recall","f1","return"}

reset(seed, config) accepts {"mode": "closed"|"train", "episode_dir": dir} (loads that episode exactly,
ids and order untouched) or {"mode": ..., "team_config": <config>} (team generator). Actions use the
envelope {"name": ..., "args": {...}}; submit args are {"findings": <findings.json>}. The native
environment already speaks this protocol, so the adapter is a thin pass-through.

`config` / `team_config` is a preset name from configs/ (e.g. "default", "clean", "type_fx_conversion")
or a dict of GeneratorConfig fields, optionally with "extends": "<preset>" (see README, "Конфигурации").
Invalid submissions are written to the evaluator log (env AUDIT_GRADER_LOG, default: stderr).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from audit_env.env import AuditEnv
from audit_env.generator import Episode
from audit_env.generator import generate_episode as _generate
from audit_env.grader import grade


def _team_config(config: Any) -> Any:
    if isinstance(config, dict) and "team_config" in config:
        return config["team_config"]
    if isinstance(config, dict) and not config:
        return None
    return config


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    _generate(seed, _team_config(config)).write(output_dir, include_closed=True)


def make_env() -> AuditEnv:
    return AuditEnv(mode="closed")


def _log(record: dict) -> None:
    path = os.environ.get("AUDIT_GRADER_LOG")
    line = json.dumps(record, sort_keys=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    else:
        print(line, file=sys.stderr)


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    episode = Episode.load(episode_dir)
    report = grade(findings, episode.truth, inspection_steps, episode.evidence_ids())
    if not report.valid:
        _log({"event": "invalid_submission", "episode_id": episode.episode_id, "error": report.error})
    return report.judge_view()
