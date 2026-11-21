"""Gymnasium-style JSON environment: reset(seed, config) -> (obs, info); step(action) -> 5-tuple.

Actions are JSON objects {"tool": <name>, "args": {...}}:
  list_records(table, cursor=None, limit=50)   table in ledger|invoices|payments|fx, <=50 rows/page
  get_record(table, id)                        ledger: row_id, invoices: invoice_id, payments: payment_id,
                                               fx: "FX:<date>:<source>:<target>"
  get_document(doc_id)                         invoice/payment document, or "manifest" / "policy"
  calculator(expression)                       decimal arithmetic only: numbers + - * / ( )
  submit(episode_id, findings)                 findings.json; ends the episode (terminated=True)

Modes:
  train - the terminal reward (and grading summary) is returned to the policy;
  eval  - the policy always receives reward 0.0 and an empty info; the grade report stays with the
          evaluator (`grader_report()`), available only after the episode has ended.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from audit_env.calculator import CalculatorError, evaluate
from audit_env.generator import Episode, generate_episode
from audit_env.grader import grade
from audit_env.money import fx_id
from audit_env.version import ENV_VERSION, SCHEMA_VERSION

MAX_STEPS = 100
PAGE_LIMIT = 50
TIME_LIMIT_S = 60.0
TABLES = ("ledger", "invoices", "payments", "fx")
TOOLS: dict[str, dict[str, Any]] = {
    "list_records": {"args": {"table": "ledger|invoices|payments|fx", "cursor": "string|null (optional)",
                              "limit": f"1..{PAGE_LIMIT} (optional, default {PAGE_LIMIT})"}},
    "get_record": {"args": {"table": "ledger|invoices|payments|fx", "id": "string"}},
    "get_document": {"args": {"doc_id": "string (DOC-..., or 'manifest' / 'policy')"}},
    "calculator": {"args": {"expression": "string, numbers and + - * / ( ) only"}},
    "submit": {"args": {"episode_id": "string", "findings": "list (see task)"}},
}
_ALLOWED_ARGS = {
    "list_records": ({"table"}, {"cursor", "limit"}),
    "get_record": ({"table", "id"}, set()),
    "get_document": ({"doc_id"}, set()),
    "calculator": ({"expression"}, set()),
    "submit": ({"episode_id", "findings"}, set()),
}

TASK = (
    "You are auditing the books of one fictional company for one reporting period. Inspect the records "
    "with the tools and submit findings with evidence. Issue types: duplicate_invoice, payment_allocation, "
    "fx_conversion, period_cutoff (meaning, canonical affected_ids and amount conventions are in "
    "policy.finding_conventions). The books may be clean: correct-but-unusual cases (split or combined "
    "payments, refunds, recurring amounts, timezone edges) are not issues. Each finding: {type, affected_ids, "
    "claimed_amount_minor, currency, evidence_ids, explanation}; at most 30 findings, 10 evidence ids and "
    "1000 explanation characters per finding. Submit {episode_id, findings} with the submit tool. "
    "Reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); false and repeated findings cost precision."
)


class EnvError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message[:200]


def _copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


class AuditEnv:
    def __init__(self, mode: str = "train", max_steps: int = MAX_STEPS, time_limit_s: float = TIME_LIMIT_S,
                 clock: Callable[[], float] = time.monotonic):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._clock = clock
        self._episode: Episode | None = None
        self._report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ public API
    def reset(self, seed: int, config: Any = None, options: dict | None = None) -> tuple[dict, dict]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        episode = generate_episode(seed, config)
        self._load(episode)
        obs = {
            "event": "reset",
            "task": TASK,
            "manifest": _copy(episode.manifest),
            "policy": _copy(episode.policy),
            "tools": _copy(TOOLS),
            "max_steps": self.max_steps,
            "steps_remaining": self.max_steps,
        }
        info = {"episode_id": episode.episode_id, "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
                "mode": self.mode}
        return obs, info

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if self._episode is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            obs = self._error_obs(None, "episode_finished", "episode already ended; call reset()")
            return obs, 0.0, self._terminated, self._truncated, {}
        self._steps += 1
        tool = action.get("tool") if isinstance(action, dict) else None
        if self._clock() - self._t0 > self.time_limit_s:
            return self._finish(None, tool if isinstance(tool, str) else None, truncated=True,
                                reason="time_limit")
        try:
            tool, args = self._parse_action(action)
            if tool == "submit":
                return self._finish(args, tool, truncated=False)
            result = getattr(self, f"_tool_{tool}")(**args)
            obs = {"event": "step", "ok": True, "tool": tool, "result": result,
                   "steps_remaining": self.max_steps - self._steps}
        except EnvError as exc:
            obs = self._error_obs(tool if isinstance(tool, str) else None, exc.code, exc.message)
        if self._steps >= self.max_steps:
            return self._finish(None, obs.get("tool"), truncated=True, reason="step_limit", last_obs=obs)
        return obs, 0.0, False, False, {}

    def grader_report(self) -> dict[str, Any]:
        """Evaluator-side grade of the finished episode (never part of an observation)."""
        if self._report is None:
            raise RuntimeError("episode not finished")
        return _copy(self._report)

    @property
    def episode_id(self) -> str | None:
        return self._episode.episode_id if self._episode else None

    # ------------------------------------------------------------------ internals
    def _load(self, episode: Episode) -> None:
        self._episode = episode
        self._tables = {name: getattr(episode, name) for name in TABLES}
        self._by_id = {
            "ledger": {r["row_id"]: r for r in episode.ledger},
            "invoices": {r["invoice_id"]: r for r in episode.invoices},
            "payments": {r["payment_id"]: r for r in episode.payments},
            "fx": {fx_id(r["date"], r["source_currency"], r["target_currency"]): r for r in episode.fx},
        }
        self._docs = {d["doc_id"]: d for d in episode.invoices + episode.payments}
        self._docs["manifest"] = episode.manifest
        self._docs["policy"] = episode.policy
        self._steps = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._report = None
        self._t0 = self._clock()

    def _parse_action(self, action: Any) -> tuple[str, dict]:
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
            raise EnvError("invalid_action", "action must be an object {tool, args}")
        if set(action) - {"tool", "args"}:
            raise EnvError("invalid_action", "action may only contain 'tool' and 'args'")
        tool = action["tool"]
        if tool not in TOOLS:
            raise EnvError("unknown_action", f"unknown tool; available: {sorted(TOOLS)}")
        args = action.get("args", {})
        if not isinstance(args, dict):
            raise EnvError("invalid_arguments", "args must be an object")
        required, optional = _ALLOWED_ARGS[tool]
        if not required <= set(args) or set(args) - required - optional:
            raise EnvError("invalid_arguments", f"{tool} expects {sorted(required)} (+ optional {sorted(optional)})")
        return tool, args

    def _table(self, table: Any) -> str:
        if table not in TABLES:
            raise EnvError("invalid_arguments", f"table must be one of {list(TABLES)}")
        return table

    def _tool_list_records(self, table: Any, cursor: Any = None, limit: Any = PAGE_LIMIT) -> dict:
        rows = self._tables[self._table(table)]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= PAGE_LIMIT:
            raise EnvError("invalid_arguments", f"limit must be an integer in 1..{PAGE_LIMIT}")
        offset = 0
        if cursor is not None:
            if not isinstance(cursor, str) or not cursor.startswith(f"{table}:") or \
                    not cursor.split(":", 1)[1].isdigit():
                raise EnvError("invalid_cursor", "cursor is not valid for this table")
            offset = int(cursor.split(":", 1)[1])
            if offset > len(rows):
                raise EnvError("invalid_cursor", "cursor is out of range")
        page = rows[offset:offset + limit]
        nxt = offset + len(page)
        return {"table": table, "rows": _copy(page), "total": len(rows),
                "next_cursor": f"{table}:{nxt}" if nxt < len(rows) else None}

    def _tool_get_record(self, table: Any, id: Any) -> dict:  # noqa: A002 - tool argument name
        index = self._by_id[self._table(table)]
        if not isinstance(id, str) or id not in index:
            raise EnvError("not_found", "no record with this id in this table")
        return {"table": table, "record": _copy(index[id])}

    def _tool_get_document(self, doc_id: Any) -> dict:
        if not isinstance(doc_id, str) or doc_id not in self._docs:
            raise EnvError("not_found", "no document with this doc_id")
        return {"doc_id": doc_id, "document": _copy(self._docs[doc_id])}

    def _tool_calculator(self, expression: Any) -> dict:
        try:
            return {"expression": expression, "value": evaluate(expression)}
        except CalculatorError as exc:
            raise EnvError("calculator_error", str(exc)) from None

    def _error_obs(self, tool: str | None, code: str, message: str) -> dict:
        return {"event": "step", "ok": False, "tool": tool, "error": {"code": code, "message": message},
                "steps_remaining": max(self.max_steps - getattr(self, "_steps", 0), 0)}

    def _finish(self, submission: Any, tool: str | None, truncated: bool, reason: str = "submitted",
                last_obs: dict | None = None) -> tuple[dict, float, bool, bool, dict]:
        inspection_steps = self._steps if truncated else self._steps - 1
        report = grade(None if truncated else submission, self._episode.truth, inspection_steps)
        self._done, self._terminated, self._truncated = True, not truncated, truncated
        self._report = {**report.to_dict(), "terminated": not truncated, "truncated": truncated,
                        "end_reason": reason if report.valid else "invalid_submission",
                        "elapsed_s": round(self._clock() - self._t0, 4), "mode": self.mode}
        obs = dict(last_obs) if last_obs else {"event": "step", "ok": report.valid or truncated, "tool": tool}
        obs.update({"event": "end", "end_reason": reason, "steps_remaining": self.max_steps - self._steps})
        if not truncated:
            obs["result"] = {"accepted": report.valid}
            if not report.valid and self.mode == "train":
                obs["error"] = {"code": "invalid_submission", "message": report.error or ""}
        if self.mode == "train":
            info = {"grade": {k: self._report[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                                           "reward", "inspection_steps", "valid")}}
            return obs, report.reward, not truncated, truncated, info
        return obs, 0.0, not truncated, truncated, {}
