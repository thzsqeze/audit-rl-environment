"""Gymnasium-style JSON environment: reset(seed, config) -> (obs, info); step(action) -> 5-tuple.

Action envelope (judge convention): {"name": <action>, "args": {...}}
  list_records  {"table", "cursor"?: null|str, "limit"?: 1..50}  -> {"records": [...], "next_cursor": null|str}
  get_record    {"table", "id"}                                  -> the record object
                ids: ledger row_id | invoices invoice_id | payments payment_id | fx "FX:<date>:<src>:<dst>"
  get_document  {"doc_id"}                                       -> the invoice/payment document object
                (also "manifest" and "policy")
  calculator    {"expression"}                                   -> decimal string, e.g. "0.3"
  submit        {"findings": <findings.json object>}             -> {"accepted": true}; ends the episode

Observations
  after reset: {"task", "manifest", "policy", "tools", "remaining_steps": 100}
  after step:  {"result": value|null, "error": null|{"code", "message"}, "remaining_steps": int}

Modes (set per reset through config["mode"] or by the constructor)
  train  - the terminal reward and a grade summary (info["grade"]) are returned to the policy;
  closed - reward is always 0 and info is {}; the grade stays evaluator-side (`grader_report()`).

Config passed to reset:
  {"mode": "closed"|"train", "episode_dir": <dir>}           load that episode exactly (judge protocol)
  {"mode": "closed"|"train", "team_config": <config>}        generate with the team generator
  <config> (preset name, dict or GeneratorConfig)            generate, constructor mode
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
MODES = {"train": "train", "closed": "closed", "eval": "closed"}
TOOLS: dict[str, dict[str, Any]] = {
    "list_records": {"args": {"table": "ledger|invoices|payments|fx", "cursor": "null|string (optional)",
                              "limit": f"1..{PAGE_LIMIT} (optional, default {PAGE_LIMIT})"},
                     "result": {"records": "list", "next_cursor": "null|string"}},
    "get_record": {"args": {"table": "ledger|invoices|payments|fx",
                            "id": "row_id | invoice_id | payment_id | FX:<date>:<source>:<target>"},
                   "result": "record object"},
    "get_document": {"args": {"doc_id": "doc_id of an invoice/payment, or 'manifest' / 'policy'"},
                     "result": "document object"},
    "calculator": {"args": {"expression": "numbers and + - * / ( ) only"}, "result": "decimal string"},
    "submit": {"args": {"findings": "{episode_id, findings: [...]}"}, "result": {"accepted": True}},
}
_ALLOWED_ARGS = {
    "list_records": ({"table"}, {"cursor", "limit"}),
    "get_record": ({"table", "id"}, set()),
    "get_document": ({"doc_id"}, set()),
    "calculator": ({"expression"}, set()),
    "submit": ({"findings"}, set()),
}

TASK = (
    "Audit the books of one fictional company for one reporting period. Inspect the records with the tools "
    "and submit findings with evidence. Issue types: duplicate_invoice, payment_allocation, fx_conversion, "
    "period_cutoff; their meaning, canonical affected_ids and amount conventions are in "
    "policy.finding_conventions. The books may be clean: correct-but-unusual cases (split or combined "
    "payments, full or partial refunds, recurring amounts, same-amount invoices for different purchase orders, next-period receipts that are not booked, timezone edges) are not issues. Submit "
    "{\"findings\": {\"episode_id\": ..., \"findings\": [{type, affected_ids, claimed_amount_minor, currency, "
    "evidence_ids, explanation}]}}: at most 30 findings, 10 evidence ids and 1000 explanation characters each. "
    "Ids, types and currencies are compared exactly; amounts are exact integers in minor units. "
    "Reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)."
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
        if mode not in MODES:
            raise ValueError("mode must be 'train' or 'closed'")
        self.default_mode = MODES[mode]
        self.mode = self.default_mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._clock = clock
        self._episode: Episode | None = None
        self._report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ public API
    def reset(self, seed: int, config: Any = None, options: dict | None = None) -> tuple[dict, dict]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.mode = self.default_mode
        if isinstance(config, dict) and config.get("mode") in MODES and \
                ("episode_dir" in config or "team_config" in config):
            self.mode = MODES[config["mode"]]
            if config.get("episode_dir"):
                episode = Episode.load(config["episode_dir"])
            else:
                episode = generate_episode(seed, config.get("team_config"))
        else:
            episode = generate_episode(seed, config)
        self._load(episode)
        obs = {
            "task": TASK,
            "manifest": _copy(episode.manifest),
            "policy": _copy(episode.policy),
            "tools": _copy(TOOLS),
            "remaining_steps": self.max_steps,
        }
        info = {"episode_id": episode.episode_id, "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
                "mode": self.mode}
        return obs, info

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if self._episode is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            obs = self._obs(None, EnvError("episode_finished", "episode already ended; call reset()"))
            return obs, 0.0, self._terminated, self._truncated, {}
        self._steps += 1
        if self._clock() - self._t0 > self.time_limit_s:
            return self._finish(None, truncated=True, reason="time_limit",
                                obs=self._obs(None, EnvError("time_limit", "episode time limit exceeded")))
        try:
            name, args = self._parse_action(action)
            if name == "submit":
                return self._finish(args["findings"], truncated=False, reason="submitted")
            obs = self._obs(getattr(self, f"_tool_{name}")(**args), None)
        except EnvError as exc:
            obs = self._obs(None, exc)
        if self._steps >= self.max_steps:
            return self._finish(None, truncated=True, reason="step_limit", obs=obs)
        return obs, 0.0, False, False, {}

    def time_left(self) -> float:
        """Evaluator-side: seconds left in the episode time budget (60 s from reset, policy time included)."""
        if self._episode is None or self._done:
            return 0.0
        return max(self.time_limit_s - (self._clock() - self._t0), 0.0)

    def abort(self, reason: str) -> tuple[dict, float, bool, bool, dict]:
        """Evaluator-side end of an episode whose policy hung, crashed or answered garbage.

        Same outcome as the time/step limit: truncated=true and empty findings. Never called by a policy.
        """
        if self._episode is None:
            raise RuntimeError("call reset() before abort()")
        if self._done:
            return self._obs(None, None), 0.0, self._terminated, self._truncated, {}
        code = "time_limit" if reason == "time_limit" else "policy_error"
        message = "episode time limit exceeded" if code == "time_limit" else "policy process failed"
        return self._finish(None, truncated=True, reason=reason, obs=self._obs(None, EnvError(code, message)))

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
        self._known_ids = episode.evidence_ids()
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

    def _obs(self, result: Any, error: EnvError | None) -> dict:
        return {"result": result, "error": None if error is None else {"code": error.code, "message": error.message},
                "remaining_steps": max(self.max_steps - self._steps, 0)}

    def _parse_action(self, action: Any) -> tuple[str, dict]:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str) or \
                set(action) - {"name", "args"}:
            raise EnvError("invalid_action", "action must be an object {name, args}")
        name = action["name"]
        if name not in TOOLS:
            raise EnvError("unknown_action", f"unknown action; available: {sorted(TOOLS)}")
        args = action.get("args", {})
        if not isinstance(args, dict):
            raise EnvError("invalid_arguments", "args must be an object")
        required, optional = _ALLOWED_ARGS[name]
        if not required <= set(args) or set(args) - required - optional:
            raise EnvError("invalid_arguments", f"{name} expects {sorted(required)} (+ optional {sorted(optional)})")
        return name, args

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
            prefix, _, number = cursor.partition(":") if isinstance(cursor, str) else ("", "", "")
            # ASCII digits only: str.isdigit() also accepts e.g. '²', which int() rejects (crash, found by fuzzing)
            if prefix != table or not (number.isascii() and number.isdigit()) or int(number) > len(rows):
                raise EnvError("invalid_cursor", "cursor is not valid for this table")
            offset = int(number)
        page = rows[offset:offset + limit]
        nxt = offset + len(page)
        return {"records": _copy(page), "next_cursor": f"{table}:{nxt}" if nxt < len(rows) else None}

    def _tool_get_record(self, table: Any, id: Any) -> dict:  # noqa: A002 - documented argument name
        index = self._by_id[self._table(table)]
        if not isinstance(id, str) or id not in index:
            raise EnvError("not_found", "no record with this id in this table")
        return _copy(index[id])

    def _tool_get_document(self, doc_id: Any) -> dict:
        if not isinstance(doc_id, str) or doc_id not in self._docs:
            raise EnvError("not_found", "no document with this doc_id")
        return _copy(self._docs[doc_id])

    def _tool_calculator(self, expression: Any) -> str:
        try:
            return evaluate(expression)
        except CalculatorError as exc:
            raise EnvError("calculator_error", str(exc)) from None

    def _finish(self, submission: Any, truncated: bool, reason: str,
                obs: dict | None = None) -> tuple[dict, float, bool, bool, dict]:
        inspection_steps = self._steps if truncated else self._steps - 1  # submit is not an inspection step
        report = grade(None if truncated else submission, self._episode.truth, inspection_steps, self._known_ids)
        self._done, self._terminated, self._truncated = True, not truncated, truncated
        self._report = {**report.to_dict(), "terminated": not truncated, "truncated": truncated,
                        "end_reason": reason if report.valid else "invalid_submission",
                        "elapsed_s": round(self._clock() - self._t0, 4), "mode": self.mode}
        if obs is None:
            error = None if report.valid else EnvError("invalid_submission", report.error or "invalid submission")
            obs = self._obs({"accepted": True} if report.valid else None, error)
        if self.mode == "train":
            info = {"grade": {k: self._report[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                                           "reward", "inspection_steps", "valid", "end_reason")}}
            return obs, report.reward, not truncated, truncated, info
        return obs, 0.0, not truncated, truncated, {}
