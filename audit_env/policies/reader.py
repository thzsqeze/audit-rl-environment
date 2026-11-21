"""Shared helper: page through tables via list_records."""

from __future__ import annotations

from typing import Any

from audit_env.policies.base import Policy


class TableReader(Policy):
    tables: tuple[str, ...] = ("ledger", "invoices", "payments", "fx")
    reserve_steps = 1  # keep one step for submit

    def reset(self, obs, info, seed, truth=None):
        super().reset(obs, info, seed, truth)
        self.data: dict[str, list[dict]] = {t: [] for t in self.tables}
        self.queue = list(self.tables)
        self.cursor: str | None = None
        self.pending: str | None = None

    def read_step(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        """Return the next list_records action, or None when all tables are read / budget is low."""
        if self.pending and obs.get("ok") and obs.get("result", {}).get("table") == self.pending:
            res = obs["result"]
            self.data[self.pending].extend(res["rows"])
            self.cursor = res["next_cursor"]
            if self.cursor is None:
                self.queue.pop(0)
        self.pending = None
        if not self.queue or obs.get("steps_remaining", 0) <= self.reserve_steps:
            return None
        table = self.queue[0]
        self.pending = table
        args: dict[str, Any] = {"table": table, "limit": 50}
        if self.cursor:
            args["cursor"] = self.cursor
        return {"tool": "list_records", "args": args}
