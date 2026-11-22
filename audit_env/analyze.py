"""Measured failure analysis (`audit-env analyze`, also a step of `audit-env check`).

Evaluator-side: reads a run directory (grades.jsonl + trajectories.jsonl, as written by `run` and `check`)
and the closed files of the episodes (truth.json, provenance.json) and writes a Markdown report:

* per policy: episodes, P/R/F1/return means, steps, wall time, truncations, invalid submissions;
* the same split by episode kind (clean / single error / interacting errors);
* false positives attributed to the correct-but-unusual case they hit (provenance.distractors, e.g.
  `recurring_invoice`) or, if the finding names no distractor, to the closest matcher failure
  (`near_miss:amount`, `near_miss:evidence`, ... from GradeReport.finding_status);
* false negatives by (type, variant, scenario) of the missed truth issue;
* actions per policy and the share of steps that returned an error.

The report body is a pure function of its inputs (no clock, no paths), so the same run gives the same file.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

KINDS = ("clean", "single", "interacting")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def episode_index(roots: list[str | Path]) -> dict[str, Path]:
    """episode_id -> directory, for every episode directory (manifest.json) below the given roots."""
    index: dict[str, Path] = {}
    for root in roots:
        for manifest in sorted(Path(root).rglob("manifest.json")):
            with open(manifest, encoding="utf-8") as fh:
                index[json.load(fh)["episode_id"]] = manifest.parent
    return index


def _closed(directory: Path) -> tuple[dict, dict]:
    with open(directory / "truth.json", encoding="utf-8") as fh:
        truth = json.load(fh)
    provenance_file = directory / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8")) if provenance_file.exists() else {}
    return truth, provenance


def episode_kind(truth: dict) -> str:
    if not truth["issues"]:
        return "clean"
    return "interacting" if any(t.get("scenario") for t in truth["issues"]) else "single"


def _findings_by_episode(trajectories: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """(policy, episode_id) -> submitted findings (structurally as sent; [] if no valid list was sent)."""
    out: dict[tuple[str, str], list[dict]] = {}
    for line in trajectories:
        action = line.get("action")
        if isinstance(action, dict) and action.get("name") == "submit":
            payload = (action.get("args") or {}).get("findings")
            findings = payload.get("findings") if isinstance(payload, dict) else None
            out[(line.get("policy", ""), line["episode_id"])] = findings if isinstance(findings, list) else []
    return out


def analyze(run_dir: str | Path, episodes: dict[str, Path]) -> dict[str, Any]:
    run = Path(run_dir)
    grades = _read_jsonl(run / "grades.jsonl")
    trajectories = _read_jsonl(run / "trajectories.jsonl")
    submitted = _findings_by_episode(trajectories)
    closed = {eid: _closed(d) for eid, d in episodes.items()}

    per_policy: dict[str, list[dict]] = defaultdict(list)
    fp_counts: Counter = Counter()  # (policy, attributed_to, finding_type)
    fn_counts: Counter = Counter()  # (policy, type, variant, scenario)
    unknown_episodes = sorted({g["episode_id"] for g in grades} - set(closed))
    for g in grades:
        if g["episode_id"] not in closed:
            continue
        truth, provenance = closed[g["episode_id"]]
        g = {**g, "kind": g.get("kind") or episode_kind(truth)}
        per_policy[g["policy"]].append(g)
        distractor_of = {}
        for entry in provenance.get("distractors", []):
            kind, _, ref = entry.partition(":")
            distractor_of.setdefault(ref, kind)
        findings = submitted.get((g["policy"], g["episode_id"]), [])
        for finding, status in zip(findings, g.get("finding_status") or [], strict=False):
            if status == "tp":
                continue
            ids = finding.get("affected_ids") if isinstance(finding.get("affected_ids"), list) else []
            hits = sorted({distractor_of[i] for i in ids if isinstance(i, str) and i in distractor_of})
            if hits:
                cause = "distractor:" + "+".join(hits)
            elif status == "no_issue_in_books":
                cause = "clean_books:no_distractor_named"
            else:
                cause = f"near_miss:{status}"
            fp_counts[(g["policy"], cause, str(finding.get("type")))] += 1
        issues = {t["issue_id"]: t for t in truth["issues"]}
        for issue_id in g.get("missed") or []:
            t = issues.get(issue_id)
            if t:
                fn_counts[(g["policy"], t["type"], t.get("variant") or "-", t.get("scenario") or "-")] += 1

    actions: dict[str, Counter] = defaultdict(Counter)
    errors: Counter = Counter()
    for line in trajectories:
        action = line.get("action")
        if line.get("step", 0) == 0 or action is None:
            continue
        name = action.get("name") if isinstance(action, dict) else None
        actions[line.get("policy", "")][name if isinstance(name, str) else "<invalid>"] += 1
        if (line.get("observation") or {}).get("error") is not None:
            errors[line.get("policy", "")] += 1
    return {"per_policy": per_policy, "fp": fp_counts, "fn": fn_counts, "actions": actions, "errors": errors,
            "unknown_episodes": unknown_episodes}


def _summary_row(name: str, reports: list[dict]) -> str:
    def m(key: str) -> float:
        return mean((r.get(key) or 0) for r in reports)

    return (f"| {name} | {len(reports)} | {m('precision'):.3f} | {m('recall'):.3f} | {m('f1'):.3f} | "
            f"{m('reward'):.3f} | {sum(r.get('fp') or 0 for r in reports)} | {sum(r.get('fn') or 0 for r in reports)} | "
            f"{m('steps'):.1f} | {m('wall_time_s'):.3f} | {max(r.get('wall_time_s') or 0 for r in reports):.3f} | "
            f"{sum(bool(r.get('truncated')) for r in reports)} | {sum(not r.get('valid', True) for r in reports)} |")


def render(result: dict[str, Any], title: str) -> str:
    out = [f"# {title}", "",
           "Generated by `audit-env analyze` from grades.jsonl, trajectories.jsonl and the closed episode files.",
           "Means are per episode; on clean episodes P = R = 0 by the brief's definition, F1 = 1 for an empty answer.",
           "", "## Policies", "",
           "| policy | episodes | P | R | F1 | return | FP | FN | steps | time, s | max time, s | truncated | invalid |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_policy = result["per_policy"]
    out += [_summary_row(p, rs) for p, rs in per_policy.items()]
    out += ["", "## By episode kind", "", "| policy | kind | episodes | F1 | return | FP | FN |", "|---|---|---|---|---|---|---|"]
    for p, rs in per_policy.items():
        for kind in KINDS:
            sub = [r for r in rs if r["kind"] == kind]
            if sub:
                out.append(f"| {p} | {kind} | {len(sub)} | {mean(r.get('f1') or 0 for r in sub):.3f} | "
                           f"{mean(r['reward'] for r in sub):.3f} | {sum(r.get('fp') or 0 for r in sub)} | "
                           f"{sum(r.get('fn') or 0 for r in sub)} |")
    out += ["", "## False positives by cause", "",
            "`distractor:<kind>` - the finding names a correct-but-unusual case of that kind (provenance.distractors); "
            "`near_miss:<reason>` - closest truth issue failed on that matcher criterion; "
            "`clean_books:no_distractor_named` - clean books, no distractor id in the finding.", ""]
    if result["fp"]:
        out += ["| policy | cause | finding type | count |", "|---|---|---|---|"]
        out += [f"| {p} | {c} | {t} | {n} |" for (p, c, t), n in sorted(result["fp"].items())]
    else:
        out.append("No false positives.")
    out += ["", "## Missed issues (false negatives)", ""]
    if result["fn"]:
        out += ["| policy | type | variant | scenario | count |", "|---|---|---|---|---|"]
        out += [f"| {p} | {t} | {v} | {s} | {n} |" for (p, t, v, s), n in sorted(result["fn"].items())]
    else:
        out.append("No missed issues.")
    out += ["", "## Actions", "", "| policy | actions | error steps | per action |", "|---|---|---|---|"]
    for p in sorted(result["actions"]):
        counts = result["actions"][p]
        total = sum(counts.values())
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        out.append(f"| {p} | {total} | {result['errors'][p]} ({result['errors'][p] / total:.0%}) | {detail} |")
    if result["unknown_episodes"]:
        out += ["", f"Episodes without closed files (skipped): {len(result['unknown_episodes'])}"]
    return "\n".join(out) + "\n"


def write_report(run_dir: str | Path, episode_roots: list[str | Path], out: str | Path, title: str) -> dict:
    result = analyze(run_dir, episode_index(episode_roots))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render(result, title))
    return result
