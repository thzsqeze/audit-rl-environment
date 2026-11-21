import json
import subprocess
import sys


def test_stdio_server_roundtrip_keeps_grade_out_of_band(tmp_path):
    log = tmp_path / "grader.jsonl"
    reqs = [
        {"cmd": "step", "action": {"name": "calculator", "args": {"expression": "1"}}},
        {"cmd": "reset", "seed": 3, "config": "clean"},
        {"cmd": "reset", "seed": 3, "config": "../configs/clean.json"},
        {"cmd": "step", "action": {"name": "list_records", "args": {"table": "fx", "limit": 2}}},
        "garbage",
    ]
    lines = [json.dumps(r) if not isinstance(r, str) else r for r in reqs]
    proc = subprocess.Popen([sys.executable, "-m", "audit_env", "serve", "--mode", "closed", "--grader-log", str(log)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out = []
    for line in lines:
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        out.append(json.loads(proc.stdout.readline()))
    episode_id = out[1]["info"]["episode_id"]
    proc.stdin.write(json.dumps({"cmd": "step", "action": {"name": "submit", "args": {"findings": {
        "episode_id": episode_id, "findings": []}}}}) + "\n")
    proc.stdin.flush()
    final = json.loads(proc.stdout.readline())
    proc.stdin.close()
    proc.wait(timeout=10)
    assert "error" in out[0]
    assert out[1]["observation"]["remaining_steps"] == 100
    assert "error" in out[2]  # paths are not accepted as configs
    assert len(out[3]["observation"]["result"]["records"]) == 2
    assert "error" in out[4]
    assert final["terminated"] and final["reward"] == 0.0 and final["info"] == {}
    report = json.loads(log.read_text().splitlines()[-1])
    assert report["episode_id"] == episode_id and report["reward"] == 0.998  # 1 inspection step
