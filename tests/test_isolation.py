"""Policies run in a separate locked-down process (T2, jury block C20 "Граница доступа")."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from audit_env import runner
from audit_env.env import AuditEnv
from audit_env.generator import generate_episode
from audit_env.policies import POLICIES
from audit_env.policies.base import Policy
from audit_env.runner import RemotePolicy, policy_process_spec, run_episode

REPO = Path(__file__).resolve().parent.parent
KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps", "end_reason", "terminated",
        "truncated")


@pytest.fixture(scope="module")
def episode_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("jury") / "D16"
    generate_episode(9, "interacting").write(directory)
    return directory


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_isolated_run_equals_in_process_run(name, episode_dir):
    config = {"mode": "closed", "episode_dir": str(episode_dir)}
    a_traj, a = run_episode(name, 9, config, isolated=True)
    b_traj, b = run_episode(name, 9, config)
    assert [line["action"] for line in a_traj] == [line["action"] for line in b_traj]
    assert [line["observation"] for line in a_traj] == [line["observation"] for line in b_traj]
    assert {k: a[k] for k in KEYS} == {k: b[k] for k in KEYS}
    assert a["isolated"] and not b["isolated"]
    assert all(line["reward"] == 0.0 for line in a_traj)  # closed mode


def test_policy_process_gets_no_episode_path_seed_or_config(episode_dir, tmp_path):
    argv, env = policy_process_spec("rules_auditor", str(tmp_path))
    text = json.dumps([argv, env])
    for secret in (str(episode_dir), episode_dir.name, "truth", "provenance", "episodes", "interacting", "seed"):
        assert secret not in text, secret
    assert argv[1:3] == ["-I", "-B"]  # isolated mode: no PYTHON* variables, no user site, no cwd on sys.path
    assert set(env) <= {"PYTHONIOENCODING", "SYSTEMROOT"}


def test_policy_process_runs_in_an_empty_private_directory(episode_dir):
    remote = RemotePolicy("random")
    try:
        work = Path(remote.workdir)
        assert work.is_dir() and not any(work.iterdir())
        assert episode_dir.resolve() not in work.resolve().parents
    finally:
        remote.close()
    assert not work.exists()


def _worker_snippet(code: str) -> subprocess.CompletedProcess:
    bootstrap = (f"import sys; sys.path.insert(0, {str(REPO)!r}); "
                 "from audit_env.policy_worker import lock_down; import audit_env.policy_worker as w; " + code)
    return subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap], capture_output=True, text=True, timeout=60)


def test_worker_does_not_import_environment_generator_or_grader():
    proc = _worker_snippet("print(sorted(m for m in sys.modules if m.startswith('audit_env')))")
    modules = proc.stdout
    assert "audit_env.policies" in modules
    for forbidden in ("audit_env.grader", "audit_env.generator", "audit_env.env", "audit_env.validate"):
        assert f"'{forbidden}" not in modules, forbidden


def test_locked_down_worker_cannot_read_episode_files_or_use_network_or_processes(episode_dir):
    truth = str(episode_dir / "truth.json")
    code = f"""
lock_down()
results = {{}}
def attempt(name, fn):
    try:
        fn(); results[name] = 'allowed'
    except PermissionError:
        results[name] = 'refused'
attempt('read_truth', lambda: open({truth!r}).read())
attempt('read_repo_episode', lambda: open({str(REPO / 'README.md')!r}).read())
attempt('list_episode_dir', lambda: __import__('os').listdir({str(episode_dir)!r}))
attempt('write_file', lambda: open('x.txt', 'w'))
attempt('subprocess', lambda: __import__('subprocess').run(['echo', 'x']))
attempt('socket', lambda: __import__('socket').create_connection(('127.0.0.1', 9), timeout=1))
attempt('stdlib_lazy_import', lambda: __import__('fractions'))
# bypasses found in review: low-level write inside the code, Windows API, UDP/DNS, hard links
import os, socket
pkg = os.path.dirname(w.__file__)
attempt('os_open_write_in_code', lambda: os.close(os.open(os.path.join(pkg, '_probe.tmp'), os.O_WRONLY | os.O_CREAT)))
attempt('os_open_read_outside', lambda: os.close(os.open({truth!r}, os.O_RDONLY)))
attempt('udp_sendto', lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 9)))
attempt('dns', lambda: socket.gethostbyname('localhost'))
attempt('hard_link', lambda: os.link({truth!r}, os.path.join(pkg, '_probe_link.tmp')))
if os.name == 'nt':
    import _winapi
    attempt('winapi_createfile', lambda: _winapi.CreateFile({truth!r}, 0x80000000, 1, 0, 3, 0x80, 0))
    attempt('winapi_createprocess', lambda: _winapi.CreateProcess(None, 'cmd /c echo x', None, None, 0, 0, None, None, None))
print(__import__('json').dumps(results))
"""
    proc = _worker_snippet(code)
    for leftover in ("_probe.tmp", "_probe_link.tmp"):
        probe = REPO / "audit_env" / leftover
        assert not probe.exists(), f"{leftover} was created"
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = {"read_truth": "refused", "read_repo_episode": "refused", "list_episode_dir": "refused",
                "write_file": "refused", "subprocess": "refused", "socket": "refused",
                "stdlib_lazy_import": "allowed", "os_open_write_in_code": "refused",
                "os_open_read_outside": "refused", "udp_sendto": "refused", "dns": "refused", "hard_link": "refused"}
    if sys.platform == "win32":
        expected.update(winapi_createfile="refused", winapi_createprocess="refused")
    assert results == expected, results


def test_endless_reply_line_is_a_policy_failure(monkeypatch):
    monkeypatch.setattr(runner, "MAX_REPLY_CHARS", 1000)
    script = "import sys\nfor _ in sys.stdin:\n    sys.stdout.write('x' * 5000); sys.stdout.flush()"
    monkeypatch.setattr(runner, "policy_process_spec", lambda name, workdir: ([sys.executable, "-c", script], {}))
    traj, report = run_episode("random", 9, "interacting", "closed", isolated=True)
    assert report["truncated"] and report["end_reason"] == "policy_error"


def test_hung_policy_cannot_stretch_the_episode_beyond_its_budget(monkeypatch):
    monkeypatch.setattr(runner, "policy_process_spec",
                        lambda name, workdir: ([sys.executable, "-c", "import time; time.sleep(30)"], {}))
    env = AuditEnv(mode="closed", time_limit_s=1.0)
    _, report = run_episode("random", 9, "interacting", env=env, isolated=True)
    assert report["end_reason"] == "time_limit"
    assert report["wall_time_s"] < 1.0 + runner.ISOLATION_GRACE_S + 1.0  # budget + grace + process teardown


def test_policies_never_receive_the_generation_seed(monkeypatch):
    seen = []

    class Spy(Policy):
        name = "spy"

        def reset(self, obs, info, seed, truth=None):
            super().reset(obs, info, seed, truth)
            seen.append((seed, truth))

        def act(self, obs):
            return self.submit([])

    monkeypatch.setitem(POLICIES, "spy", Spy)
    run_episode("spy", 12345, "clean", "closed")
    run_episode("spy", 12345, "clean", "closed", policy_seed=7)
    assert seen == [(0, None), (7, None)]


@pytest.mark.parametrize("script,reason", [
    ("import sys; sys.exit(3)", "policy_error"),
    ("import sys; [print('not json', flush=True) for _ in sys.stdin]", "policy_error"),
    ("import sys, json; [print(json.dumps({'error': 'boom'}), flush=True) for _ in sys.stdin]", "policy_error"),
    ("import time; time.sleep(30)", "time_limit"),
])
def test_failing_policy_process_ends_episode_as_truncated_with_empty_answer(monkeypatch, script, reason):
    monkeypatch.setattr(runner, "policy_process_spec", lambda name, workdir: ([sys.executable, "-c", script], {}))
    monkeypatch.setattr(runner, "ISOLATION_GRACE_S", 0.2)
    env = AuditEnv(mode="closed", time_limit_s=1.0)
    traj, report = run_episode("random", 9, "interacting", env=env, isolated=True)
    assert report["truncated"] and not report["terminated"]
    assert report["end_reason"] == reason
    assert report["n_findings"] == 0 and report["tp"] == 0 and report["fn"] > 0  # empty findings graded
    assert traj[-1]["truncated"] and traj[-1]["action"] is None and traj[-1]["reward"] == 0.0
    assert traj[-1]["observation"]["error"]["code"] in ("policy_error", "time_limit")
    # the evaluator keeps working afterwards
    assert run_episode("oracle", 9, "interacting", "closed")[1]["reward"] == 1.0


def test_isolated_run_respects_train_mode_reward():
    traj, report = run_episode("oracle", 9, "interacting", "train", isolated=True)
    assert traj[-1]["reward"] == report["reward"] == 1.0
