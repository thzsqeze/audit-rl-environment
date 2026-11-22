"""Policy process for isolated runs (`audit-env run --isolated`, `audit-env check`).

The evaluator starts `python -I -m audit_env.policy_worker <policy>` with an empty working directory and a
minimal environment: no episode path, seed, config or truth is passed in argv, environment variables or
files. The worker only imports the policy package (never audit_env.env / generator / grader), then locks
itself down with an audit hook (PEP 578) before it reads the first message:

  * `open` is allowed only for reading files of the Python installation and of the audit_env package
    (lazy stdlib imports); any other path - an episode directory, truth.json, the repository - is refused;
  * starting processes, sockets and loading native libraries are refused.

Protocol (JSON lines; the worker answers every message with exactly one line):
  evaluator -> {"type": "reset", "observation": obs, "info": public_info, "policy_seed": int,
                "oracle_findings": [...] | absent}       worker -> {"ok": true}
  evaluator -> {"type": "act", "observation": obs}       worker -> {"action": <JSON action>}
  any failure                                            worker -> {"error": "<short text>"} and exits 1

`oracle_findings` is sent only to the evaluator-side oracle (as in the jury protocol, the evaluator builds the
correct findings from the closed truth and the oracle submits them through the ordinary submit action).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from audit_env.policies import POLICIES  # imports policies only: no environment, generator or grader

# Audit-event prefixes refused after lock-down: processes, sockets/DNS, FFI, Windows API wrappers, links,
# file-system changes. Found by review: _winapi.CreateFile/CreateProcess, socket.sendto/gethostbyname and
# os.link bypassed the first version.
_BLOCKED_EVENTS = ("subprocess.", "_posixsubprocess", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
                   "os.fork", "os.forkpty", "os.startfile", "os.kill", "os.putenv", "os.unsetenv", "socket.",
                   "ctypes.", "mmap.", "_winapi.", "winreg.", "msvcrt.", "urllib.", "http.", "ftplib.", "smtplib.",
                   "webbrowser.", "shutil.", "os.remove", "os.rename", "os.replace", "os.rmdir", "os.mkdir",
                   "os.chdir", "os.chmod", "os.chown", "os.chflags", "os.utime", "os.truncate", "os.link",
                   "os.symlink", "os.add_dll_directory", "sqlite3.", "fcntl.", "resource.", "pty.")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_PATH_EVENTS = ("os.listdir", "os.scandir", "glob.glob")  # allowed only inside the code roots (imports list dirs)


class Refused(PermissionError):
    pass


def _allowed_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             str(Path(__file__).resolve().parent)}
    return [os.path.normcase(os.path.realpath(r)) + os.sep for r in roots if r]


def lock_down() -> None:
    """Install an audit hook that refuses file access outside the code, processes, sockets and FFI."""
    roots = _allowed_roots()

    def inside_code(path: Any) -> bool:
        if path is None or path == "":
            path = "."  # listdir() without an argument lists the (empty) working directory
        real = os.path.normcase(os.path.realpath(os.fsdecode(path)))
        return any((real + os.sep).startswith(r) for r in roots)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if isinstance(path, int):  # already-open descriptors (stdin/stdout)
                return
            # builtin open() reports a mode string; os.open() reports mode=None and integer flags
            writes = (
                (isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
                or (mode is not None and not isinstance(mode, str))
            )
            if writes:
                raise Refused("policy process may not write files")
            if not inside_code(path):
                raise Refused("policy process may not read files outside the code")
        elif event in _PATH_EVENTS:
            if isinstance(args[0], int) or not inside_code(args[0]):
                raise Refused("policy process may not list directories outside the code")
        elif event.startswith(_BLOCKED_EVENTS):
            raise Refused(f"policy process may not use {event}")

    sys.addaudithook(hook)


def _send(out: Any, payload: dict) -> None:
    out.write(json.dumps(payload, sort_keys=True) + "\n")
    out.flush()


def serve(policy_name: str, stdin: Any = None, stdout: Any = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    policy = POLICIES[policy_name]()
    for line in stdin:
        try:
            msg = json.loads(line)
            if msg["type"] == "reset":
                policy.reset(msg["observation"], msg["info"], int(msg["policy_seed"]), None)
                if policy.needs_truth:
                    policy.load_findings(msg["oracle_findings"])
                _send(stdout, {"ok": True})
            elif msg["type"] == "act":
                _send(stdout, {"action": policy.act(msg["observation"])})
            else:
                raise ValueError("unknown message type")
        except Exception as exc:  # report and stop: the evaluator ends the episode as truncated
            _send(stdout, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in POLICIES:
        print(json.dumps({"error": f"usage: policy_worker <{'|'.join(sorted(POLICIES))}>"}), flush=True)
        return 2
    # text streams with a fixed encoding and newline handling on every platform
    stdin = open(sys.stdin.fileno(), encoding="utf-8", newline="\n", closefd=False)  # noqa: SIM115
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)  # noqa: SIM115
    lock_down()
    return serve(args[0], stdin, stdout)


if __name__ == "__main__":
    sys.exit(main())
