from __future__ import annotations

import importlib.util
import io
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HOOK = Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "pre_commit_evals.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pre_commit_evals", HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _load()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git commit -m x", True),
        ("git add -A && git commit -m x", True),
        ("git -C repo commit", True),
        ("git status", False),
        ("git commit-tree abc", False),
        ("git log --grep commit", False),
    ],
)
def test_detects_git_commit(command: str, expected: bool) -> None:
    assert hook.is_git_commit(command) is expected


def _run(monkeypatch: pytest.MonkeyPatch, command: str, returncode: int) -> tuple[int, list[Any]]:
    calls: list[Any] = []

    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, returncode, "table", "EVAL GATE FAILED")

    payload = {"tool_name": "Bash", "tool_input": {"command": command}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook.subprocess, "run", fake_run)
    return hook.main(), calls


def test_non_commit_skips_evals(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, calls = _run(monkeypatch, "git status", 1)
    assert rc == 0 and calls == []


def test_commit_passes_when_gate_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, calls = _run(monkeypatch, "git commit -m x", 0)
    assert rc == 0
    assert calls == [["uv", "run", "--quiet", "kb-evals", "--gate"]]


def test_commit_blocked_when_gate_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, _ = _run(monkeypatch, "git commit -m x", 1)
    assert rc == 2
    assert "Commit blocked" in capsys.readouterr().err
