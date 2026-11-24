"""Claude Code PreToolUse hook: run the retrieval evals before any `git commit`.

Reads the hook payload from stdin. If the Bash command is a git commit, runs
`kb-evals --gate`; when a metric falls below evals/thresholds.yaml it exits 2, which blocks the
tool call and shows stderr to Claude so it fixes the regression instead of committing.
Any other command passes through untouched (exit 0).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

GIT_COMMIT = re.compile(r"(^|[;&|(]\s*|\s)git(\s+-C\s+\S+)?(\s+-c\s+\S+)*\s+commit(?=$|[\s;&|)])")


def is_git_commit(command: str) -> bool:
    return bool(GIT_COMMIT.search(command))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = str(payload.get("tool_input", {}).get("command", ""))
    if payload.get("tool_name") != "Bash" or not is_git_commit(command):
        return 0

    proc = subprocess.run(
        ["uv", "run", "--quiet", "kb-evals", "--gate"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode == 0:
        return 0
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
    print(
        "Commit blocked: retrieval evals are below the floors in evals/thresholds.yaml.\n"
        "Fix the regression (see evals/reports/latest.md). Do not lower thresholds or edit "
        f"the golden set to pass.\n\n{tail}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
