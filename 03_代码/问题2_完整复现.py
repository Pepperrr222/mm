#!/usr/bin/env python3
"""问题2唯一复现入口：续跑/校验全年求解，再独立复算和绘图。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "03_代码" / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run("问题2_求解.py", "--days", "334", "--scenarios", "30", "--solver-time-limit", "30")
    run("问题2_验证与绘图.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
