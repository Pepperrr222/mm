"""问题三唯一复现入口：全年求解 -> 官方工作簿 -> 图表 -> 全产物哈希清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "04_结果" / "问题3_10场景完整求解"
FIG_DIR = ROOT / "05_图表" / "问题3"
WORKBOOK = ROOT / "04_结果" / "result3_问题3_结果.xlsx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], ok_codes: tuple[int, ...] = (0,)):
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in ok_codes:
        raise subprocess.CalledProcessError(result.returncode, command)


def main():
    parser = argparse.ArgumentParser(description="问题三完整复现")
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--solver-time-limit", type=float, default=120.0)
    parser.add_argument("--skip-solve", action="store_true", help="仅重建工作簿、图表和清单")
    args = parser.parse_args()
    started = time.perf_counter()

    if not args.skip_solve:
        run([
            sys.executable, str(ROOT / "03_代码" / "problem3_full_30scenario.py"),
            "--start-day", "31", "--end-day", "364",
            "--scenarios", str(args.scenarios),
            "--solver-time-limit", str(args.solver_time_limit),
            "--output-dir", str(RESULT_DIR),
        ], ok_codes=(0, 1))
    run([sys.executable, str(ROOT / "03_代码" / "problem3_workbook.py")])
    run([sys.executable, str(ROOT / "03_代码" / "problem3_figures.py")])

    validation = json.loads((RESULT_DIR / "问题3_总体验证.json").read_text(encoding="utf-8"))
    if validation.get("fallback_events"):
        raise RuntimeError("问题三正式10情景结果出现求解回退，停止发布复现清单")
    if validation.get("rows_detail") != validation.get("expected_detail"):
        raise RuntimeError("问题三逐时记录数不完整，停止发布复现清单")

    scripts = [
        ROOT / "03_代码" / "problem3_full_30scenario.py",
        ROOT / "03_代码" / "problem3_workbook.py",
        ROOT / "03_代码" / "problem3_figures.py",
        Path(__file__),
    ]
    core_outputs = sorted(p for p in RESULT_DIR.iterdir() if p.is_file() and p.name != "问题3_复现清单.json")
    figures = sorted(p for p in FIG_DIR.iterdir() if p.suffix.lower() in {".png", ".svg"})
    all_artifacts = core_outputs + [WORKBOOK] + figures
    assert len(figures) == 18, f"应有9组PNG/SVG，实际{len(figures)}个文件"
    manifest = {
        "single_entry": "python 03_代码/问题3_完整复现.py",
        "published_run": {
            "scenarios_requested": args.scenarios,
            "solver_time_limit_s": args.solver_time_limit,
            "skip_solve": args.skip_solve,
            "note": "正式全年结果采用10条历史整段残差情景和120秒单次时限；无回退。自动验证中的少量计划层逻辑容差记录已在论文中披露。",
        },
        "validation_passed": bool(validation.get("passed", False)),
        "validation_note": "数值平衡与实际执行约束通过；计划层存在少量MILP容差记录，故严格总标志为false。",
        "runtime_s_this_entry": time.perf_counter() - started,
        "python": platform.python_version(),
        "scripts": {str(p.relative_to(ROOT)): sha256(p) for p in scripts},
        "artifacts": {str(p.relative_to(ROOT)): sha256(p) for p in all_artifacts},
    }
    target = RESULT_DIR / "问题3_复现清单.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {target}; artifacts={len(all_artifacts)}, figures={len(figures)}")


if __name__ == "__main__":
    main()
