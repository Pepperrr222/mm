from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "04_结果" / "问题2_终端价值实验"
SALVAGE_REFERENCE = 0.45
E_MIN, E_MAX = 1200.0, 10800.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary_paths = sorted(EXPERIMENT_ROOT.glob("*_??月/问题2_终端价值实验_汇总.csv"))
    detail_paths = sorted(EXPERIMENT_ROOT.glob("*_??月/问题2_终端价值实验_逐日.csv"))
    if len(summary_paths) != 4 or len(detail_paths) != 4:
        raise RuntimeError(f"预期4个季节窗口，实际汇总{len(summary_paths)}个、逐日{len(detail_paths)}个")

    window_frames = []
    detail_frames = []
    for path in summary_paths:
        frame = pd.read_csv(path)
        frame.insert(0, "window", path.parent.name)
        window_frames.append(frame)
    for path in detail_paths:
        frame = pd.read_csv(path, parse_dates=["date", "training_cutoff"])
        frame.insert(0, "window", path.parent.name)
        detail_frames.append(frame)
    windows = pd.concat(window_frames, ignore_index=True)
    details = pd.concat(detail_frames, ignore_index=True)

    expected_modes = {"fixed_v_0", "fixed_v_0.3", "fixed_v_0.5", "fixed_v_0.8", "fixed_v_1", "dynamic_v"}
    if set(windows["mode"]) != expected_modes:
        raise RuntimeError(f"模式集合错误：{set(windows['mode'])}")
    if not (windows.groupby("mode")["days"].sum() == 56).all():
        raise RuntimeError("每个模式必须覆盖四个14天窗口")

    for (window, mode), group in details.sort_values("date").groupby(["window", "mode"]):
        starts = group["E_start_kwh"].to_numpy()[1:]
        previous_ends = group["E_real_end_kwh"].to_numpy()[:-1]
        if starts.size and np.max(np.abs(starts - previous_ends)) > 1e-6:
            raise RuntimeError(f"跨日状态不连续：{window}/{mode}")
        if not (group["training_cutoff"] < group["date"]).all():
            raise RuntimeError(f"训练截止日泄漏：{window}/{mode}")
        for row in group.itertuples():
            residual_dates = [pd.Timestamp(value) for value in json.loads(row.sampled_residual_dates)]
            if any(value >= row.date for value in residual_dates):
                raise RuntimeError(f"残差日期泄漏：{window}/{mode}/{row.date.date()}")

    scenario_sets = details.groupby(["window", "date"])["sampled_residual_dates"].nunique()
    if int(scenario_sets.max()) != 1:
        raise RuntimeError("同日各模式未共享完全相同的残差情景")
    if float(details["max_residual_kwh"].max()) > 1e-5:
        raise RuntimeError("数值残差超过门限")
    if int(details[[
        "emergency_charge_exec_periods",
        "emergency_charge_scenario_periods",
        "emergency_surplus_scenario_periods",
    ]].to_numpy().sum()) != 0:
        raise RuntimeError("互斥逻辑存在违规计数")

    value_columns = [
        "total_cost_yuan", "plan_cost_yuan", "emergency_cost_yuan", "emergency_kwh",
        "surplus_kwh", "charge_exec_kwh", "discharge_exec_kwh", "solver_time_s",
    ]
    combined = windows.groupby("mode", as_index=False)[value_columns].sum()
    combined["days"] = windows.groupby("mode")["days"].sum().reindex(combined["mode"]).to_numpy()
    combined["mean_terminal_value"] = (
        details.groupby("mode")["terminal_value_used_yuan_per_kwh"].mean()
        .reindex(combined["mode"]).to_numpy()
    )
    combined["E_end_minus_v0_kwh"] = 0.0
    for window, frame in windows.groupby("window"):
        base_end = float(frame.loc[frame["mode"] == "fixed_v_0", "E_end_kwh"].iloc[0])
        mapping = frame.set_index("mode")["E_end_kwh"] - base_end
        combined["E_end_minus_v0_kwh"] += combined["mode"].map(mapping).to_numpy()
    base_cost = float(combined.loc[combined["mode"] == "fixed_v_0", "total_cost_yuan"].iloc[0])
    base_emergency = float(combined.loc[combined["mode"] == "fixed_v_0", "emergency_cost_yuan"].iloc[0])
    combined["cost_delta_vs_fixed_v0_yuan"] = combined["total_cost_yuan"] - base_cost
    combined["emergency_cost_delta_vs_fixed_v0_yuan"] = combined["emergency_cost_yuan"] - base_emergency
    combined[f"inventory_adjusted_delta_at_{SALVAGE_REFERENCE:g}_yuan"] = (
        combined["cost_delta_vs_fixed_v0_yuan"]
        - SALVAGE_REFERENCE * combined["E_end_minus_v0_kwh"]
    )
    combined["break_even_salvage_yuan_per_kwh"] = np.where(
        combined["E_end_minus_v0_kwh"] > 1e-9,
        combined["cost_delta_vs_fixed_v0_yuan"] / combined["E_end_minus_v0_kwh"],
        np.nan,
    )
    end_states = details.groupby("mode")["E_real_end_kwh"]
    combined["days_end_at_lower_bound"] = combined["mode"].map(
        end_states.apply(lambda values: int(np.sum(values <= E_MIN + 1e-5)))
    )
    combined["days_end_at_upper_bound"] = combined["mode"].map(
        end_states.apply(lambda values: int(np.sum(values >= E_MAX - 1e-5)))
    )
    combined["max_residual_kwh"] = combined["mode"].map(
        details.groupby("mode")["max_residual_kwh"].max()
    )

    windows_path = EXPERIMENT_ROOT / "问题2_终端价值实验_窗口汇总.csv"
    combined_path = EXPERIMENT_ROOT / "问题2_终端价值实验_四季汇总.csv"
    audit_path = EXPERIMENT_ROOT / "问题2_终端价值实验_总体验证.json"
    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    audit = {
        "passed": True,
        "windows": [path.parent.name for path in summary_paths],
        "days_per_mode": 56,
        "controlled_baseline": "fixed_v_0",
        "salvage_reference_yuan_per_kwh": SALVAGE_REFERENCE,
        "max_residual_kwh": float(details["max_residual_kwh"].max()),
        "information_leakage_count": 0,
        "cross_day_state_gap_max_kwh": 0.0,
        "shared_scenario_violations": 0,
        "mutual_exclusion_violations": 0,
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in summary_paths + detail_paths
        },
        "outputs": {
            str(windows_path.relative_to(ROOT)): sha256(windows_path),
            str(combined_path.relative_to(ROOT)): sha256(combined_path),
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(combined.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
