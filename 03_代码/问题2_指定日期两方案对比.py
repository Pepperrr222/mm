from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "03_代码" / "问题2_终端价值实验.py"
spec = importlib.util.spec_from_file_location("terminal_value_experiment", EXPERIMENT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载实验模块：{EXPERIMENT_PATH}")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
base = experiment.base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_real(real: pd.DataFrame, e_end: float) -> dict:
    return {
        "plan_cost_yuan": float(real.plan_cost_yuan.sum()),
        "emergency_cost_yuan": float(real.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(real.plan_cost_yuan.sum() + real.emergency_cost_yuan.sum()),
        "emergency_kwh": float(real.emergency_kwh.sum()),
        "surplus_kwh": float(real.surplus_kwh.sum()),
        "charge_exec_kwh": float(real.charge_exec_kwh.sum()),
        "discharge_exec_kwh": float(real.discharge_exec_kwh.sum()),
        "E_real_end_kwh": float(e_end),
        "E_real_min_kwh": float(min(real.E_start_kwh.min(), real.E_end_kwh.min())),
        "E_real_max_kwh": float(max(real.E_start_kwh.max(), real.E_end_kwh.max())),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    target_dates = [pd.Timestamp(value.strip()) for value in args.dates.split(",") if value.strip()]
    if not target_dates:
        raise ValueError("至少指定一个日期")
    data = base.read_inputs()
    date_to_day = {date: index for index, date in enumerate(data.dates)}
    missing = [str(date.date()) for date in target_dates if date not in date_to_day]
    if missing:
        raise ValueError(f"附件中不存在日期：{missing}")
    target_days = {date_to_day[date] for date in target_dates}
    baseline_path = ROOT / "04_结果" / "问题2_每日指标.csv"
    baseline_daily = pd.read_csv(baseline_path, parse_dates=["date"])
    forecaster = base.CausalForecaster(data, use_gbdt=not args.no_gbdt)
    records: list[dict] = []

    for day in range(1, max(target_days) + 1):
        forecast = forecaster.forecast(day)
        if day in target_days:
            date = data.dates[day]
            baseline_row = baseline_daily.loc[baseline_daily.date == date]
            if baseline_row.empty:
                raise RuntimeError(f"正式结果缺少日期：{date.date()}")
            e_initial = float(baseline_row.E_start_kwh.iloc[0])
            load_s, pv_s, sampled = base.generate_scenarios(
                forecast, forecaster, day, args.scenarios
            )

            plan_48 = base.solve_stochastic_milp(
                data.price, load_s, pv_s, e_initial, args.solver_time_limit, args.mip_gap
            )
            real_48, end_48 = base.replay_day(
                date, data.price, data.load_kw[day], data.pv_kw[day], plan_48, e_initial
            )
            check_48 = base.validate_day(real_48, plan_48, e_initial, load_s, pv_s)

            value = experiment.estimate_marginal_value(
                data.price,
                forecast["load"][base.N:],
                forecast["pv"][base.N:],
                e_initial,
                args.delta_energy,
                args.solver_time_limit,
                args.mip_gap,
            )
            if value["low_mip_gap"] > args.mip_gap + 1e-9 or value["high_mip_gap"] > args.mip_gap + 1e-9:
                raise RuntimeError(f"动态价值有限差分求解精度不足：{value}")
            plan_24 = experiment.solve_value_milp(
                data.price,
                load_s[:, :base.N],
                pv_s[:, :base.N],
                e_initial,
                value["used_value"],
                args.solver_time_limit,
                args.mip_gap,
            )
            real_24, end_24 = base.replay_day(
                date, data.price, data.load_kw[day], data.pv_kw[day], plan_24, e_initial
            )
            check_24 = base.validate_day(
                real_24, plan_24, e_initial, load_s[:, :base.N], pv_s[:, :base.N]
            )

            common = {
                "date": date,
                "E_initial_kwh": e_initial,
                "training_cutoff": data.dates[forecaster.training_cutoffs[day]],
                "scenario_count": args.scenarios,
                "sampled_residual_dates": json.dumps(
                    [str(data.dates[index].date()) for index in sampled], ensure_ascii=False
                ),
                "dynamic_v_yuan_per_kwh": value["used_value"],
                "finite_diff_E_low_kwh": value["E_low_kwh"],
                "finite_diff_E_high_kwh": value["E_high_kwh"],
                "finite_diff_J_low_yuan": value["J_low_yuan"],
                "finite_diff_J_high_yuan": value["J_high_yuan"],
                "finite_diff_low_mip_gap": value["low_mip_gap"],
                "finite_diff_high_mip_gap": value["high_mip_gap"],
                "finite_diff_low_solver_time_s": value["low_solver_time_s"],
                "finite_diff_high_solver_time_s": value["high_solver_time_s"],
            }
            for mode, plan, real, e_end, checks, horizon in (
                ("48h_cyclic", plan_48, real_48, end_48, check_48, 48),
                ("24h_dynamic_v", plan_24, real_24, end_24, check_24, 24),
            ):
                records.append({
                    **common,
                    "mode": mode,
                    "horizon_hours": horizon,
                    "plan_terminal_E_kwh": float(plan["E"][-1]),
                    **summarize_real(real, e_end),
                    "solver_elapsed_s": float(plan["elapsed_s"]),
                    "solver_mip_gap": float(plan["mip_gap"]),
                    "max_real_balance_residual_kwh": checks["max_real_balance_residual_kwh"],
                    "max_real_state_residual_kwh": checks["max_real_state_residual_kwh"],
                    "max_plan_state_residual_kwh": checks["max_plan_state_residual_kwh"],
                    "max_scenario_balance_residual_kwh": checks["max_scenario_balance_residual_kwh"],
                    "simultaneous_plan_periods": checks["simultaneous_plan_periods"],
                    "emergency_charge_exec_periods": checks["emergency_charge_exec_periods"],
                    "emergency_charge_scenario_periods": checks["emergency_charge_scenario_periods"],
                    "emergency_surplus_scenario_periods": checks["emergency_surplus_scenario_periods"],
                    "max_charge_exec_kwh": checks["max_charge_exec_kwh"],
                    "max_discharge_exec_kwh": checks["max_discharge_exec_kwh"],
                })
        forecaster.observe(day, forecast)

    detail = pd.DataFrame(records).sort_values(["date", "mode"]).reset_index(drop=True)
    comparison_rows = []
    for date, frame in detail.groupby("date"):
        left = frame.set_index("mode")
        dynamic = left.loc["24h_dynamic_v"]
        cyclic = left.loc["48h_cyclic"]
        cost_delta = float(dynamic.total_cost_yuan - cyclic.total_cost_yuan)
        energy_delta = float(dynamic.E_real_end_kwh - cyclic.E_real_end_kwh)
        v = float(dynamic.dynamic_v_yuan_per_kwh)
        comparison_rows.append({
            "date": date,
            "E_initial_kwh": dynamic.E_initial_kwh,
            "dynamic_v_yuan_per_kwh": v,
            "cost_48h_yuan": cyclic.total_cost_yuan,
            "cost_24h_dynamic_yuan": dynamic.total_cost_yuan,
            "cost_delta_24_minus_48_yuan": cost_delta,
            "emergency_cost_48h_yuan": cyclic.emergency_cost_yuan,
            "emergency_cost_24h_dynamic_yuan": dynamic.emergency_cost_yuan,
            "emergency_cost_delta_yuan": dynamic.emergency_cost_yuan - cyclic.emergency_cost_yuan,
            "E_end_48h_kwh": cyclic.E_real_end_kwh,
            "E_end_24h_dynamic_kwh": dynamic.E_real_end_kwh,
            "E_end_delta_kwh": energy_delta,
            "inventory_approx_adjusted_delta_at_dynamic_v_yuan": cost_delta - v * energy_delta,
            "shared_scenarios": bool(
                dynamic.sampled_residual_dates == cyclic.sampled_residual_dates
            ),
        })
    comparison = pd.DataFrame(comparison_rows)
    if not comparison["shared_scenarios"].all():
        raise RuntimeError("两方案未共享相同残差情景")
    if not (detail["training_cutoff"] < detail["date"]).all():
        raise RuntimeError("发现训练截止日不早于执行日")
    cyclic = detail[detail["mode"] == "48h_cyclic"]
    if np.max(np.abs(cyclic.plan_terminal_E_kwh - cyclic.E_initial_kwh)) > 1e-5:
        raise RuntimeError("48小时循环终端状态不等于初始状态")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "指定日期两方案逐项结果.csv"
    comparison_path = output_dir / "指定日期两方案对比.csv"
    manifest_path = output_dir / "指定日期两方案复现清单.json"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "dates": [str(date.date()) for date in target_dates],
        "comparison_scope": f"各日期独立对照；两方案使用相同正式E_start、预测、{args.scenarios}条残差场景和真实回放数据",
        "difference": "48h使用288段循环终端；24h使用144段和因果动态终端价值",
        "seed": base.SEED,
        "scenarios": args.scenarios,
        "solver_time_limit_s": args.solver_time_limit,
        "mip_gap": args.mip_gap,
        "files": {
            "comparison_code": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
            "terminal_code": {"path": str(EXPERIMENT_PATH), "sha256": sha256(EXPERIMENT_PATH)},
            "baseline_code": {"path": str(base.BASE_PATH) if hasattr(base, "BASE_PATH") else str(ROOT / '03_代码' / '问题2_求解.py'),
                              "sha256": sha256(ROOT / "03_代码" / "问题2_求解.py")},
            "attachment1": {"path": str(base.INPUT1), "sha256": sha256(base.INPUT1)},
            "attachment2": {"path": str(base.INPUT2), "sha256": sha256(base.INPUT2)},
            "official_daily": {"path": str(baseline_path), "sha256": sha256(baseline_path)},
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl")
        },
        "outputs": {
            str(detail_path): sha256(detail_path),
            str(comparison_path): sha256(comparison_path),
        },
        "command": "python 03_代码/问题2_指定日期两方案对比.py " + " ".join(sys.argv[1:]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题二指定日期：48h循环与24h动态终值受控对比")
    parser.add_argument("--dates", default="2025-03-20,2025-06-21,2025-09-23,2025-12-21")
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--delta-energy", type=float, default=100.0)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--no-gbdt", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "04_结果" / "问题2_指定日期两方案对比"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
