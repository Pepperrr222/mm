from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "03_代码" / "问题2_求解.py"
spec = importlib.util.spec_from_file_location("q2_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载基准模型：{BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def solve_value_milp(
    price: np.ndarray,
    load_s_kw: np.ndarray,
    pv_s_kw: np.ndarray,
    e_initial: float,
    terminal_value: float,
    time_limit: float,
    mip_gap: float,
) -> dict:
    """求解24小时随机MILP，并以 terminal_value * E_end 作为终端价值。"""
    scenarios, horizon = load_s_kw.shape
    if horizon != base.N or pv_s_kw.shape != (scenarios, horizon):
        raise ValueError(f"终端价值实验要求情景形状为(S,{base.N})")
    p = np.asarray(price[:horizon], dtype=float)
    g0, c0, d0, e0 = 0, horizon, 2 * horizon, 3 * horizon
    z0 = e0 + horizon + 1
    h0 = z0 + horizon
    w0 = h0 + scenarios * horizon
    y0 = w0 + scenarios * horizon
    variable_count = y0 + horizon

    def scenario_index(offset: int, s: int, t: int) -> int:
        return offset + s * horizon + t

    objective = np.zeros(variable_count)
    objective[g0:g0 + horizon] = p
    objective[e0 + horizon] = -float(terminal_value)
    for s in range(scenarios):
        objective[h0 + s * horizon:h0 + (s + 1) * horizon] = (
            base.EMERGENCY_MULTIPLIER * p / scenarios
        )

    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[g0:g0 + horizon] = np.max(load_s_kw * base.DT, axis=0) + base.Q_MAX
    upper[c0:c0 + horizon] = base.Q_MAX
    upper[d0:d0 + horizon] = base.Q_MAX
    lower[e0:e0 + horizon + 1] = base.E_MIN
    upper[e0:e0 + horizon + 1] = base.E_MAX
    upper[z0:z0 + horizon] = 1.0
    emergency_big_m = 1.05 * float(np.max(load_s_kw * base.DT))
    upper[h0:h0 + scenarios * horizon] = emergency_big_m
    upper[y0:y0 + horizon] = 1.0

    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[z0:z0 + horizon] = 1
    integrality[y0:y0 + horizon] = 1

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lows: list[float] = []
    highs: list[float] = []

    def add(entries, low=-np.inf, high=np.inf):
        row = len(lows)
        for column, value in entries:
            rows.append(row)
            cols.append(column)
            values.append(value)
        lows.append(low)
        highs.append(high)

    for t in range(horizon):
        add([(e0 + t + 1, 1), (e0 + t, -1), (c0 + t, -base.ETA_C),
             (d0 + t, 1 / base.ETA_D)], 0, 0)
        add([(c0 + t, 1), (z0 + t, -base.Q_MAX)], high=0)
        add([(d0 + t, 1), (z0 + t, base.Q_MAX)], high=base.Q_MAX)
        add([(c0 + t, 1), (y0 + t, base.Q_MAX)], high=base.Q_MAX)
    add([(e0, 1)], e_initial, e_initial)

    for s in range(scenarios):
        net_energy = (load_s_kw[s] - pv_s_kw[s]) * base.DT
        for t in range(horizon):
            add(
                [(g0 + t, 1), (c0 + t, -1), (d0 + t, 1),
                 (scenario_index(h0, s, t), 1),
                 (scenario_index(w0, s, t), -1)],
                net_energy[t], net_energy[t],
            )
            add([(scenario_index(h0, s, t), 1), (y0 + t, -emergency_big_m)], high=0)

    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(lows), variable_count)
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lows), np.asarray(highs)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise RuntimeError(f"终端价值MILP失败 status={result.status}: {result.message}")
    x = result.x
    return {
        "grid": x[g0:g0 + horizon],
        "charge": x[c0:c0 + horizon],
        "discharge": x[d0:d0 + horizon],
        "E": x[e0:e0 + horizon + 1],
        "z": x[z0:z0 + horizon],
        "h": x[h0:h0 + scenarios * horizon].reshape(scenarios, horizon),
        "w": x[w0:w0 + scenarios * horizon].reshape(scenarios, horizon),
        "objective": float(result.fun),
        "elapsed_s": elapsed,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "scenario_count": scenarios,
        "terminal_value_yuan_per_kwh": float(terminal_value),
    }


def estimate_marginal_value(
    price: np.ndarray,
    tomorrow_load_kw: np.ndarray,
    tomorrow_pv_kw: np.ndarray,
    reference_energy: float,
    delta_energy: float,
    time_limit: float,
    mip_gap: float,
) -> dict:
    """只用发布计划时已有的次日预测，以有限差分估计边际储能价值。"""
    low_energy = float(np.clip(reference_energy, base.E_MIN, base.E_MAX - delta_energy))
    high_energy = low_energy + delta_energy
    load = tomorrow_load_kw[None, :]
    pv = tomorrow_pv_kw[None, :]
    low_plan = solve_value_milp(price, load, pv, low_energy, 0.0, time_limit, mip_gap)
    high_plan = solve_value_milp(price, load, pv, high_energy, 0.0, time_limit, mip_gap)
    raw_value = (low_plan["objective"] - high_plan["objective"]) / delta_energy
    economic_cap = base.EMERGENCY_MULTIPLIER * float(np.max(price)) * base.ETA_D
    clipped_value = float(np.clip(raw_value, 0.0, economic_cap))
    return {
        "raw_value": float(raw_value),
        "used_value": clipped_value,
        "E_low_kwh": low_energy,
        "E_high_kwh": high_energy,
        "J_low_yuan": float(low_plan["objective"]),
        "J_high_yuan": float(high_plan["objective"]),
        "low_mip_gap": float(low_plan["mip_gap"]),
        "high_mip_gap": float(high_plan["mip_gap"]),
        "low_solver_time_s": float(low_plan["elapsed_s"]),
        "high_solver_time_s": float(high_plan["elapsed_s"]),
    }


def parse_values(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("固定终端价值必须是非负有限数")
    return values


def run(args) -> dict:
    base.run_execution_unit_tests()
    data = base.read_inputs()
    forecaster = base.CausalForecaster(data, use_gbdt=not args.no_gbdt)
    start_day = int(np.where(data.dates == pd.Timestamp(args.start_date))[0][0])
    if start_day < base.OUTPUT_START:
        raise ValueError("实验起始日不得早于2025-02-01")
    end_day = min(len(data.dates), start_day + args.days)
    fixed_values = parse_values(args.fixed_values)
    mode_names = [f"fixed_v_{value:g}" for value in fixed_values]
    if args.dynamic:
        mode_names.append("dynamic_v")

    baseline_path = ROOT / "04_结果" / "问题2_每日指标.csv"
    baseline_daily = pd.read_csv(baseline_path, parse_dates=["date"])
    baseline_start = baseline_daily.loc[baseline_daily.date == data.dates[start_day], "E_start_kwh"]
    if baseline_start.empty:
        raise ValueError("基准结果中不存在实验起始日")
    initial_energy = float(baseline_start.iloc[0])
    energies = {mode: initial_energy for mode in mode_names}
    records: list[dict] = []
    max_residual = 0.0

    for day in range(1, end_day):
        forecast = forecaster.forecast(day)
        if day < start_day:
            forecaster.observe(day, forecast)
            continue
        load_288, pv_288, sampled = base.generate_scenarios(
            forecast, forecaster, day, args.scenarios
        )
        today_load_s = load_288[:, :base.N]
        today_pv_s = pv_288[:, :base.N]
        for mode in mode_names:
            e_start = energies[mode]
            raw_value = np.nan
            finite_difference = {
                "E_low_kwh": np.nan, "E_high_kwh": np.nan,
                "J_low_yuan": np.nan, "J_high_yuan": np.nan,
                "low_mip_gap": np.nan, "high_mip_gap": np.nan,
                "low_solver_time_s": np.nan, "high_solver_time_s": np.nan,
            }
            if mode == "dynamic_v":
                finite_difference = estimate_marginal_value(
                    data.price,
                    forecast["load"][base.N:],
                    forecast["pv"][base.N:],
                    e_start,
                    args.delta_energy,
                    args.solver_time_limit,
                    args.mip_gap,
                )
                raw_value = finite_difference["raw_value"]
                terminal_value = finite_difference["used_value"]
            else:
                terminal_value = float(mode.removeprefix("fixed_v_"))
            plan = solve_value_milp(
                data.price,
                today_load_s,
                today_pv_s,
                e_start,
                terminal_value,
                args.solver_time_limit,
                args.mip_gap,
            )
            real, e_end = base.replay_day(
                data.dates[day], data.price, data.load_kw[day], data.pv_kw[day], plan, e_start
            )
            checks = base.validate_day(real, plan, e_start, today_load_s, today_pv_s)
            if checks["max_charge_exec_kwh"] > base.Q_MAX + 1e-5:
                raise RuntimeError(f"实际充电功率越界：{checks}")
            if checks["max_discharge_exec_kwh"] > base.Q_MAX + 1e-5:
                raise RuntimeError(f"实际放电功率越界：{checks}")
            real_balance = (
                real.grid_kwh + real.pv_kw * base.DT + real.discharge_exec_kwh + real.emergency_kwh
                - real.load_kw * base.DT - real.charge_exec_kwh - real.surplus_kwh
            )
            residual = max(
                checks["max_plan_state_residual_kwh"],
                checks["max_scenario_balance_residual_kwh"],
                checks["max_real_state_residual_kwh"],
                float(np.max(np.abs(real_balance))),
            )
            max_residual = max(max_residual, residual)
            records.append({
                "date": data.dates[day],
                "mode": mode,
                "terminal_value_raw_yuan_per_kwh": raw_value,
                "terminal_value_used_yuan_per_kwh": terminal_value,
                "finite_diff_E_low_kwh": finite_difference["E_low_kwh"],
                "finite_diff_E_high_kwh": finite_difference["E_high_kwh"],
                "finite_diff_J_low_yuan": finite_difference["J_low_yuan"],
                "finite_diff_J_high_yuan": finite_difference["J_high_yuan"],
                "finite_diff_low_mip_gap": finite_difference["low_mip_gap"],
                "finite_diff_high_mip_gap": finite_difference["high_mip_gap"],
                "finite_diff_low_solver_time_s": finite_difference["low_solver_time_s"],
                "finite_diff_high_solver_time_s": finite_difference["high_solver_time_s"],
                "E_start_kwh": e_start,
                "E_plan_end_kwh": float(plan["E"][-1]),
                "E_real_end_kwh": e_end,
                "real_E_min_kwh": checks["real_E_min_kwh"],
                "real_E_max_kwh": checks["real_E_max_kwh"],
                "plan_cost_yuan": float(real.plan_cost_yuan.sum()),
                "emergency_cost_yuan": float(real.emergency_cost_yuan.sum()),
                "total_cost_yuan": float(real.plan_cost_yuan.sum() + real.emergency_cost_yuan.sum()),
                "emergency_kwh": float(real.emergency_kwh.sum()),
                "surplus_kwh": float(real.surplus_kwh.sum()),
                "charge_exec_kwh": float(real.charge_exec_kwh.sum()),
                "discharge_exec_kwh": float(real.discharge_exec_kwh.sum()),
                "solver_elapsed_s": float(plan["elapsed_s"]),
                "solver_mip_gap": float(plan["mip_gap"]),
                "sampled_residual_days": json.dumps(sampled, ensure_ascii=False),
                "sampled_residual_dates": json.dumps(
                    [str(data.dates[index].date()) for index in sampled], ensure_ascii=False
                ),
                "training_cutoff": str(data.dates[forecaster.training_cutoffs[day]].date()),
                "scenario_count": int(plan["scenario_count"]),
                "horizon_hours": 24,
                "max_residual_kwh": residual,
                "max_real_state_residual_kwh": checks["max_real_state_residual_kwh"],
                "emergency_charge_exec_periods": checks["emergency_charge_exec_periods"],
                "emergency_charge_scenario_periods": checks["emergency_charge_scenario_periods"],
                "emergency_surplus_scenario_periods": checks["emergency_surplus_scenario_periods"],
                "max_charge_exec_kwh": checks["max_charge_exec_kwh"],
                "max_discharge_exec_kwh": checks["max_discharge_exec_kwh"],
            })
            energies[mode] = e_end
        forecaster.observe(day, forecast)
        print(f"[{data.dates[day].date()}] " + ", ".join(
            f"{mode}: E={energies[mode]:.1f}" for mode in mode_names
        ), flush=True)

    detail = pd.DataFrame(records)
    summary = detail.groupby("mode", as_index=False).agg(
        days=("date", "count"),
        total_cost_yuan=("total_cost_yuan", "sum"),
        plan_cost_yuan=("plan_cost_yuan", "sum"),
        emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        emergency_kwh=("emergency_kwh", "sum"),
        surplus_kwh=("surplus_kwh", "sum"),
        charge_exec_kwh=("charge_exec_kwh", "sum"),
        discharge_exec_kwh=("discharge_exec_kwh", "sum"),
        E_start_kwh=("E_start_kwh", "first"),
        E_end_kwh=("E_real_end_kwh", "last"),
        E_min_kwh=("real_E_min_kwh", "min"),
        E_max_kwh=("real_E_max_kwh", "max"),
        mean_terminal_value=("terminal_value_used_yuan_per_kwh", "mean"),
        solver_time_s=("solver_elapsed_s", "sum"),
        max_residual_kwh=("max_residual_kwh", "max"),
    )
    controlled = summary.loc[summary["mode"] == "fixed_v_0"]
    if controlled.empty:
        raise ValueError("受控比较必须在 --fixed-values 中包含0")
    controlled_cost = float(controlled.iloc[0]["total_cost_yuan"])
    controlled_emergency = float(controlled.iloc[0]["emergency_cost_yuan"])
    summary["cost_delta_vs_fixed_v0_yuan"] = summary.total_cost_yuan - controlled_cost
    summary["emergency_cost_delta_vs_fixed_v0_yuan"] = (
        summary.emergency_cost_yuan - controlled_emergency
    )

    experiment_dates = data.dates[start_day:end_day]
    official_slice = baseline_daily[baseline_daily.date.isin(experiment_dates)]
    official_reference = {
        "role": "背景参照，不用于识别终端价值的单因素因果效应",
        "horizon_hours": 48,
        "terminal_contract": "E_288=E_0",
        "days": int(len(official_slice)),
        "total_cost_yuan": float(official_slice.total_cost_yuan.sum()),
        "scenario_count_used": sorted({int(x) for x in official_slice.scenario_count_used}),
        "fallback_counts": {str(k): int(v) for k, v in official_slice.fallback.value_counts().items()},
        "residual_paths": "正式模型按其复现清单抽样，与本实验24小时受控情景不相同",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "问题2_终端价值实验_逐日.csv"
    summary_path = output_dir / "问题2_终端价值实验_汇总.csv"
    manifest_path = output_dir / "问题2_终端价值实验_复现清单.json"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = {
        "start_date": args.start_date,
        "days": int(end_day - start_day),
        "scenarios": args.scenarios,
        "fixed_terminal_values": fixed_values,
        "dynamic": args.dynamic,
        "delta_energy_kwh": args.delta_energy,
        "initial_energy_kwh": initial_energy,
        "information_rule": "动态价值仅使用当日0:00可得的未来一天预测，不使用未来真实值",
        "baseline": "正式48小时循环终端模型；实验分支为24小时线性终端价值模型",
        "controlled_comparison": "实验分支均使用同一预测、同一残差情景、同一求解门限；fixed_v_0是单因素受控基线",
        "official_reference": official_reference,
        "max_experiment_residual_kwh": max_residual,
        "seed": base.SEED,
        "files": {
            "experiment_code": {"path": str(Path(__file__)), "sha256": base.sha256(Path(__file__))},
            "baseline_code": {"path": str(BASE_PATH), "sha256": base.sha256(BASE_PATH)},
            "attachment1": {"path": str(base.INPUT1), "sha256": base.sha256(base.INPUT1)},
            "attachment2": {"path": str(base.INPUT2), "sha256": base.sha256(base.INPUT2)},
            "official_daily": {"path": str(baseline_path), "sha256": base.sha256(baseline_path)},
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl")
        },
        "command": "python 03_代码/问题2_终端价值实验.py " + " ".join(args.raw_argv),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return {"detail": str(detail_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题二：日末储能价值可行性实验")
    parser.add_argument("--start-date", default="2025-02-01")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--fixed-values", default="0,0.5,1.0")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--delta-energy", type=float, default=100.0)
    parser.add_argument("--solver-time-limit", type=float, default=10.0)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--no-gbdt", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "04_结果" / "问题2_终端价值实验"),
    )
    args = parser.parse_args()
    args.raw_argv = sys.argv[1:]
    return args


if __name__ == "__main__":
    run(parse_args())
