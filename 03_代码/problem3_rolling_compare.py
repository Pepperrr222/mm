"""Problem 3: causal rolling comparison of 48-hour and 24-hour horizons.

Both policies re-optimise every six hours.  Only the next six hours are
executed; later decisions are discarded at the next update.  Forecasts are
generated from data available before the day starts and the realised data are
used only for replay/settlement.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("03"))
Q2_PATH = next(p for p in CODE_DIR.iterdir() if p.suffix == ".py" and p.stat().st_size > 30000)
spec = importlib.util.spec_from_file_location("q2_base", Q2_PATH)
q2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(q2)

OUT = ROOT / "04_结果" / "问题3_48h与24h滚动对比"
OUT.mkdir(parents=True, exist_ok=True)
DT = q2.DT
N = q2.N
ETA_C = q2.ETA_C
ETA_D = q2.ETA_D
E_MIN, E_MAX = q2.E_MIN, q2.E_MAX
Q_MAX = q2.Q_MAX
EMERGENCY_MULTIPLIER = q2.EMERGENCY_MULTIPLIER
UPDATE = 36  # six hours, in ten-minute intervals
SEED = 20260912


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def solve_horizon(price: np.ndarray, load_kw: np.ndarray, pv_kw: np.ndarray,
                  e_start: float, time_limit: float = 20.0) -> dict:
    """Deterministic storage MILP for one rolling horizon."""
    H = len(load_kw)
    if len(price) != H or len(pv_kw) != H:
        raise ValueError("horizon arrays must have equal length")
    # g, c, d, E, z, h, w.  A single robust scenario is used so both policies
    # differ only by horizon length, not by scenario count.
    g0, c0, d0, e0, z0, h0, w0 = 0, H, 2 * H, 3 * H, 4 * H + 1, 5 * H + 1, 6 * H + 1
    nv = 7 * H + 2
    obj = np.zeros(nv)
    obj[g0:g0 + H] = price
    obj[h0:h0 + H] = EMERGENCY_MULTIPLIER * price
    lo = np.zeros(nv)
    hi = np.full(nv, np.inf)
    hi[g0:g0 + H] = np.maximum(load_kw * DT - pv_kw * DT, 0) + Q_MAX
    hi[c0:c0 + H] = Q_MAX
    hi[d0:d0 + H] = Q_MAX
    lo[e0:e0 + H + 1] = E_MIN
    hi[e0:e0 + H + 1] = E_MAX
    hi[z0:z0 + H] = 1
    hi[h0:h0 + H] = 1.05 * float(np.max(load_kw * DT))
    hi[w0:w0 + H] = 1.05 * float(np.max(load_kw * DT + pv_kw * DT))
    # The linear relaxation is exact for this dispatch objective: simultaneous
    # charging/discharging is strictly dominated by round-trip losses and a
    # positive grid price.  Keeping z continuous makes the 8,000+ rolling
    # solves for the annual experiment practical while preserving all energy
    # and cost constraints.
    integer = np.zeros(nv, dtype=np.uint8)
    rows, cols, vals, low, upp = [], [], [], [], []

    def add(entries, a=-np.inf, b=np.inf):
        r = len(low)
        for c, v in entries:
            rows.append(r); cols.append(c); vals.append(v)
        low.append(a); upp.append(b)

    for t in range(H):
        add([(e0 + t + 1, 1), (e0 + t, -1), (c0 + t, -ETA_C), (d0 + t, 1 / ETA_D)], 0, 0)
        add([(c0 + t, 1), (z0 + t, -Q_MAX)], b=0)
        add([(d0 + t, 1), (z0 + t, Q_MAX)], b=Q_MAX)
        add([(g0 + t, 1), (c0 + t, -1), (d0 + t, 1),
             (h0 + t, 1), (w0 + t, -1)],
            (load_kw[t] - pv_kw[t]) * DT, (load_kw[t] - pv_kw[t]) * DT)
    add([(e0, 1)], e_start, e_start)
    # End-of-horizon inventory neutrality prevents the finite horizon from
    # emptying the battery merely to reduce its last few hours' cost.
    add([(e0 + H, 1)], e_start, e_start)
    mat = coo_matrix((vals, (rows, cols)), shape=(len(low), nv)).tocsr()
    started = time.perf_counter()
    res = milp(obj, integrality=integer, bounds=Bounds(lo, hi),
               constraints=LinearConstraint(mat, np.asarray(low), np.asarray(upp)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-7, "presolve": True})
    elapsed = time.perf_counter() - started
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP failed: status={res.status}, {res.message}")
    x = res.x
    return {"grid": x[g0:g0 + H], "charge": x[c0:c0 + H],
            "discharge": x[d0:d0 + H], "E": x[e0:e0 + H + 1],
            "objective": float(res.fun), "elapsed_s": elapsed,
            "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0)}


def extend_forecast(fc: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a 48-hour causal forecast; day+2 uses the latest predicted day
    pattern only when a 48-hour window reaches beyond the two-day forecast."""
    load, pv = np.asarray(fc["load"], float), np.asarray(fc["pv"], float)
    if horizon <= len(load):
        return load[:horizon], pv[:horizon]
    tail_l, tail_p = load[N:2 * N], pv[N:2 * N]
    while len(load) < horizon:
        load = np.concatenate([load, tail_l])
        pv = np.concatenate([pv, tail_p])
    return load[:horizon], pv[:horizon]


def load_attachment3_forecast(root: Path, dates: pd.DatetimeIndex) -> dict:
    """Read the four intraday PV forecast releases in attachment 3.

    The workbook contains one row per (day, issue time), 24 hourly values.
    Each hourly value is expanded to six ten-minute intervals.  Missing issue
    rows are deliberately rejected rather than silently replaced by actual PV.
    """
    one = next(p for p in root.iterdir() if p.is_dir() and p.name.startswith("01"))
    raw = next(p for p in one.rglob("*.xlsx")
               if "result" not in p.name.lower() and pd.read_excel(p, header=None, nrows=2).shape[1] == 26)
    df = pd.read_excel(raw, header=0)
    if df.shape[1] < 26:
        raise ValueError("attachment 3 must contain date, issue time and 24 forecasts")
    date_col, issue_col = df.columns[:2]
    out = {}
    cur_date = None
    for _, row in df.iterrows():
        if pd.notna(row[date_col]):
            cur_date = pd.Timestamp(str(row[date_col]).strip()).normalize()
        if cur_date is None or pd.isna(row[issue_col]):
            continue
        issue = str(row[issue_col]).strip()
        hh, mm = issue.split(":")[:2]
        k = (int(hh) * 60 + int(mm)) // 10
        vals = pd.to_numeric(row.iloc[2:26], errors="coerce").to_numpy(float)
        if vals.size != 24 or not np.isfinite(vals).all():
            raise ValueError(f"invalid attachment 3 row at {cur_date} {issue}")
        out[(cur_date, k)] = np.repeat(np.maximum(vals, 0.0), 6)
    for d in dates[31:]:
        for k in (0, 36, 72, 108):
            if (pd.Timestamp(d).normalize(), k) not in out:
                raise ValueError(f"missing attachment 3 forecast for {d.date()} k={k}")
    return out


def causal_window_forecast(data, fc, pv_fc, day: int, k: int, horizon: int):
    """Build a correctly aligned local window beginning at update slot k."""
    load_full = np.asarray(fc["load"], float)
    # The issue-k attachment-3 row is already a local forecast whose first
    # value follows the release.  Do not slice it again by k.  Beyond the
    # supplied 24 hours, use the causal day+1 proxy from the same forecast
    # object (never the next day's future release).
    date = pd.Timestamp(data.dates[day]).normalize()
    pv_today = pv_fc[(date, k)]
    pv_next = np.asarray(fc["pv"][N:2 * N], float)
    pv_full = np.concatenate([pv_today, pv_next])
    while len(pv_full) < horizon:
        pv_full = np.concatenate([pv_full, pv_next])
    load_local = np.concatenate([load_full[k:2 * N], load_full[N:2 * N]])
    while len(load_local) < horizon:
        load_local = np.concatenate([load_local, load_full[N:2 * N]])
    if len(load_local) < horizon:
        load_full = np.concatenate([load_full, load_full[N:2 * N]])
    return load_local[:horizon], pv_full[:horizon]


def execute(e, g, c_plan, d_plan, load_e, pv_e):
    d_cap = max(0.0, (e - E_MIN) * ETA_D)
    deficit = max(0.0, load_e - g - pv_e)
    d = min(max(d_plan, 0.0), d_cap, Q_MAX, deficit)
    surplus = max(0.0, g + pv_e - load_e)
    c_cap = max(0.0, (E_MAX - e) / ETA_C)
    c = min(max(c_plan, 0.0), surplus, Q_MAX, c_cap)
    h = max(0.0, load_e + c - g - pv_e - d)
    w = max(0.0, g + pv_e + d - load_e - c)
    en = e + ETA_C * c - d / ETA_D
    return c, d, h, w, en


def run_policy(data, name: str, horizon: int, first_day: int, last_day: int,
               solver_limit: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The rolling comparison uses the causal lag/mean ensemble.  Keeping the
    # optional tree learner off makes the two policies reproducible and keeps
    # the annual four-update backtest tractable; both policies use the same
    # forecaster, so this choice cannot favour either horizon.
    forecaster = q2.CausalForecaster(data, use_gbdt=False)
    pv_fc = load_attachment3_forecast(ROOT, data.dates)
    detail, daily = [], []
    e_real = float(q2.E_INITIAL)
    for day in range(first_day, last_day + 1):
        fc = forecaster.forecast(day)
        e = e_real
        day_start = time.perf_counter()
        rows = []
        effective = np.zeros(N)
        baseline = None
        adjustment_total = 0.0
        for k in range(0, N, UPDATE):
            H = horizon
            load_f, pv_f = causal_window_forecast(data, fc, pv_fc, day, k, H)
            local_price = np.asarray([data.price[(k + tau) % N] for tau in range(H)])
            plan = solve_horizon(local_price, load_f, pv_f, e, solver_limit)
            if k == 0:
                baseline = plan["grid"][:N].copy()
                effective[:] = baseline
                up = down = 0.0
            else:
                old = effective[k:N].copy()
                new = plan["grid"][:N - k].copy()
                up = float(np.sum(data.price[k:N] * np.maximum(new - old, 0.0) * 1.5))
                down = float(np.sum(data.price[k:N] * np.maximum(old - new, 0.0) * 0.5))
                effective[k:N] = new
            adjustment_total += up - down
            for j in range(min(UPDATE, N - k)):
                t = k + j
                c, d, h, w, en = execute(
                    e, float(effective[t]), float(plan["charge"][j]),
                    float(plan["discharge"][j]), data.load_kw[day, t] * DT,
                    data.pv_kw[day, t] * DT,
                )
                rows.append({"date": data.dates[day], "policy": name, "t": t,
                             "update_slot": k, "price": data.price[t],
                             "load_kw": data.load_kw[day, t], "pv_kw": data.pv_kw[day, t],
                             "grid_kwh": effective[t], "charge_exec_kwh": c,
                             "discharge_exec_kwh": d, "emergency_kwh": h, "surplus_kwh": w,
                             "E_start_kwh": e, "E_end_kwh": en,
                             "baseline_cost_yuan": data.price[t] * baseline[t],
                             "adjustment_cost_yuan": (data.price[t] * (1.5 * max(effective[t] - (baseline[t] if k == 0 else old[j]), 0.0)
                                                   - 0.5 * max((baseline[t] if k == 0 else old[j]) - effective[t], 0.0))
                                                   if k > 0 else 0.0),
                             "plan_cost_yuan": data.price[t] * effective[t],
                             "emergency_cost_yuan": EMERGENCY_MULTIPLIER * data.price[t] * h,
                             "up_adjustment_kwh": max(effective[t] - (baseline[t] if k == 0 else old[j]), 0.0) if k > 0 else 0.0,
                             "down_adjustment_kwh": max((baseline[t] if k == 0 else old[j]) - effective[t], 0.0) if k > 0 else 0.0,
                             "issue_time_slot": k, "training_cutoff": data.dates[day - 1],
                             "solver_elapsed_s": plan["elapsed_s"], "solver_mip_gap": plan["mip_gap"],
                             "forecast_horizon_h": horizon / 6})
                e = en
        forecaster.observe(day, fc)
        e_real = e
        frame = pd.DataFrame(rows)
        detail.append(frame)
        regular_settlement = float(frame.baseline_cost_yuan.sum() + adjustment_total)
        daily.append({"date": data.dates[day], "policy": name,
                      "total_cost_yuan": regular_settlement + frame.emergency_cost_yuan.sum(),
                      "regular_cost_yuan": regular_settlement,
                      "baseline_cost_yuan": frame.baseline_cost_yuan.sum(),
                      "adjustment_cost_yuan": adjustment_total,
                      "emergency_cost_yuan": frame.emergency_cost_yuan.sum(),
                      "emergency_kwh": frame.emergency_kwh.sum(),
                      "emergency_intervals": int((frame.emergency_kwh > 1e-7).sum()),
                      "end_storage_kwh": float(e),
                      "solver_time_s": frame.solver_elapsed_s.sum(),
                      "max_mip_gap": float(frame.solver_mip_gap.max()),
                      "wall_time_s": time.perf_counter() - day_start})
    return pd.concat(detail, ignore_index=True), pd.DataFrame(daily)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-day", type=int, default=31)
    ap.add_argument("--end-day", type=int, default=364)
    ap.add_argument("--solver-time-limit", type=float, default=20.0)
    args = ap.parse_args()
    data = q2.read_inputs()
    one = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("01"))
    input_files = sorted(one.rglob("*.xlsx"))
    attachment3 = next(p for p in input_files
                       if "result" not in p.name.lower() and pd.read_excel(p, header=None, nrows=2).shape[1] == 26)
    detail_parts, daily_parts = [], []
    for name, horizon in (("48h", 2 * N), ("24h", N)):
        d, m = run_policy(data, name, horizon, args.start_day, args.end_day, args.solver_time_limit)
        detail_parts.append(d); daily_parts.append(m)
        d.to_csv(OUT / f"问题3_{name}_逐时段结果.csv", index=False, encoding="utf-8-sig")
        m.to_csv(OUT / f"问题3_{name}_每日指标.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(detail_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary = daily.groupby("policy").agg(
        days=("date", "count"), total_cost_yuan=("total_cost_yuan", "sum"),
        regular_cost_yuan=("regular_cost_yuan", "sum"), emergency_cost_yuan=("emergency_cost_yuan", "sum"),
        emergency_kwh=("emergency_kwh", "sum"), emergency_intervals=("emergency_intervals", "sum"),
        mean_end_storage_kwh=("end_storage_kwh", "mean"), min_end_storage_kwh=("end_storage_kwh", "min"),
        max_mip_gap=("max_mip_gap", "max"), total_solver_time_s=("solver_time_s", "sum"),
    ).reset_index()
    summary.to_csv(OUT / "问题3_两方案汇总.csv", index=False, encoding="utf-8-sig")
    pivot = daily.pivot(index="date", columns="policy", values="total_cost_yuan").reset_index()
    pivot["delta_24h_minus_48h_yuan"] = pivot["24h"] - pivot["48h"]
    pivot.to_csv(OUT / "问题3_每日配对差.csv", index=False, encoding="utf-8-sig")
    by = summary.set_index("policy")
    delta = float(by.loc["24h", "total_cost_yuan"] - by.loc["48h", "total_cost_yuan"])
    pct = 100 * delta / float(by.loc["48h", "total_cost_yuan"])
    win_rate = float((pivot["delta_24h_minus_48h_yuan"] < 0).mean()) if len(pivot) else float("nan")
    rec = "24小时方案" if delta < 0 else "48小时方案"
    report = f"""# 第三问两种滚动方案实验报告

## 1. 实验目的

比较两种完全因果的滚动调度方案：方案 A 在每个更新时刻预测未来 48 小时，方案 B 预测未来 24 小时；两者均每 6 小时重新优化，但只执行接下来 6 小时，已经执行的购电承诺不可回改。

## 2. 统一实验口径

- 回测期：{data.dates[args.start_day].date()} 至 {data.dates[args.end_day].date()}，共 {len(pivot)} 天。
- 预测：负荷使用仅基于更新前历史的因果预测；光伏使用附件 3 在 0:00、6:00、12:00、18:00 发布的对应预测。48 小时窗口的第二天以后使用最新可得的日模式扩展，未使用未来实际值。
- 执行：每次只冻结 36 个十分钟时段；实际负荷、实际光伏和上一时刻真实储能状态用于回放，缺口按 5 倍电价紧急购电。
- 结算：保留 0:00 基准计划；每次更新相对上一次有效承诺计算上调 1.5 倍、下调按原价 50% 的结算费用。
- 储能：容量范围 [{E_MIN:.0f}, {E_MAX:.0f}] kWh，充放电效率 {ETA_C:.1f}，两方案使用同一储能约束和实际状态传递。

## 3. 结果

|方案|总费用/元|常规结算/元|紧急购电费用/元|紧急购电量/kWh|紧急时段数|平均日末储能/kWh|最大求解间隙|
|---|---:|---:|---:|---:|---:|---:|---:|
|48小时|{by.loc['48h','total_cost_yuan']:.2f}|{by.loc['48h','regular_cost_yuan']:.2f}|{by.loc['48h','emergency_cost_yuan']:.2f}|{by.loc['48h','emergency_kwh']:.2f}|{by.loc['48h','emergency_intervals']:.0f}|{by.loc['48h','mean_end_storage_kwh']:.2f}|{by.loc['48h','max_mip_gap']:.2g}|
|24小时|{by.loc['24h','total_cost_yuan']:.2f}|{by.loc['24h','regular_cost_yuan']:.2f}|{by.loc['24h','emergency_cost_yuan']:.2f}|{by.loc['24h','emergency_kwh']:.2f}|{by.loc['24h','emergency_intervals']:.0f}|{by.loc['24h','mean_end_storage_kwh']:.2f}|{by.loc['24h','max_mip_gap']:.2g}|

24小时相对48小时的总费用差为 **{delta:.2f} 元（{pct:.3f}%）**；按日配对比较，24小时方案费用更低的日期比例为 **{win_rate:.1%}**。因此在本简化回测中，{rec}可作为辅助实验的候选基线，不能替代第三问正式合同模型的主方案结论。

## 4. 解释与限制

48小时方案能看到更远的光伏与负荷变化，适合储能跨日预留；24小时方案窗口内预测更可靠、模型规模更小、更新后更快纠偏。最终推荐以真实结算费用和紧急购电为主，而不是只比较预测 MAE。若改动终端储能价值、场景数或预测器，应重新运行本脚本并更新本报告。

本实验是两种窗口长度的可复现实验基线：计划层采用单一因果预测场景，充放电互斥变量使用连续松弛（正电价和往返效率损失下同时充放电为支配劣解），并将整点预测按六个十分钟段展开；调整费用用于结算记录，尚未进入滚动优化目标。因此它只适合比较窗口长度的启发式影响，不承担第三问正式主结论。正式提交还应加入历史整日残差场景、将上/下调变量直接纳入目标、严格互斥约束及分段线性插值敏感性复核。

## 5. 复现

```powershell
python 03_代码/problem3_rolling_compare.py --start-day {args.start_day} --end-day {args.end_day} --solver-time-limit 20
```
"""
    (OUT / "问题3_两方案对比报告.md").write_text(report, encoding="utf-8")
    meta = {"script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
            "input_script": str(Q2_PATH), "input_sha256": sha256(Q2_PATH),
            "input_xlsx_sha256": {str(p): sha256(p) for p in input_files if "result" not in p.name.lower()},
            "attachment3": str(attachment3),
            "start_day_index": args.start_day, "end_day_index": args.end_day,
            "execution_step_h": 6, "horizons_h": [48, 24], "seed": SEED,
            "model": "causal daily forecast + deterministic storage MILP + actual six-hour replay",
            "no_future_actuals_in_plan": True, "outputs": [p.name for p in OUT.iterdir()]}
    (OUT / "问题3_复现清单.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
