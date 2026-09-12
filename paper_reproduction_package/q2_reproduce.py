"""Reproduce the paper's Question 2 stochastic plan and value-function replay."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from q2_forecast import q2

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHECKPOINT = RESULTS / "q2_checkpoint.csv"
N = 144
ETA = 0.9
EMIN, EMAX, E0 = 1200.0, 10800.0, 6000.0
S = 5000.0 / 6.0
V = 0.481548


@dataclass
class PWL:
    x: np.ndarray
    y: np.ndarray

    def value(self, z):
        return np.interp(z, self.x, self.y)


def _unique(values):
    a = np.asarray(values, float)
    a = a[(a >= EMIN - 1e-9) & (a <= EMAX + 1e-9)]
    return np.unique(np.round(np.clip(a, EMIN, EMAX), 9))


def _simplify(x, y, tol=1e-9):
    """Return the convex lower hull; Bellman values are convex PWL functions."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = []
    for i in range(len(x)):
        while len(keep) >= 2:
            a, b = keep[-2], keep[-1]
            s1 = (y[b] - y[a]) / (x[b] - x[a])
            s2 = (y[i] - y[b]) / (x[i] - x[b])
            if s1 < s2 - tol:
                break
            keep.pop()
        keep.append(i)
    return PWL(x[keep], y[keep])

def _largest_argmin(f: PWL, slope: float) -> float:
    val = f.y + slope * f.x
    best = val.min()
    return float(f.x[np.where(val <= best + 1e-9)[0][-1]])


def bellman(f: PWL, r: float, price: float) -> PWL:
    """Exact convex PWL Bellman step for one residual path and one interval."""
    if r <= 0:
        a = ETA * min(S, -r)
        xs = _unique(np.r_[EMIN, EMAX, EMAX - a, f.x - a])
        z = np.minimum(EMAX, xs + a)
        return _simplify(xs, f.value(z))
    a = min(S, r) / ETA
    k = 5.0 * price * ETA
    reserve = _largest_argmin(f, k)
    xs = _unique(np.r_[EMIN, EMAX, reserve, reserve + a, f.x, f.x + a])
    z = np.where(xs < reserve, xs, np.where(xs > reserve + a, xs - a, reserve))
    z = np.clip(z, np.maximum(EMIN, xs - a), xs)
    y = 5.0 * price * r - k * xs + f.value(z) + k * z
    return _simplify(xs, y)


def average_pwl(fs: list[PWL]) -> PWL:
    xs = _unique(np.concatenate([f.x for f in fs]))
    ys = np.mean([f.value(xs) for f in fs], axis=0)
    return _simplify(xs, ys)


def reserve_levels(price: np.ndarray, residual_s: np.ndarray) -> np.ndarray:
    """Paper equations (22)-(24),(29)-(30), retaining continuous PWL values."""
    m = residual_s.shape[0]
    future = [PWL(np.array([EMIN, EMAX]), np.array([-V * EMIN, -V * EMAX])) for _ in range(m)]
    reserve = np.empty(N)
    for t in range(N - 1, -1, -1):
        mean_future = average_pwl(future)
        reserve[t] = _largest_argmin(mean_future, 5.0 * price[t] * ETA)
        future = [bellman(future[j], float(residual_s[j, t]), float(price[t])) for j in range(m)]
    return reserve


def solve_plan(price: np.ndarray, load_s_kw: np.ndarray, pv_s_kw: np.ndarray, initial_soc: float):
    """Paper LP (20)-(21): common g and scenario-specific b,C,D,E,U."""
    m = load_s_kw.shape[0]
    block = 4 * N + (N + 1)
    total = N + m * block
    c = np.zeros(total)
    c[:N] = price
    bounds = [(0.0, None)] * total

    def ix(s, part, t):
        base = N + s * block
        offsets = {"b": 0, "C": N, "D": 2*N, "E": 3*N, "U": 4*N + 1}
        return base + offsets[part] + t

    for s in range(m):
        prob = 1.0 / m
        for t in range(N):
            c[ix(s, "b", t)] = prob * 5.0 * price[t]
            bounds[ix(s, "C", t)] = (0.0, S)
            bounds[ix(s, "D", t)] = (0.0, S)
        for t in range(N + 1):
            bounds[ix(s, "E", t)] = (EMIN, EMAX)
        bounds[ix(s, "E", 0)] = (initial_soc, initial_soc)
        c[ix(s, "E", N)] = -prob * V

    rows, cols, vals, rhs = [], [], [], []
    row = 0
    demand_e = (load_s_kw - pv_s_kw) / 6.0
    for s in range(m):
        for t in range(N):
            # g+b+D-C-U = net demand
            for col, val in ((t, 1), (ix(s,"b",t),1), (ix(s,"D",t),1),
                             (ix(s,"C",t),-1), (ix(s,"U",t),-1)):
                rows.append(row); cols.append(col); vals.append(val)
            rhs.append(float(demand_e[s, t])); row += 1
        for t in range(N):
            # E[t+1]-E[t]-eta*C+D/eta = 0
            for col, val in ((ix(s,"E",t+1),1), (ix(s,"E",t),-1),
                             (ix(s,"C",t),-ETA), (ix(s,"D",t),1/ETA)):
                rows.append(row); cols.append(col); vals.append(val)
            rhs.append(0.0); row += 1
    A = coo_matrix((vals, (rows, cols)), shape=(row, total)).tocsr()
    result = linprog(c, A_eq=A, b_eq=np.asarray(rhs), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"planning LP failed: {result.message}")
    x = result.x
    simultaneous = 0
    for s in range(m):
        C = x[[ix(s,"C",t) for t in range(N)]]
        D = x[[ix(s,"D",t) for t in range(N)]]
        simultaneous += int(np.sum((C > 1e-7) & (D > 1e-7)))
    return x[:N], float(result.fun), simultaneous


def scenario_set(day: int, data, load_hat, pv_hat):
    history = np.arange(max(1, day - 30), day)
    load_s = np.maximum(0.0, load_hat[day][None, :] + data.load_kw[history] - load_hat[history])
    pv_s = np.maximum(0.0, pv_hat[day][None, :] + data.pv_kw[history] - pv_hat[history])
    return load_s, pv_s, history


def replay_day(price, load_kw, pv_kw, g, reserve, initial_soc):
    e = float(initial_soc)
    emergency = waste = charge = discharge = 0.0
    max_balance = 0.0
    for t in range(N):
        r = (load_kw[t] - pv_kw[t]) / 6.0 - g[t]
        if r > 0:
            d = min(r, S, ETA * max(e - reserve[t], 0.0))
            c = 0.0
        else:
            c = min(-r, S, max(EMAX - e, 0.0) / ETA)
            d = 0.0
        h = max(r + c - d, 0.0)
        w = max(-r - c + d, 0.0)
        e2 = e + ETA * c - d / ETA
        bal = g[t] + d + h - ((load_kw[t]-pv_kw[t])/6.0 + c + w)
        max_balance = max(max_balance, abs(bal))
        emergency += h; waste += w; charge += c; discharge += d; e = e2
    if not EMIN - 1e-7 <= e <= EMAX + 1e-7 or max_balance > 1e-7:
        raise AssertionError((e, max_balance))
    plan_cost = float(price @ g)
    emergency_cost = float(5.0 * np.sum(price * np.maximum((load_kw-pv_kw)/6.0 + 0, 0)))  # replaced below
    # Recompute priced emergencies because the aggregate above is not enough.
    e = float(initial_soc); emergency_cost = 0.0
    for t in range(N):
        r = (load_kw[t] - pv_kw[t]) / 6.0 - g[t]
        d = min(r, S, ETA * max(e-reserve[t],0.0)) if r > 0 else 0.0
        c = min(-r, S, max(EMAX-e,0.0)/ETA) if r <= 0 else 0.0
        h = max(r+c-d,0.0); emergency_cost += 5.0*price[t]*h
        e += ETA*c-d/ETA
    return dict(plan_cost=plan_cost, emergency_cost=emergency_cost,
                total_cost=plan_cost+emergency_cost, planned_energy=float(g.sum()),
                emergency_energy=emergency, waste_energy=waste, charge_energy=charge,
                discharge_energy=discharge, final_soc=e, max_balance_residual=max_balance)


def run(start_day: int, end_day: int, fresh: bool = False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = q2.read_inputs()
    cache = np.load(HERE / "cache_q2_forecasts.npz")
    load_hat, pv_hat = cache["load_kw"], cache["pv_kw"]
    rows = []
    soc = E0
    if CHECKPOINT.exists() and not fresh:
        old = pd.read_csv(CHECKPOINT)
        old["date"] = pd.to_datetime(old.date)
        usable = old[(old.day_index >= start_day) & (old.day_index < end_day)]
        rows = usable.to_dict("records")
        if rows:
            soc = float(rows[-1]["final_soc"])
            start_day = int(rows[-1]["day_index"]) + 1
    started = time.perf_counter()
    for day in range(start_day, end_day):
        load_s, pv_s, hist = scenario_set(day, data, load_hat, pv_hat)
        print(f"{data.dates[day].date()} planning LP start", flush=True)
        g, lp_obj, simultaneous = solve_plan(data.price, load_s, pv_s, soc)
        print(f"planning LP done objective={lp_obj:.3f}", flush=True)
        residual_s = (load_s - pv_s) / 6.0 - g[None, :]
        print("value recursion start", flush=True)
        reserve = reserve_levels(data.price, residual_s)
        print("value recursion done", flush=True)
        metrics = replay_day(data.price, data.load_kw[day], data.pv_kw[day], g, reserve, soc)
        metrics.update(day_index=day, date=str(data.dates[day].date()), initial_soc=soc,
                       lp_objective=lp_obj, scenario_count=len(hist), simultaneous_cd=simultaneous,
                       runtime_seconds=time.perf_counter()-started)
        rows.append(metrics); soc = metrics["final_soc"]
        pd.DataFrame(rows).to_csv(CHECKPOINT, index=False, encoding="utf-8-sig")
        print(f"{metrics['date']} total={metrics['total_cost']:.2f} soc={soc:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    summary = {"days": len(frame), "start": str(frame.date.iloc[0]), "end": str(frame.date.iloc[-1]),
               "plan_cost": float(frame.plan_cost.sum()), "emergency_cost": float(frame.emergency_cost.sum()),
               "total_cost": float(frame.total_cost.sum()), "planned_energy": float(frame.planned_energy.sum()),
               "emergency_energy": float(frame.emergency_energy.sum()), "waste_energy": float(frame.waste_energy.sum()),
               "final_soc": float(frame.final_soc.iloc[-1]), "simultaneous_cd": int(frame.simultaneous_cd.sum()),
               "runtime_seconds": float(frame.runtime_seconds.iloc[-1])}
    suffix = f"{summary['days']}d"
    frame.to_csv(RESULTS / f"paper_q2_daily_{suffix}.csv", index=False, encoding="utf-8-sig")
    (RESULTS / f"paper_q2_summary_{suffix}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    run(31, min(365, 31 + args.days), args.fresh)
