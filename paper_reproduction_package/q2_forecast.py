"""Faithful causal forecasts from paper equations (17a)--(19)."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q2_CODE = ROOT / "03_代码" / "问题2_求解.py"
spec = importlib.util.spec_from_file_location("paper_q2_inputs", Q2_CODE)
q2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q2
spec.loader.exec_module(q2)


def day_type(ts: pd.Timestamp) -> int:
    return 0 if ts.weekday() in (4, 5) else 1  # Friday/Saturday


def forecasts(data):
    load_e = np.asarray(data.load_kw, float) / 6.0
    pv_kw = np.asarray(data.pv_kw, float)
    days, slots = load_e.shape
    dates = pd.DatetimeIndex(data.dates)
    kind = np.array([day_type(x) for x in dates], int)
    B = load_e.sum(axis=1)
    load_hat_e = np.full_like(load_e, np.nan)
    pv_hat_kw = np.full_like(pv_kw, np.nan)
    beta = np.full(days, np.nan)
    B_hat = np.full(days, np.nan)

    last_beta = None
    for d in range(1, days):
        pv_hat_kw[d] = pv_kw[max(0, d - 3):d].mean(axis=0)
        if d < 14:  # causal warmup before Jan 15
            same_weekday = np.array([i for i in range(d) if dates[i].weekday() == dates[d].weekday()])
            history = same_weekday if len(same_weekday) >= 2 else np.arange(max(0, d - 7), d)
            load_hat_e[d] = load_e[history].mean(axis=0)
            B_hat[d] = load_hat_e[d].sum()
            continue

        same_type = np.where((np.arange(days) < d) & (kind == kind[d]))[0][-3:]
        shape = load_e[same_type].sum(axis=0) / B[same_type].sum()
        shape /= shape.sum()
        assert np.isfinite(shape).all() and shape.min() >= 0

        if kind[d] == kind[d - 1]:
            B_hat[d] = B[d - 1]
        else:
            start = max(1, d - 35)
            switches = [i for i in range(start, d) if kind[i] != kind[i - 1]]
            values = [np.log(B[i] / B[i - 1]) / (kind[i] - kind[i - 1]) for i in switches]
            if values:
                last_beta = float(np.median(values))
            if last_beta is not None:
                beta[d] = last_beta
                B_hat[d] = B[d - 1] * np.exp(last_beta * (kind[d] - kind[d - 1]))
            else:
                prior = np.where((np.arange(days) < d) & (kind == kind[d]))[0]
                B_hat[d] = float(np.median(B[prior]))
        load_hat_e[d] = B_hat[d] * shape
        assert abs(load_hat_e[d].sum() - B_hat[d]) < 1e-7

    return {
        "load_kw": load_hat_e * 6.0,
        "pv_kw": pv_hat_kw,
        "B_real": B,
        "B_hat": B_hat,
        "beta": beta,
        "kind": kind,
    }


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    data = q2.read_inputs()
    f = forecasts(data)
    sl = slice(31, 365)
    actual_load = data.load_kw[sl]
    actual_pv = data.pv_kw[sl]
    load_hat = f["load_kw"][sl]
    pv_hat = f["pv_kw"][sl]
    load_err = actual_load - load_hat
    pv_err = actual_pv - pv_hat
    metrics = pd.DataFrame([{
        "period": "2025-02-01..2025-12-31",
        "load_mae_kw": float(np.abs(load_err).mean()),
        "load_mape_pct": float((np.abs(load_err) / np.maximum(actual_load, 1e-12)).mean() * 100),
        "daily_energy_mape_pct": float((np.abs(f["B_real"][sl] - f["B_hat"][sl]) / f["B_real"][sl]).mean() * 100),
        "pv_mae_kw": float(np.abs(pv_err).mean()),
        "net_mae_kw": float(np.abs(load_err - pv_err).mean()),
    }])
    metrics.to_csv(out / "paper_q2_forecast_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(HERE / "cache_q2_forecasts.npz", **f)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
