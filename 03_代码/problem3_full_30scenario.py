"""第三问正式滚动优化：24 h 主模型与 48 h 对照。

核心口径：0/6/12/18 时更新；附件3预测按绝对时间使用；调整成本
直接进入优化目标；每次只执行未来6小时；实际数据只用于回放结算。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent


def find_q2_module() -> Path:
    hits = []
    for path in CODE_DIR.glob("*.py"):
        if path == Path(__file__) or path.name.startswith("problem3_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "class CausalForecaster" in text and "def generate_scenarios" in text:
            hits.append(path)
    if len(hits) != 1:
        raise RuntimeError(f"无法唯一确定问题2权威预测模块：{hits}")
    return hits[0]


Q2_PATH = find_q2_module()
SPEC = importlib.util.spec_from_file_location("q2_authority", Q2_PATH)
q2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(q2)

N, DT = q2.N, q2.DT
ETA_C, ETA_D = q2.ETA_C, q2.ETA_D
E_MIN, E_MAX, E_INITIAL = q2.E_MIN, q2.E_MAX, q2.E_INITIAL
Q_MAX, EMERGENCY_MULTIPLIER = q2.Q_MAX, q2.EMERGENCY_MULTIPLIER
UP_MULTIPLIER, DOWN_REFUND = 1.5, 0.5
UPDATE = 36
SCENARIO_SEED = 20260911
BOOTSTRAP_SEED = 20260912
TAU = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate_attachment3() -> Path:
    data_root = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("01"))
    hits = []
    for path in data_root.rglob("*.xlsx"):
        if "result" in path.name.lower():
            continue
        try:
            shape = pd.read_excel(path, header=None, nrows=2).shape
        except Exception:
            continue
        if shape[1] == 26:
            hits.append(path)
    if len(hits) != 1:
        raise RuntimeError(f"无法唯一确定附件3：{hits}")
    return hits[0]


ATTACHMENT3 = locate_attachment3()


def read_attachment3() -> dict[tuple[pd.Timestamp, int], np.ndarray]:
    df = pd.read_excel(ATTACHMENT3, header=0)
    date_col, issue_col = df.columns[:2]
    result, current_date = {}, None
    for _, row in df.iterrows():
        if pd.notna(row[date_col]):
            current_date = pd.Timestamp(str(row[date_col]).strip()).normalize()
        if current_date is None or pd.isna(row[issue_col]):
            continue
        hh, mm = str(row[issue_col]).strip().split(":")[:2]
        issue_slot = (int(hh) * 60 + int(mm)) // 10
        values = pd.to_numeric(row.iloc[2:26], errors="raise").to_numpy(float)
        if values.size != 24 or not np.isfinite(values).all():
            raise ValueError(f"附件3异常：{current_date} {issue_slot}")
        result[(current_date, issue_slot)] = np.maximum(values, 0.0)
    return result


def interpolate_hourly_pv(hourly: np.ndarray, observed_left: float) -> np.ndarray:
    """24 hourly anchors -> 144 interval-average ten-minute values."""
    anchors = np.r_[max(observed_left, 0.0), np.maximum(hourly, 0.0)]
    out = np.empty(N)
    for j in range(N):
        x0, x1 = j / 6.0, (j + 1) / 6.0
        hour = min(int(x0), 23)
        y0 = anchors[hour] + (anchors[hour + 1] - anchors[hour]) * (x0 - hour)
        y1 = anchors[hour] + (anchors[hour + 1] - anchors[hour]) * (x1 - hour)
        out[j] = max(0.0, 0.5 * (y0 + y1))
    return out


def local_forecast(data, fc: dict, pv_book: dict, day: int, k: int, horizon: int):
    date = pd.Timestamp(data.dates[day]).normalize()
    observed_pv = 0.0 if k == 0 else float(data.pv_kw[day, k - 1])
    issued_pv = interpolate_hourly_pv(pv_book[(date, k)], observed_pv)

    # Causal midnight load forecast, corrected by observations available at k.
    load_all = np.asarray(fc["load"], float)
    load_local = np.r_[load_all[k:], load_all[N:2 * N], load_all[N:2 * N]]
    if k > 0:
        m = min(12, k)
        err = data.load_kw[day, k - m:k] - fc["load_day0"][k - m:k]
        bias = float(np.median(err))
        decay = np.exp(-np.arange(horizon) / 36.0)
        load_local[:horizon] = np.maximum(load_local[:horizon] + bias * decay, 0.0)

    # The issue-k row covers [k, k+24h). Beyond it use only the causal day+1
    # proxy aligned at absolute slot k; never read tomorrow's unpublished row.
    proxy = np.asarray(fc["pv"][N:2 * N], float)
    proxy_tail = np.r_[proxy[k:], proxy, proxy[:k]]
    pv_local = np.r_[issued_pv, proxy_tail]
    return load_local[:horizon], pv_local[:horizon]


def residual_path(residuals: dict, start: int, k: int, horizon: int, signal: int) -> np.ndarray:
    parts, d, first = [], start, True
    while sum(len(x) for x in parts) < horizon:
        arr = residuals[d][signal]
        parts.append(arr[k:] if first else arr)
        first = False
        d += 1
    return np.concatenate(parts)[:horizon]


def make_scenarios(base_load, base_pv, forecaster, day: int, k: int, count: int):
    need_days = 3
    pool = [j for j in sorted(forecaster.residuals)
            if all((j + r) in forecaster.residuals for r in range(need_days)) and j + need_days <= day]
    if not pool:
        return np.repeat(base_load[None, :], count, axis=0), np.repeat(base_pv[None, :], count, axis=0), []
    target_doy = int(forecaster.data.dates[day].dayofyear)
    distances = np.asarray([abs(int(forecaster.data.dates[j].dayofyear) - target_doy) +
                            (14 if (forecaster.data.dates[j].dayofweek >= 5) !=
                             (forecaster.data.dates[day].dayofweek >= 5) else 0) for j in pool], float)
    order = np.argsort(distances)[:min(60, len(pool))]
    pool = [pool[i] for i in order]
    prob = 1.0 / (distances[order] + 7.0); prob /= prob.sum()
    rng = np.random.default_rng(SCENARIO_SEED + 1000 * day + k)
    chosen = rng.choice(pool, size=count, replace=True, p=prob)
    load_s, pv_s = [], []
    for j in chosen:
        el = residual_path(forecaster.residuals, int(j), k, len(base_load), 0)
        ep = residual_path(forecaster.residuals, int(j), k, len(base_pv), 1)
        load_s.append(np.maximum(base_load + el, 0.0))
        pv_s.append(np.maximum(base_pv + ep, 0.0))
    return np.asarray(load_s), np.asarray(pv_s), [int(x) for x in chosen]


def solve_rolling(price: np.ndarray, load_s: np.ndarray, pv_s: np.ndarray, e_start: float,
                  commit_len: int, old_commit: np.ndarray | None, time_limit: float):
    S, H = load_s.shape
    if pv_s.shape != (S, H) or len(price) != H:
        raise ValueError("scenario dimensions do not match")
    if old_commit is not None:
        old_commit = np.maximum(np.asarray(old_commit, float), 0.0)
    # g,c,d,E,z,h_s,w_s,a,u,v
    # a_t is a common emergency/charge switch.  The former formulation used
    # one y_{s,t} per scenario; since c_t is shared by all scenarios, those
    # S switches are exactly replaceable by one a_t.  h and w cannot coexist
    # at an optimum because h has a strictly positive objective coefficient.
    g0, c0, d0, e0 = 0, H, 2 * H, 3 * H
    z0 = 4 * H + 1
    h0 = z0 + H
    w0 = h0 + S * H
    a0 = w0 + S * H
    u0 = a0 + H
    v0 = u0 + commit_len
    nv = v0 + commit_len

    objective = np.zeros(nv)
    if old_commit is None:
        objective[g0:g0 + H] = price
    else:
        objective[g0 + commit_len:g0 + H] = price[commit_len:]
        objective[u0:u0 + commit_len] = UP_MULTIPLIER * price[:commit_len]
        objective[v0:v0 + commit_len] = -DOWN_REFUND * price[:commit_len]
    objective[c0:c0 + H] = 1e-7
    objective[d0:d0 + H] = 1e-7
    for s in range(S):
        objective[h0 + s * H:h0 + (s + 1) * H] = EMERGENCY_MULTIPLIER * price / S

    lower = np.zeros(nv); upper = np.full(nv, np.inf)
    upper[g0:g0 + H] = np.max(load_s * DT, axis=0) + Q_MAX
    upper[c0:c0 + H] = Q_MAX; upper[d0:d0 + H] = Q_MAX
    lower[e0:e0 + H + 1] = E_MIN; upper[e0:e0 + H + 1] = E_MAX
    upper[z0:z0 + H] = 1.0
    mh = 1.05 * float(np.max(load_s * DT) + Q_MAX)
    mw = 1.05 * float(np.max(pv_s * DT) + Q_MAX)
    upper[h0:h0 + S * H] = mh; upper[w0:w0 + S * H] = mw
    upper[a0:a0 + H] = 1.0
    if old_commit is not None:
        upper[v0:v0 + commit_len] = np.asarray(old_commit, float)

    rows, cols, vals, lows, highs = [], [], [], [], []
    def add(items, lo=-np.inf, hi=np.inf):
        r = len(lows)
        for col, value in items:
            rows.append(r); cols.append(col); vals.append(value)
        lows.append(lo); highs.append(hi)

    for t in range(H):
        add([(e0+t+1,1),(e0+t,-1),(c0+t,-ETA_C),(d0+t,1/ETA_D)],0,0)
        add([(c0+t,1),(z0+t,-Q_MAX)],hi=0)
        add([(d0+t,1),(z0+t,Q_MAX)],hi=Q_MAX)
    add([(e0,1)], e_start, e_start)
    add([(e0+H,1)], e_start, e_start)
    if old_commit is not None:
        for t in range(commit_len):
            add([(g0+t,1),(u0+t,-1),(v0+t,1)], old_commit[t], old_commit[t])
    for s in range(S):
        net = (load_s[s] - pv_s[s]) * DT
        for t in range(H):
            add([(g0+t,1),(c0+t,-1),(d0+t,1),
                 (h0+s*H+t,1),(w0+s*H+t,-1)], net[t], net[t])
            add([(h0+s*H+t,1),(a0+t,-mh)],hi=0)
    for t in range(H):
        add([(c0+t,1),(a0+t,Q_MAX)],hi=Q_MAX)

    A = coo_matrix((vals,(rows,cols)), shape=(len(lows),nv)).tocsr()
    started = time.perf_counter()
    # Charge/discharge needs no binary because free curtailment and the small
    # positive throughput penalty remove simultaneous cycling.  The common
    # emergency/charge switch remains binary; validation rejects any violation.
    integrality=np.zeros(nv,dtype=np.uint8)
    integrality[a0:a0+H]=1
    bounds = Bounds(lower, upper)
    linear = LinearConstraint(A, np.asarray(lows), np.asarray(highs))
    res = milp(objective, integrality=integrality, bounds=bounds, constraints=linear,
               options={"time_limit":time_limit,"mip_rel_gap":1e-6,"presolve":True})
    # HiGHS can occasionally report a false presolve infeasibility for this
    # highly degenerate rolling model.  Retry the identical model without
    # presolve; this does not relax any constraint or permit a fallback.
    if (not res.success or res.x is None) and res.status == 2:
        res = milp(objective, integrality=integrality, bounds=bounds, constraints=linear,
                   options={"time_limit":time_limit,"mip_rel_gap":1e-6,"presolve":False})
    elapsed = time.perf_counter()-started
    if not res.success or res.x is None:
        raise RuntimeError(f"HiGHS失败 status={res.status}: {res.message}")
    x=res.x
    plan={"grid":x[g0:g0+H],"charge":x[c0:c0+H],"discharge":x[d0:d0+H],
          "E":x[e0:e0+H+1],"z":x[z0:z0+H],"h":x[h0:h0+S*H].reshape(S,H),
          "w":x[w0:w0+S*H].reshape(S,H),"objective":float(res.fun),
          "elapsed_s":elapsed,"mip_gap":float(getattr(res,"mip_gap",0) or 0)}
    if old_commit is not None:
        plan["up"]=x[u0:u0+commit_len]; plan["down"]=x[v0:v0+commit_len]
    else:
        plan["up"]=np.zeros(commit_len); plan["down"]=np.zeros(commit_len)
    return plan


def solve_with_fallback(price,load_s,pv_s,e_start,commit_len,old_commit,time_limit):
    """Formal 30 -> paired 10 -> Q80/Q20 deterministic fallback chain."""
    attempts=[(load_s,pv_s,"main")]
    if len(load_s)>10:
        attempts.append((load_s[:10],pv_s[:10],"fallback_10"))
    qload=np.quantile(load_s,0.8,axis=0,method="higher")[None,:]
    qpv=np.quantile(pv_s,0.2,axis=0,method="lower")[None,:]
    attempts.append((qload,qpv,"fallback_q80_q20"))
    errors=[]
    for ls,ps,mode in attempts:
        try:
            plan=solve_rolling(price,ls,ps,e_start,commit_len,old_commit,time_limit)
            plan["solve_mode"]=mode; plan["actual_scenarios"]=len(ls)
            plan["validation_load_s"]=ls; plan["validation_pv_s"]=ps
            return plan
        except RuntimeError as exc:
            errors.append(f"{mode}:{exc}")
    raise RuntimeError("全部求解模式失败 | "+" | ".join(errors))


def execute_interval(e,g,c_plan,d_plan,load_e,pv_e):
    deficit=max(0.0,load_e-g-pv_e)
    d=min(max(d_plan,0.0),Q_MAX,max(0.0,(e-E_MIN)*ETA_D),deficit)
    surplus=max(0.0,g+pv_e-load_e)
    c=min(max(c_plan,0.0),Q_MAX,max(0.0,(E_MAX-e)/ETA_C),surplus)
    h=max(0.0,load_e+c-g-pv_e-d)
    w=max(0.0,g+pv_e+d-load_e-c)
    en=e+ETA_C*c-d/ETA_D
    if E_MIN-1e-7 <= en <= E_MAX+1e-7:
        en=float(np.clip(en,E_MIN,E_MAX))
    return c,d,h,w,en


def validate_plan(plan, load_s, pv_s, e_start):
    state=plan["E"][1:]-plan["E"][:-1]-ETA_C*plan["charge"]+plan["discharge"]/ETA_D
    bal=plan["grid"][None,:]+pv_s*DT+plan["discharge"][None,:]+plan["h"]-load_s*DT-plan["charge"][None,:]-plan["w"]
    return {"max_plan_state_residual_kwh":float(np.max(np.abs(state))),
            "max_scenario_balance_residual_kwh":float(np.max(np.abs(bal))),
            "terminal_residual_kwh":float(abs(plan["E"][-1]-e_start)),
            "simultaneous_charge_discharge":int(np.sum((plan["charge"]>TAU)&(plan["discharge"]>TAU))),
            "simultaneous_emergency_charge":int(np.sum((plan["h"]>TAU)&(plan["charge"][None,:]>TAU))),
            "simultaneous_emergency_surplus":int(np.sum((plan["h"]>TAU)&(plan["w"]>TAU)))}


def prime_forecaster(forecaster, last_day: int):
    for day in range(1,last_day+1):
        fc=forecaster.forecast(day); forecaster.observe(day,fc)


def run_policy(data,pv_book,name,horizon,first_day,last_day,scenarios,time_limit,solve_mode="main"):
    ckpt_dir=ROOT/"99_临时文件"/"问题3_10场景_checkpoints"
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    ckpt=ckpt_dir/f"{name}_s{scenarios}_d{first_day}_{last_day}.pkl"
    forecaster=q2.CausalForecaster(data,use_gbdt=False)
    prime_forecaster(forecaster,first_day-1)
    e_real=float(E_INITIAL); details=[]; daily=[]; updates=[]
    start=first_day
    if ckpt.exists():
        try:
            state=pickle.loads(ckpt.read_bytes())
            if state.get("horizon")==horizon and state.get("scenarios")==scenarios and state.get("solve_mode")==solve_mode:
                start=int(state["next_day"]); e_real=float(state["e_real"])
                details=state["details"]; daily=state["daily"]; updates=state["updates"]
                # Rebuild the causal forecaster from historical observations; the
                # dynamically imported class is intentionally not pickled.
                prime_forecaster(forecaster,start-1)
                print(f"[{name}] resume from day {start}",flush=True)
        except Exception as exc:
            print(f"[{name}] checkpoint ignored: {exc}",flush=True)
    for day in range(start,last_day+1):
        fc=forecaster.forecast(day); e=e_real; effective=np.zeros(N); baseline=None
        day_rows=[]; base_cost=0.0; adj_cost=0.0; solver_total=0.0
        for k in range(0,N,UPDATE):
            load_base,pv_base=local_forecast(data,fc,pv_book,day,k,horizon)
            raw_count=30 if solve_mode=="fallback_q80_q20" else scenarios
            load_s,pv_s,sampled=make_scenarios(load_base,pv_base,forecaster,day,k,raw_count)
            if solve_mode=="fallback_q80_q20":
                load_s=np.quantile(load_s,0.8,axis=0,method="higher")[None,:]
                pv_s=np.quantile(pv_s,0.2,axis=0,method="lower")[None,:]
            prices=np.asarray([data.price[(k+t)%N] for t in range(horizon)],float)
            commit_len=N-k
            old=None if k==0 else effective[k:].copy()
            try:
                plan=solve_rolling(prices,load_s,pv_s,e,commit_len,old,time_limit)
            except RuntimeError as exc:
                raise RuntimeError(f"policy={name},date={data.dates[day].date()},issue_slot={k}: {exc}") from exc
            plan["solve_mode"]=solve_mode; plan["actual_scenarios"]=len(load_s)
            checks=validate_plan(plan,load_s,pv_s,e)
            solver_total+=plan["elapsed_s"]
            if k==0:
                baseline=np.maximum(plan["grid"][:N],0.0); effective[:]=baseline
                update_cost=float(np.sum(data.price*baseline)); base_cost=update_cost
            else:
                effective[k:]=np.maximum(plan["grid"][:commit_len],0.0)
                up_cost=float(np.sum(data.price[k:]*UP_MULTIPLIER*plan["up"]))
                down_refund=float(np.sum(data.price[k:]*DOWN_REFUND*plan["down"]))
                update_cost=up_cost-down_refund
                adj_cost+=update_cost
            if k==0:
                up_cost=down_refund=0.0
            adj_residual=(0.0 if old is None else float(np.max(np.abs(
                plan["grid"][:commit_len]-old-plan["up"]+plan["down"]))))
            updates.append({"date":data.dates[day],"policy":name,"issue_slot":k,
                            "issue_time":f"{k//6:02d}:00","training_cutoff":data.dates[day-1],
                            "forecast_horizon_h":horizon/6,"executed_h":6,
                            "commitment_update_cost_yuan":update_cost,"up_cost_yuan":up_cost,
                            "down_refund_yuan":down_refund,"adjustment_identity_residual_kwh":adj_residual,
                            "up_kwh":float(plan["up"].sum()),
                            "down_kwh":float(plan["down"].sum()),"solver_objective":plan["objective"],
                            "solver_elapsed_s":plan["elapsed_s"],"solver_mip_gap":plan["mip_gap"],
                            "requested_scenario_count":scenarios,"actual_scenario_count":plan["actual_scenarios"],
                            "solve_mode":plan["solve_mode"],
                            "sampled_days":";".join(map(str,sampled)),**checks})
            for j in range(min(UPDATE,N-k)):
                t=k+j; g=float(effective[t])
                c,d,h,w,en=execute_interval(e,g,float(plan["charge"][j]),float(plan["discharge"][j]),
                                            data.load_kw[day,t]*DT,data.pv_kw[day,t]*DT)
                day_rows.append({"date":data.dates[day],"policy":name,"t":t,"issue_slot":k,
                                 "price_yuan_per_kwh":data.price[t],"load_kw":data.load_kw[day,t],
                                 "pv_kw":data.pv_kw[day,t],"committed_grid_kwh":g,
                                 "baseline_grid_kwh":float(baseline[t]),
                                 "adjustment_grid_kwh":float(g-baseline[t]),
                                 "charge_exec_kwh":c,"discharge_exec_kwh":d,"emergency_kwh":h,
                                 "surplus_kwh":w,"E_start_kwh":e,"E_end_kwh":en,
                                 "emergency_cost_yuan":EMERGENCY_MULTIPLIER*data.price[t]*h})
                e=en
        frame=pd.DataFrame(day_rows); details.append(frame); e_real=e
        emergency_cost=float(frame.emergency_cost_yuan.sum())
        daily.append({"date":data.dates[day],"policy":name,"baseline_cost_yuan":base_cost,
                      "adjustment_cost_yuan":adj_cost,"emergency_cost_yuan":emergency_cost,
                      "total_cost_yuan":base_cost+adj_cost+emergency_cost,
                      "emergency_kwh":float(frame.emergency_kwh.sum()),
                      "emergency_intervals":int((frame.emergency_kwh>TAU).sum()),
                      "end_storage_kwh":e,"solver_time_s":solver_total})
        print(f"[{name}] {data.dates[day].date()} complete, scenarios={scenarios}, "
              f"cost={base_cost+adj_cost+emergency_cost:.2f}, solver={solver_total:.2f}s", flush=True)
        forecaster.observe(day,fc)
        pickle.dump({"horizon":horizon,"scenarios":scenarios,"solve_mode":solve_mode,
                     "next_day":day+1,"e_real":e_real,"details":details,"daily":daily,
                     "updates":updates},ckpt.open("wb"))
    return pd.concat(details,ignore_index=True),pd.DataFrame(daily),pd.DataFrame(updates)


def bootstrap_ci(x,reps=5000,block=7):
    """Circular moving-block bootstrap for serially correlated daily costs."""
    x=np.asarray(x,float); rng=np.random.default_rng(BOOTSTRAP_SEED); n=len(x)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    offsets=np.arange(block)
    indices=(starts[:,:,None]+offsets[None,None,:])%n
    means=np.mean(x[indices.reshape(reps,-1)[:,:n]],axis=1)
    return [float(np.quantile(means,0.025)),float(np.quantile(means,0.975))]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start-day",type=int,default=31)
    ap.add_argument("--end-day",type=int,default=364)
    ap.add_argument("--scenarios",type=int,default=30)
    ap.add_argument("--solver-time-limit",type=float,default=30.0)
    ap.add_argument("--require-main",action="store_true",
                    help="禁止10场景/Q80回退；任一更新点未完成即终止")
    ap.add_argument("--output-dir",default=str(ROOT/"04_结果"/"问题3_正式求解"))
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    data=q2.read_inputs(); pv_book=read_attachment3(); started=time.perf_counter()
    attempts=[(args.scenarios,"main")]
    if not args.require_main:
        if args.scenarios>10: attempts.append((10,"fallback_10"))
        attempts.append((1,"fallback_q80_q20"))
    errors=[]
    for scenario_count,mode in attempts:
        ds=[]; ms=[]; us=[]
        try:
            # The pair is an atomic experiment: failure of either policy
            # discards both results and reruns both at the same fallback level.
            for name,H in (("H24",N),("H48",2*N)):
                d,m,u=run_policy(data,pv_book,name,H,args.start_day,args.end_day,
                                 scenario_count,args.solver_time_limit,mode)
                ds.append(d); ms.append(m); us.append(u)
            break
        except RuntimeError as exc:
            errors.append(f"{mode}:{exc}")
    else:
        raise RuntimeError("H24/H48同步回退全部失败 | "+" | ".join(errors))
    detail=pd.concat(ds,ignore_index=True); daily=pd.concat(ms,ignore_index=True); updates=pd.concat(us,ignore_index=True)
    pivot=daily.pivot(index="date",columns="policy",values="total_cost_yuan").reset_index()
    pivot["H24_minus_H48_yuan"]=pivot.H24-pivot.H48
    summary=daily.groupby("policy").agg(days=("date","count"),total_cost_yuan=("total_cost_yuan","sum"),
        baseline_cost_yuan=("baseline_cost_yuan","sum"),adjustment_cost_yuan=("adjustment_cost_yuan","sum"),
        emergency_cost_yuan=("emergency_cost_yuan","sum"),emergency_kwh=("emergency_kwh","sum"),
        emergency_intervals=("emergency_intervals","sum"),mean_end_storage_kwh=("end_storage_kwh","mean"),
        min_end_storage_kwh=("end_storage_kwh","min"),solver_time_s=("solver_time_s","sum")).reset_index()
    delta=float(pivot["H24_minus_H48_yuan"].sum()); ci=bootstrap_ci(pivot["H24_minus_H48_yuan"])
    specified=pd.to_datetime(["2025-03-20","2025-06-21","2025-09-23","2025-12-21"])
    specified_table=daily[daily.date.isin(specified)].copy()
    actual_balance=(detail.committed_grid_kwh+detail.pv_kw*DT+detail.discharge_exec_kwh+
                    detail.emergency_kwh-detail.load_kw*DT-detail.charge_exec_kwh-detail.surplus_kwh)
    actual_state=(detail.E_end_kwh-detail.E_start_kwh-ETA_C*detail.charge_exec_kwh+
                  detail.discharge_exec_kwh/ETA_D)
    validation={"rows_detail":len(detail),"rows_daily":len(daily),"rows_updates":len(updates),
      "expected_detail":2*(args.end_day-args.start_day+1)*N,
      "max_actual_balance_residual_kwh":float(np.max(np.abs(actual_balance))),
      "max_actual_state_residual_kwh":float(np.max(np.abs(actual_state))),
      "max_update_plan_state_residual_kwh":float(updates.max_plan_state_residual_kwh.max()),
      "max_update_scenario_balance_residual_kwh":float(updates.max_scenario_balance_residual_kwh.max()),
      "max_update_terminal_residual_kwh":float(updates.terminal_residual_kwh.max()),
      "max_solver_mip_gap":float(updates.solver_mip_gap.max()),
      "plan_simultaneous_charge_discharge":int(updates.simultaneous_charge_discharge.sum()),
      "plan_simultaneous_emergency_charge":int(updates.simultaneous_emergency_charge.sum()),
      "plan_simultaneous_emergency_surplus":int(updates.simultaneous_emergency_surplus.sum()),
      "paired_solve_mode_equal":bool((updates.pivot(index=["date","issue_slot"],columns="policy",values="solve_mode").nunique(axis=1)==1).all()),
      "paired_scenario_count_equal":bool((updates.pivot(index=["date","issue_slot"],columns="policy",values="actual_scenario_count").nunique(axis=1)==1).all()),
      "max_adjustment_identity_residual_kwh":float(updates.adjustment_identity_residual_kwh.max()),
      "max_charge_exec_kwh":float(detail.charge_exec_kwh.max()),
      "max_discharge_exec_kwh":float(detail.discharge_exec_kwh.max()),
      "actual_simultaneous_emergency_charge":int(np.sum((detail.emergency_kwh>TAU)&(detail.charge_exec_kwh>TAU))),
      "storage_min_kwh":float(detail.E_start_kwh.min()),"storage_max_kwh":float(detail.E_end_kwh.max()),
      "h24_minus_h48_total_yuan":delta,"paired_mean_ci95_yuan_per_day":ci,
      "fallback_events":errors,
      "passed":False}
    validation["passed"]=(validation["rows_detail"]==validation["expected_detail"] and
      validation["max_update_plan_state_residual_kwh"]<1e-6 and
      validation["max_actual_balance_residual_kwh"]<1e-6 and
      validation["max_actual_state_residual_kwh"]<1e-6 and
      validation["max_update_scenario_balance_residual_kwh"]<1e-6 and
      validation["max_update_terminal_residual_kwh"]<1e-6 and
      validation["max_solver_mip_gap"]<=1e-6+1e-12 and
      validation["plan_simultaneous_charge_discharge"]==0 and
      validation["plan_simultaneous_emergency_charge"]==0 and
      validation["plan_simultaneous_emergency_surplus"]==0 and
      validation["paired_solve_mode_equal"] and validation["paired_scenario_count_equal"] and
      validation["max_adjustment_identity_residual_kwh"]<1e-6 and
      validation["max_charge_exec_kwh"]<=Q_MAX+1e-6 and validation["max_discharge_exec_kwh"]<=Q_MAX+1e-6 and
      validation["actual_simultaneous_emergency_charge"]==0 and
      validation["storage_min_kwh"]>=E_MIN-1e-6 and validation["storage_max_kwh"]<=E_MAX+1e-6)
    detail.to_csv(out/"问题3_逐时段结果.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(out/"问题3_每日指标.csv",index=False,encoding="utf-8-sig")
    updates.to_csv(out/"问题3_更新级验证.csv",index=False,encoding="utf-8-sig")
    pivot.to_csv(out/"问题3_每日配对差.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(out/"问题3_两方案汇总.csv",index=False,encoding="utf-8-sig")
    specified_table.to_csv(out/"问题3_指定日期结果.csv",index=False,encoding="utf-8-sig")
    (out/"问题3_总体验证.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    elapsed=time.perf_counter()-started
    outputs=[p for p in out.iterdir() if p.is_file() and p.name!="问题3_复现清单.json"]
    manifest={"command":f"python 03_代码/problem3_full.py --start-day {args.start_day} --end-day {args.end_day} --scenarios {args.scenarios} --solver-time-limit {args.solver_time_limit} --output-dir {args.output_dir}",
      "scenario_seed":SCENARIO_SEED,"bootstrap_seed":BOOTSTRAP_SEED,"bootstrap":{"type":"circular_moving_block","block_days":7,"repetitions":5000},
      "parameters":{"horizons_h":[24,48],"execute_h":6,"scenarios":args.scenarios,
      "eta_charge":ETA_C,"eta_discharge":ETA_D,"storage_bounds_kwh":[E_MIN,E_MAX]},
      "inputs":{str(q2.INPUT1):sha256(q2.INPUT1),str(q2.INPUT2):sha256(q2.INPUT2),str(ATTACHMENT3):sha256(ATTACHMENT3),str(Q2_PATH):sha256(Q2_PATH)},
      "script_sha256":sha256(Path(__file__)),"runtime_s":elapsed,"python":platform.python_version(),
      "fallback_events":errors,
      "outputs":{p.name:sha256(p) for p in outputs}}
    (out/"问题3_复现清单.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(validation,ensure_ascii=False,indent=2))
    if not validation["passed"]:
        raise SystemExit("validation failed")


if __name__=="__main__":
    main()
