"""Causal 24h look-ahead / 6h commitment experiments. No original outputs written."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import sys
from pathlib import Path
# Reuse the installed read-only Excel dependency without installing anything.
try:
    import openpyxl
except ImportError:
    bundled = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages'
    sys.path.append(str(bundled))
import argparse, hashlib, importlib.util, json, time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'results'
CACHE = HERE/'cache'
N, Q, EMIN, EMAX, ETA = 144, 5000/6, 1200., 10800., .9
UPDATES = {'K0': (0,), 'K06': (0,36), 'K0612': (0,36,72), 'Kall': (0,36,72,108)}
CODE = ROOT/'03_代码/问题2_求解.py'
spec = importlib.util.spec_from_file_location('q2_reused', CODE)
q2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q2
spec.loader.exec_module(q2)

def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inputs():
    data = q2.read_inputs()
    raw = pd.read_excel(ROOT/'01_题目与数据/原始附件/附件3.xlsx')
    assert raw.shape == (1460,26)
    # Dates are vertically merged across four issue rows.
    raw.iloc[:,0] = raw.iloc[:,0].ffill()
    official = np.full((365,4,24), np.nan)
    seen = set()
    for row in raw.itertuples(index=False, name=None):
        day = (pd.Timestamp(row[0])-data.dates[0]).days
        hour = int(str(row[1]).split(':')[0])
        assert hour in (0,6,12,18) and (day,hour) not in seen
        seen.add((day,hour))
        official[day,hour//6] = row[2:]
    assert np.isfinite(official).all() and official.min() >= 0
    return data, official

def pv_interpolate(anchor, future):
    knots = np.r_[anchor, future]
    edges = np.arange(145)/6
    values = np.interp(edges, np.arange(25), knots)
    linear = (values[:-1]+values[1:])/2
    f = PchipInterpolator(np.arange(25), knots)
    primitive = f.antiderivative()
    pchip = np.diff(primitive(edges))*6
    assert np.max(np.abs(values[::6]-knots)) < 1e-8
    assert np.max(np.abs(f(np.arange(25))-knots)) < 1e-8
    return np.maximum(linear,0), np.maximum(pchip,0)

def prepare():
    OUT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    data, official = inputs()
    fingerprint = {str(p.relative_to(ROOT)):sha(p) for p in [CODE, q2.INPUT1, q2.INPUT2, ROOT/'01_题目与数据/原始附件/附件3.xlsx']}
    cachefile = CACHE/'forecasts_v1.npz'
    if cachefile.exists():
        assert json.loads((CACHE/'input_hashes.json').read_text('utf-8')) == fingerprint
        return data, dict(np.load(cachefile))
    started = time.perf_counter()
    f = q2.CausalForecaster(data)
    loads = np.full((365,4,N),np.nan)
    corrected = loads.copy(); linear = loads.copy(); pchip = loads.copy()
    provenance, weights = [], []
    for day in range(1,365):
        # Unmodified Q2 load ensemble, including recursive second-day GBDT features.
        p0,p1,candidates,ww = f._forecast_signal('load',day)
        whole = np.r_[p0,p1]
        for ix,k in enumerate((0,36,72,108)):
            issue = data.dates[day]+pd.Timedelta(minutes=10*k)
            cutoff = data.dates[day]
            target = issue+pd.to_timedelta(np.arange(1,145)*10,unit='min')
            assert cutoff <= issue and (issue < target).all()
            loads[day,ix] = whole[k:k+144]
            bias = np.median(data.load_kw[day,k-12:k]-p0[k-12:k]) if k else 0.
            corrected[day,ix] = np.maximum(0,loads[day,ix]+np.exp(-np.arange(144)/36)*bias)
            anchor = data.pv_kw[day,k-1] if k else data.pv_kw[day-1,-1]
            linear[day,ix],pchip[day,ix] = pv_interpolate(anchor,official[day,ix])
            for t,tt in enumerate(target):
                provenance.append((issue,cutoff,tt,issue,day,k,t))
        weights.append({'date':str(data.dates[day].date()),**ww})
        # Reveal the day's targets only AFTER generating that day's forecasts.
        for name in q2.MODEL_NAMES:
            f.base_mae['load'][name].append(float(np.mean(np.abs(data.load_kw[day]-candidates[name]))) if name in candidates else np.nan)
        if day%20 == 0:
            print(f'forecast {day}/364 elapsed={time.perf_counter()-started:.1f}s',flush=True)
    residuals = {}
    net = (data.load_kw-data.pv_kw).ravel()
    for interp,pv in [('linear',linear),('pchip',pchip)]:
        for corr,ll in [(False,loads),(True,corrected)]:
            res = np.full_like(loads,np.nan)
            for day in range(1,365):
                for ix,k in enumerate((0,36,72,108)):
                    start=day*144+k
                    if start+144 <= len(net):
                        res[day,ix] = net[start:start+144]-(ll[day,ix]-pv[day,ix])
            residuals[f'residual_{interp}_{int(corr)}'] = res
    chosen = np.full((365,4,10),-1,dtype=int)
    for day in range(31,365):
        for ix,k in enumerate((0,36,72,108)):
            # A path ending at the current issue is now completely observable.
            pool=[j for j in range(1,day) if j*144+k+144 <= day*144+k][-60:]
            typ=data.dates[day].dayofweek>=5
            matched=[j for j in reversed(pool) if (data.dates[j].dayofweek>=5)==typ]
            other=[j for j in reversed(pool) if (data.dates[j].dayofweek>=5)!=typ]
            chosen[day,ix]=(matched+other)[:10]
            assert (chosen[day,ix]<day).all()
            assert (chosen[day,ix]*144+k+144 <= day*144+k).all()
    arrays=dict(load=loads,load_corrected=corrected,pv_linear=linear,pv_pchip=pchip,chosen=chosen,**residuals)
    np.savez_compressed(cachefile,**arrays)
    dump(fingerprint,CACHE/'input_hashes.json')
    dump({'elapsed_s':time.perf_counter()-started,'predictor':'Unmodified Q2 load ensemble; 00:00 48h base sliced at each issue; no future actual lag1'},CACHE/'forecast_config.json')
    csv(pd.DataFrame(provenance,columns=['issue_time','training_cutoff','target_time','feature_cutoff','day','k','lead_slot']),OUT/'forecast_provenance.csv')
    csv(pd.DataFrame(weights),OUT/'load_weights.csv')
    csv(pd.DataFrame([(d,k*6,s,int(j)) for d in range(31,365) for k in range(4) for s,j in enumerate(chosen[d,k])],columns=['day','issue_hour','scenario','historical_day']),OUT/'scenario_sources.csv')
    return data,arrays

def solve(price, scenarios, soc, k, prev, variant, mode='physical'):
    """Shared g,c,d,E; scenario emergency epigraph. Waste recovered from balance.

    Always solve LP first. After observed violations, one charge-mode bit per time enforces both
    physical restrictions in a single exact MILP.
    """
    start=time.perf_counter(); S,H=scenarios.shape
    assert H==144
    net=scenarios/6
    p=price[(np.arange(H)+k)%144]
    # [g,c,d,E(145),h(S*H),u,v,z,y]. z/y initially continuous and fixed zero.
    G,C,D,E,B,U,V,Z,Y=0,H,2*H,3*H,4*H+1,(4+S)*H+1,(5+S)*H+1,(6+S)*H+1,(7+S)*H+1
    nv=(8+S)*H+1
    obj=np.zeros(nv); obj[B:B+S*H]=np.tile(5*p/S,S)
    lo=np.zeros(nv); hi=np.full(nv,np.inf)
    hi[C:C+H]=Q;hi[D:D+H]=Q;lo[E:E+H+1]=EMIN;hi[E:E+H+1]=EMAX
    lo[E]=hi[E]=soc
    hi[U:V+H]=0;hi[Z:Y+H]=0
    rows=[];cols=[];vals=[];lower=[];upper=[]
    def add(entries,l=-np.inf,u=np.inf):
        r=len(lower)
        for col,val in entries:rows.append(r);cols.append(col);vals.append(val)
        lower.append(l);upper.append(u)
    remaining=144-k
    adjustable=remaining if variant=='ALL' else min(36,remaining)
    if k==0:obj[G:G+H]=p
    else:
        hi[U:U+adjustable]=np.inf;hi[V:V+adjustable]=prev[k:k+adjustable]
        obj[U:U+adjustable]=1.5*p[:adjustable];obj[V:V+adjustable]=-.5*p[:adjustable]
        for t in range(adjustable):add([(G+t,1),(U+t,-1),(V+t,1)],prev[k+t],prev[k+t])
        lo[G+adjustable:G+remaining]=prev[k+adjustable:]
        hi[G+adjustable:G+remaining]=prev[k+adjustable:]
        obj[G+remaining:G+H]=p[remaining:]
    for t in range(H):
        add([(E+t+1,1),(E+t,-1),(C+t,-ETA),(D+t,1/ETA)],0,0)
    for s in range(S):
        for t in range(H):
            add([(G+t,1),(D+t,1),(B+s*H+t,1),(C+t,-1)],net[s,t])
    integ=np.zeros(nv,dtype=int); cdset=set(); hcset=set(); iterations=0
    first_cd=first_hc=0; tightened=False
    while True:
        matrix=coo_matrix((vals,(rows,cols)),shape=(len(lower),nv)).tocsc()
        ll=np.array(lower);uu=np.array(upper)
        if not integ.any():
            eq=np.isfinite(ll)&np.isfinite(uu)&(ll==uu)
            ub=~eq
            # Initial nonequalities are all lower bounds (emergency epigraph).
            result=linprog(obj,A_ub=-matrix[ub],b_ub=-ll[ub],A_eq=matrix[eq],b_eq=ll[eq],bounds=np.c_[lo,hi],method='highs')
        else:
            result=milp(obj,integrality=integ,bounds=Bounds(lo,hi),constraints=LinearConstraint(matrix,ll,uu),options={'time_limit':120.,'mip_rel_gap':1e-8,**({'mip_feasibility_tolerance':1e-10} if tightened else {})})
        if not result.success:raise RuntimeError(f'solver failed: {result.message}')
        x=result.x
        g=x[G:G+H];c=x[C:C+H];d=x[D:D+H]
        h=np.maximum(net+c-g-d,0);w=np.maximum(g+d-net-c,0)
        cds=set(np.flatnonzero((c>1e-6)&(d>1e-6)).tolist())
        hcs=set(np.flatnonzero((c>1e-6)&np.any(h>1e-6,axis=0)).tolist())
        if iterations==0:first_cd=len(cds);first_hc=int(np.sum((c[None,:]>1e-6)&(h>1e-6)))
        if mode=='lp' or not(cds or hcs):break
        newcd=cds-cdset;newhc=hcs-hcset
        if not(newcd or newhc):
            dump(dict(k=k,soc=soc,cd_times=sorted(cds),hc_times=sorted(hcs),
                 values=[dict(t=t,c=float(c[t]),d=float(d[t]),max_h=float(h[:,t].max()),z=float(x[Z+t]),g=float(g[t]),netmax=float(net[:,t].max())) for t in sorted(cds|hcs)]),
                 OUT/'solver_tolerance_diagnostic.json')
            if tightened:
                raise RuntimeError('Exclusivity violates numerical tolerance after strict solve')
            tightened=True
            continue
        # One shared charge-mode bit per time (144, not S*144). This enforces
        # both observed physical issues in one exact MILP, avoiding repeated
        # lazy-cut solves as the optimizer moves charge to another time.
        for t in range(H):
            integ[Z+t]=1;hi[Z+t]=1
            add([(C+t,1),(Z+t,-Q)],u=0)
            add([(D+t,1),(Z+t,Q)],u=Q)
            # If charging (z=1), d=0 and grid-c must cover the largest
            # scenario net load. If z=0, c=0 and this reduces to grid>=0.
            # Positive emergency prices make the epigraph h=0 when covered.
            # Exact equivalent, with one row instead of S big-M rows.
            add([(G+t,1),(C+t,-1),(Z+t,-float(net[:,t].max()))],l=0)
        cdset=set(range(H));hcset=set(range(H));iterations+=1
    assert np.max(np.abs(g+d+h-net-c-w))<1e-6
    e=x[E:E+H+1]
    assert np.max(np.abs(np.diff(e)-ETA*c+d/ETA))<1e-6
    return dict(g=np.maximum(g,0),c=np.maximum(c,0),d=np.maximum(d,0),E=e,h=h,
                objective=float(result.fun),elapsed_s=time.perf_counter()-start,
                lp_simultaneous_cd=first_cd,lp_emergency_charge=first_hc,
                final_simultaneous_cd=len(cds),final_emergency_charge=len(hcs),
                binaries=int(integ.sum()),iterations=iterations,tolerance_retry=int(tightened),status=int(result.status),
                terminal_soc=float(e[-1]))

def simulate(data,a,variant='ALL',policy='Kall',days=334,interp='linear',correction=False,mode='physical',label=None):
    label=label or f'{variant}_{policy}_{interp}_bias{int(correction)}_{mode}_v3_{days}d'
    directory=OUT/label
    if (directory/'summary.json').exists():
        return json.loads((directory/'summary.json').read_text('utf-8'))
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    started=time.perf_counter();soc=6000.;details=[];daily=[];solvers=[];adjustments=[]
    l=a['load_corrected' if correction else 'load'];pv=a[f'pv_{interp}'];res=a[f'residual_{interp}_{int(correction)}']
    for day in range(31,31+days):
        date=str(data.dates[day].date())
        saved=checkpoint/f'{date}.json'
        if saved.exists():
            saved_day=json.loads(saved.read_text('utf-8'))
            assert abs(saved_day['daily']['soc_0000']-soc)<1e-6
            daily.append(saved_day['daily']);details.extend(saved_day['details'])
            solvers.extend(saved_day['solvers']);adjustments.extend(saved_day['adjustments'])
            soc=saved_day['daily']['soc_2400']
            continue
        solver_start=len(solvers);adjustment_start=len(adjustments)
        prev=np.zeros(N);g0=None
        cp=np.zeros(N);dp=np.zeros(N);up=np.zeros(N);down=np.zeros(N)
        terminal={k:None for k in (0,36,72,108)}
        estart=soc;etrace=[soc]
        for k in (0,36,72,108):
            if k in UPDATES[policy]:
                ix=k//36;js=a['chosen'][day,ix]
                assert np.all(js<day) and np.isfinite(res[js,ix]).all()
                ns=l[day,ix]-pv[day,ix]+res[js,ix]
                old=prev.copy()
                plan=solve(data.price,ns,soc,k,prev,variant,mode)
                terminal[k]=plan['terminal_soc']
                solvers.append({'date':date,'issue_hour':k//6,**{key:value for key,value in plan.items() if np.isscalar(value)}})
                if k==0:
                    prev=plan['g'].copy();g0=prev.copy()
                else:
                    stop=N if variant=='ALL' else k+36
                    prev[k:stop]=plan['g'][:stop-k]
                    assert np.array_equal(old[:k],prev[:k])
                    if variant=='BLOCK':assert np.array_equal(old[k+36:],prev[k+36:])
                    u=np.maximum(prev-old,0);v=np.maximum(old-prev,0)
                    up+=u;down+=v
                    for t in range(k,stop):
                        adjustments.append(dict(date=date,issue_hour=k//6,t=t,previous=float(old[t]),new=float(prev[t]),up=float(u[t]),down=float(v[t]),price=float(data.price[t])))
                # If an update is disabled, keep the latest frozen plan for that
                # block, without consulting its new forecast or real future.
                cp[k:]=plan['c'][:144-k];dp[k:]=plan['d'][:144-k]
            for t in range(k,k+36):
                oldsoc=soc
                c,d,h,w,soc=q2.execute_interval(soc,prev[t],cp[t],dp[t],data.load_kw[day,t]/6,data.pv_kw[day,t]/6)
                assert EMIN-1e-6<=soc<=EMAX+1e-6 and c<=Q+1e-6 and d<=Q+1e-6
                net=(data.load_kw[day,t]-data.pv_kw[day,t])/6
                assert abs(prev[t]+d+h-net-c-w)<1e-6
                assert not(c>1e-6 and d>1e-6) and not(c>1e-6 and h>1e-6)
                details.append(dict(date=date,t=t,price=float(data.price[t]),load_kw=float(data.load_kw[day,t]),pv_kw=float(data.pv_kw[day,t]),g0=float(g0[t]),g_final=float(prev[t]),up=float(up[t]),down=float(down[t]),c_plan=float(cp[t]),d_plan=float(dp[t]),c_exec=c,d_exec=d,h_real=h,w_real=w,soc_start=oldsoc,soc_end=soc))
                etrace.append(soc)
        z=pd.DataFrame(details[-144:]);p=data.price
        amounts=[float(p@g0),float(1.5*p@up),float(-.5*p@down),float(5*p@z.h_real)]
        row=dict(date=date,cost_plan=amounts[0],cost_adjust_up=amounts[1],cost_adjust_down=amounts[2],cost_emergency=amounts[3],cost_total=sum(amounts),energy_g0=float(g0.sum()),energy_final_commitment=float(prev.sum()),energy_up=float(up.sum()),energy_down=float(down.sum()),adjustment_churn=float((up+down).sum()-np.abs(prev-g0).sum()),energy_emergency=float(z.h_real.sum()),energy_waste=float(z.w_real.sum()),charge_energy=float(z.c_exec.sum()),discharge_energy=float(z.d_exec.sum()))
        row.update({f'soc_{h:02d}00':float(etrace[h*6]) for h in (0,6,12,18,24)})
        row.update({f'terminal_proxy_soc_{k//6:02d}00':terminal[k] for k in terminal})
        if daily:assert abs(daily[-1]['soc_2400']-estart)<1e-6
        if variant=='BLOCK':assert abs(row['adjustment_churn'])<1e-6
        daily.append(row)
        dump(dict(daily=row,details=details[-144:],solvers=solvers[solver_start:],adjustments=adjustments[adjustment_start:]),saved)
        if (day-30)%14==0 or day==30+days:
            print(f'{label} {day-30}/{days} cost={sum(r["cost_total"] for r in daily):.2f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    detail=pd.DataFrame(details);dd=pd.DataFrame(daily);ss=pd.DataFrame(solvers)
    adj=pd.DataFrame(adjustments,columns=['date','issue_hour','t','previous','new','up','down','price'])
    summary={name:float(dd[name].sum()) for name in ['cost_plan','cost_adjust_up','cost_adjust_down','cost_emergency','cost_total','energy_up','energy_down','energy_emergency','energy_waste','adjustment_churn']}
    summary.update(label=label,variant=variant,policy=policy,days=days,interpolation=interp,correction=correction,mode=mode,final_soc=soc,elapsed_s=float(ss.elapsed_s.sum()),session_wall_s=time.perf_counter()-started,solver_s=float(ss.elapsed_s.sum()),tolerance_retries=int(ss.get('tolerance_retry',pd.Series(dtype=float)).sum()),lp_simultaneous_cd=int(ss.lp_simultaneous_cd.sum()),lp_emergency_charge=int(ss.lp_emergency_charge.sum()),binaries=int(ss.binaries.sum()),directory=str(directory))
    csv(detail,directory/'detail.csv');csv(dd,directory/'daily.csv');csv(ss,directory/'solver.csv');csv(adj,directory/'adjustments.csv')
    # Reconciliation 1: interval ledger; 2: independently reread order changes.
    direct=float((detail.price*(detail.g0+1.5*detail.up-.5*detail.down+5*detail.h_real)).sum())
    orders=pd.read_csv(directory/'adjustments.csv')
    independent=float((detail.price*detail.g0).sum()+(orders.price*(1.5*(orders.new-orders.previous).clip(lower=0)-.5*(orders.previous-orders.new).clip(lower=0))).sum()+(detail.price*5*detail.h_real).sum())
    assert abs(direct-summary['cost_total'])<1e-6 and abs(independent-summary['cost_total'])<1e-6
    dump(dict(interval_recompute=direct,order_recompute=independent,reported=summary['cost_total'],balance_max=float(np.max(np.abs(detail.g_final+detail.d_exec+detail.h_real-(detail.load_kw-detail.pv_kw)/6-detail.c_exec-detail.w_real))),soc_min=float(min(detail.soc_start.min(),detail.soc_end.min())),soc_max=float(max(detail.soc_start.max(),detail.soc_end.max()))),directory/'verification.json')
    dump(summary,directory/'summary.json')
    return summary

def baseline():
    result=dict(method='original_q3_baseline',status='unavailable_missing_code',commit='0340916',cost_total=None,cost_plan=None,cost_adjust_up=None,cost_adjust_down=None,cost_emergency=None,energy_emergency=None,final_soc=None,elapsed_s=None,reason='No Q3 solver or completed result in any available branch. Report is not executable evidence.')
    csv(pd.DataFrame([result]),OUT/'baseline_summary.csv')
    print(result['reason'],flush=True)

def forecast_metrics(data,a):
    rows=[]
    actual={'load':data.load_kw.ravel(),'pv':data.pv_kw.ravel(),'net':(data.load_kw-data.pv_kw).ravel()}
    # Four same-day issues share only 18:10--24:00. Also report 0/6/12
    # common daylight targets and all complete in-year 24h target observations.
    scopes=[('paired_all4',range(31,365),(0,1,2,3),108),('paired_0_6_12',range(31,365),(0,1,2),72),('full_available',range(31,365),(0,1,2,3),None),('january_calibration',range(1,31),(0,1,2,3),None)]
    for interp in ('linear','pchip'):
        for scope,days,issues,start_common in scopes:
            for ix in issues:
                for signal in actual:
                    errors=[]
                    for day in days:
                        k=ix*36
                        local=np.arange(144) if start_common is None else np.arange(start_common-k,144-k)
                        targets=day*144+k+local
                        valid=targets<len(actual[signal]);local=local[valid];targets=targets[valid]
                        predicted=a['load'][day,ix] if signal=='load' else a[f'pv_{interp}'][day,ix] if signal=='pv' else a['load'][day,ix]-a[f'pv_{interp}'][day,ix]
                        errors.extend(actual[signal][targets]-predicted[local])
                    err=np.asarray(errors)
                    rows.append(dict(scope=scope,interpolation=interp,issue_hour=ix*6,signal=signal,count=len(err),mae_kw=float(np.abs(err).mean()),rmse_kw=float(np.sqrt(np.mean(err**2)))))
    df=pd.DataFrame(rows);csv(df,OUT/'forecast_metrics.csv')
    jan=df[(df.scope=='january_calibration')&(df.signal=='pv')]
    scores=jan.groupby('interpolation').mae_kw.mean()
    gain=float(1-scores['pchip']/scores['linear'])
    dump(dict(selection_period='2025-01-02 to 2025-01-31',linear_mae=float(scores['linear']),pchip_mae=float(scores['pchip']),relative_mae_improvement=gain,threshold=.02,run_annual_pchip=gain>=.02),OUT/'interpolation_decision.json')
    return gain

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['prepare','baseline','all24','block24','updates','interpolation','final'],required=True)
    p.add_argument('--policy',choices=list(UPDATES),default='Kall');p.add_argument('--days',type=int,default=334);p.add_argument('--mode',choices=['lp','physical'],default='physical')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.experiment=='baseline':baseline();return
    if args.experiment in ('all24','block24','updates','final') and args.days==334:
        for vv in ('ALL','BLOCK'):
            for nd in (1,7,28):
                assert (OUT/f'{vv}_Kall_linear_bias0_physical_v3_{nd}d'/'verification.json').exists(), 'Complete pilot stages before annual execution.'
    data,a=prepare()
    if args.experiment=='prepare':forecast_metrics(data,a);return
    if args.experiment in ('all24','block24'):
        print(json.dumps(simulate(data,a,'ALL' if args.experiment=='all24' else 'BLOCK',policy=args.policy,days=args.days,mode=args.mode),ensure_ascii=False));return
    if args.experiment=='interpolation':print(forecast_metrics(data,a));return
    allresult=simulate(data,a,'ALL');blockresult=simulate(data,a,'BLOCK')
    csv(pd.DataFrame([allresult,blockresult]),OUT/'structural_ablation.csv')
    structural=min([allresult,blockresult],key=lambda x:x['cost_total'])
    updates=[]
    for policy in UPDATES:
        r=simulate(data,a,structural['variant'],policy)
        r['marginal_saving']=None if not updates else updates[-1]['cost_total']-r['cost_total']
        updates.append(r)
    csv(pd.DataFrame(updates),OUT/'update_ablation.csv')
    if args.experiment=='updates':return
    gain=forecast_metrics(data,a)
    best=min(updates,key=lambda x:x['cost_total']);candidates=[best]
    if gain>=.02:candidates.append(simulate(data,a,best['variant'],best['policy'],interp='pchip'))
    selected=min(candidates,key=lambda x:x['cost_total'])
    candidates.append(simulate(data,a,selected['variant'],selected['policy'],interp=selected['interpolation'],correction=True))
    csv(pd.DataFrame(candidates),OUT/'final_candidates.csv')
    best=min(candidates,key=lambda x:x['cost_total']);dump(best,OUT/'best.json')
    finalize(data,best,allresult,blockresult,updates,gain)

def finalize(data,best,allresult,blockresult,updates,gain):
    directory=Path(best['directory']);dd=pd.read_csv(directory/'daily.csv');dt=pd.read_csv(directory/'detail.csv')
    csv(dd,OUT/'daily_metrics_best.csv');csv(dt,OUT/'soc_trace_best.csv')
    specified=dd[dd.date.isin(['2025-03-20','2025-06-21','2025-09-23','2025-12-21'])].copy()
    specified['cost_adjust_net']=specified.cost_adjust_up+specified.cost_adjust_down
    csv(specified,OUT/'specified_days.csv')
    episodes=[]
    for date,group in dt.groupby('date',sort=False):
        h=group.h_real.to_numpy();found=False;t=0
        while t<144:
            if h[t]<=1e-6:t+=1;continue
            start=t
            while t<144 and h[t]>1e-6:t+=1
            found=True
            episodes.append(dict(date=date,start_slot=start,end_slot=t,interval=f'{start//6}:{start%6*10:02d}-{t//6}:{t%6*10:02d}',energy_kwh=float(h[start:t].sum()),cost=float((group.price.to_numpy()[start:t]*5*h[start:t]).sum())))
        if not found:episodes.append(dict(date=date,start_slot=None,end_slot=None,interval='无',energy_kwh=0.,cost=0.))
    csv(pd.DataFrame(episodes),OUT/'emergency_events_best.csv')
    solver=[]
    for path in sorted(OUT.glob('*/solver.csv')):
        ff=pd.read_csv(path);ff.insert(0,'experiment',path.parent.name);solver.append(ff)
    csv(pd.concat(solver,ignore_index=True),OUT/'solver_metrics.csv')
    config=dict(commit='0340916',evaluation=['2025-02-01','2025-12-31'],days=334,initial_soc=6000,scenario_count=10,lookahead_hours=24,commitment_hours=6,terminal_constraint='bounds only',emergency_multiplier=5,adjustment_cost='1.5*p*up - 0.5*p*down',scenario_selection='10 most recent same weekday type first, pool latest60 same-issue complete historical paths; weekend means Sat/Sun, no inferred holiday calendar',ablation_semantics='Only enabled times optimize/use new forecasts. Disabled blocks execute latest available planned storage; BLOCK adjustment span remains exactly6h.',load_predictor='Q2 unchanged daily load ensemble, recursive48h sliced for each issue; optional 2h median bias with exp(-lag/36)',interpolation_selection='Jan out-of-sample PV MAE improvement>=2%',solver='SciPy HiGHS; LP first; on observed violation add one shared charge-mode binary per time (144 total), enforcing both c/d and emergency/charge exclusion',baseline='unavailable: missing Q3 source',best=best,input_hashes=json.loads((CACHE/'input_hashes.json').read_text('utf-8')))
    dump(config,OUT/'experiment_config.json')
    baseline()
    report(data,best,allresult,blockresult,updates,gain)

def report(data,best,ar,br,updates,gain):
    def line(name,r):
        return f"| {name} | {r['cost_total']:.2f} | {r['cost_plan']:.2f} | {r['cost_adjust_up']+r['cost_adjust_down']:.2f} | {r['cost_emergency']:.2f} | {r['energy_emergency']:.2f} | {r['adjustment_churn']:.2f} | {r['final_soc']:.2f} | {r['solver_s']:.2f} |"
    ss=pd.read_csv(Path(best['directory'])/'solver.csv')
    terminal_count=int((ss.terminal_soc<=1200+1e-6).sum())
    text='''# C题第3问：24h滚动实验报告

评价期2025-02-01至12-31，共334天。人民币元、电量kWh；全部费用来自本次程序运行和两次独立复算。原始附件和模板未覆盖。

## 原实现与baseline（问题1—7）

当前提交0340916只有第3问建模报告，没有第3问求解代码或结果；baseline无法原样运行，总费用不可用，1320.52万元没有被视为实测值。原第三问的未来信息泄漏、场景储能、规划执行一致性均无法从不存在的实现判定。第2问代码的训练窗口、递归跨日特征和安全投影已审计并复用，详见AUDIT.md。

报告描述的48h、后24h额外预测、30场景、终点SOC循环都是报告设定，不能当成已执行代码。缺少baseline，无法计算单独48h变24h的效果或相对原baseline的节省额和比例（问题25）。本次ALL/BLOCK使用同一预测缓存、K=10、共同储能变量、相同费用和执行公式，结构比较只改变允许调整的时段。

## 模型和信息合同

每次144段窗口，共同g/c/d/E，情景只有紧急购电和弃电；不做完整未来情景储能recourse。0时当天购电按原价，日内调整相对上一次有效承诺按1.5pu−0.5pv。BLOCK更远当天承诺固定；跨日proxy计入目标但不进入实际合同、账本或次日初始计划。初始SOC6000，以真实回放末SOC跨日连续。无人工终端SOC等式或价值项。

负荷直接复用Q2合法日初48h递归组合预测，四个issue切取对应24h；没有使用未来真实lag1。GBDT训练目标严格早于当日，28日样本外MAE动态权重；没有随机拆分。基础方案日内不重新训练负荷。2小时偏差校正另做可选对照。预测target_time用区间右端点，issue_time<target_time，PV曲线在issue实际观测和未来1至24小时官方预测之间积分。四个时点的预报均按发布时间+lead映射。缓存包含时间来源记录和历史场景日期，历史路径必须已经完整观测，历史issue日严格小于当前日。周末类型按星期六/日，不自行假定节假日调休。

核心Kall只执行下一6h计划动作并按当前真实SOC/净负荷安全投影。更新时间消融：被关闭的更新既不获取其新预测也不重新优化，延用最近一次优化的储能计划到下次启用更新；BLOCK仍只调整下一6h，缺席更新的区块保留0时承诺。这是对更新机会、信息和控制共同作用的策略消融，边际节省不能解释为纯PV预测的独立市场价格。

LP先运行；若出现同时充放电或情景紧急购电与充电并存，则启用每时段一个共同充电模式二元变量，同时禁止充放电并存及紧急购电充电，共144个二元变量，不按场景扩展。单日和7日试验发现前一问题不存在但后一问题确实存在；统一模式也防止后续出现充放电并存。所有结构比较采用相同的最终物理检查规则。无CVaR、复杂场景树或参数网格搜索。

## 全年比较（问题8—14、24）

| 方法 | 总费用 | 计划费 | 调整净费用 | 紧急费 | 紧急电量 | churn | 年末SOC | 累计求解时间s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原baseline | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 | 未运行 |
'''
    text+='\n'.join([line('24h-ALL',ar),line('24h-BLOCK',br),line('最终最佳方案',best)])+'\n\n'
    text+=f"BLOCK减ALL：总费用 {br['cost_total']-ar['cost_total']:.2f} 元；调整净费用 {(br['cost_adjust_up']+br['cost_adjust_down'])-(ar['cost_adjust_up']+ar['cost_adjust_down']):.2f} 元；紧急购电量 {br['energy_emergency']-ar['energy_emergency']:.2f} kWh；紧急费用 {br['cost_emergency']-ar['cost_emergency']:.2f} 元；churn {br['adjustment_churn']-ar['adjustment_churn']:.2f} kWh。BLOCK每段最多调整一次，理论及数值churn均为0。减少churn本身不保证总费用下降，保留全部变差结果。\n\n"
    text+='## 更新时间消融（问题15—22）\n\n| 更新时间策略 | 总费用 | 边际节省 | 调整费用 | 紧急费用 | 紧急电量 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in updates:
        marginal='不适用' if r.get('marginal_saving') is None else f"{r['marginal_saving']:.2f}"
        text+=f"| {r['policy']} | {r['cost_total']:.2f} | {marginal} | {r['cost_adjust_up']+r['cost_adjust_down']:.2f} | {r['cost_emergency']:.2f} | {r['energy_emergency']:.2f} |\n"
    text+=f"\n最终推荐：{best['variant']} / {best['policy']} / {best['interpolation']} / 2小时偏差修正={best['correction']}。先按全年实际费用选择ALL/BLOCK，再固定结构比较四种更新策略；未在全年误差上调参。此为同一评价年上的策略比较，不声称是独立测试集上的泛化收益。\n\n"
    text+=f"## 插值、终端与验证（问题23）\n\nPCHIP相对Linear的1月历史PV MAE改善 {gain*100:.4f}%；2%门槛事先固定。{'达到门槛，额外完成全年PCHIP对照。' if gain>=.02 else '未达到门槛，不运行全年PCHIP成本对照。'} 详细逐issue MAE/RMSE由程序重算见forecast_metrics.csv。四个同日issue的共同target仅18:10—24:00，光伏大多为零，不能据此概括白天预测；同时另存0/6/12共同下午目标和全24h可用目标的指标，不混用样本。\n\n"
    text+=f"最佳策略 {len(ss)} 次优化中，有 {terminal_count} 次计划终端SOC贴1200。未额外加终端约束；这一有限窗口现象需要如实保留，不能只凭look-ahead声称末端效应已消除。LP初解同时充放电计数 {best['lp_simultaneous_cd']}、情景紧急购电同时充电计数 {best['lp_emergency_charge']}；最终互斥检查全部通过，详见solver_metrics.csv。\n\n"
    text+=f"最终分项：计划 {best['cost_plan']:.2f}，上调 {best['cost_adjust_up']:.2f}，下调净退款 {best['cost_adjust_down']:.2f}，紧急 {best['cost_emergency']:.2f}，合计 {best['cost_total']:.2f} 元。每个实验先逐时段复算，再读取逐次承诺变更重新结算，和逐日总费用差<1e-6元。每10分钟能量平衡差<1e-6kWh，SOC/功率/跨日连续性、已执行承诺冻结、BLOCK远段不改均有断言。\n\n"
    text+='求解容差补充：全年运行中发现HiGHS默认整数容差会让z接近1却留出微小紧急购电。物理检查仍严格使用1e-6kWh；仅在失败时以mip_feasibility_tolerance=1e-10重新求解，不接受不合格解。容差含义参考[HiGHS官方选项文档](https://ergo-code.github.io/HiGHS/dev/options/definitions/#mip_feasibility_tolerance)。表中运行时间为实际累计求解时间，跨检查点重启仍累计；共享预测缓存及输出写入时间不计入。\n\n'
    text+='开发试运行的LP、逐次互斥版本及大M版本均保留在results下，并汇总于pilot_history.csv。其时域/天数或求解表达不与正式v3全年结果混用。等价数学表达的多重最优计划可能产生不同回放费用，选择紧凑表达依据是速度，未按开发试运行费用挑选。\n\n'
    text+='指定日期逐项数据见specified_days.csv；真实SOC与执行动作见soc_trace_best.csv；连续紧急事件见emergency_events_best.csv。最终Excel按原模板四表语义输出，调整购电量写g_final，非g_final−g0。\n\n没有执行 git push。\n'
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
