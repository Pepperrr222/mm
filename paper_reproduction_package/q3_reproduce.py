"""Faithful reproduction of the paper's Question 3 rolling policy."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from q2_forecast import q2, day_type
from q2_reproduce import PWL, _simplify, average_pwl, _largest_argmin, EMIN, EMAX, E0, ETA, S, V, N

HERE=Path(__file__).resolve().parent; RESULTS=HERE/'results'; CACHE=HERE/'cache_q3.npz'
ISSUES=(0,36,72,108); POLICIES={'K0':(0,), 'K06':(0,36), 'K0612':(0,36,72), 'Kall':ISSUES, 'frozen':ISSUES}

def official_inputs(data):
    raw=pd.read_excel(q2.ROOT/'01_题目与数据/原始附件/附件3.xlsx')
    raw.iloc[:,0]=raw.iloc[:,0].ffill(); out=np.full((365,4,24),np.nan)
    for row in raw.itertuples(index=False,name=None):
        d=(pd.Timestamp(row[0])-data.dates[0]).days; h=int(str(row[1]).split(':')[0]); out[d,h//6]=row[2:]
    assert np.isfinite(out).all(); return out

def interp24(anchor, hourly):
    knots=np.r_[anchor,hourly]
    # Paper (35): the interpolated right endpoint approximates each interval mean.
    return np.maximum(np.interp(np.arange(1,145)/6,np.arange(25),knots),0)

def next_day_load(day,data,f):
    dates=pd.DatetimeIndex(data.dates); load_e=data.load_kw/6; B=f['B_real']; kind=f['kind']
    target=dates[day]+pd.Timedelta(days=1); kt=day_type(target); kc=kind[day]
    prior=np.where((np.arange(365)<day)&(kind==kt))[0][-3:]
    if len(prior)==0: prior=np.arange(max(0,day-7),day)
    shape=load_e[prior].sum(0)/B[prior].sum(); base=f['B_hat'][day]
    if kt!=kc:
        start=max(1,day-35); sw=[i for i in range(start,day) if kind[i]!=kind[i-1]]
        if sw:
            beta=np.median([np.log(B[i]/B[i-1])/(kind[i]-kind[i-1]) for i in sw])
            base*=np.exp(beta*(kt-kc))
    return 6*base*shape

def prepare():
    if CACHE.exists(): return q2.read_inputs(),dict(np.load(CACHE))
    data=q2.read_inputs(); off=official_inputs(data); f=dict(np.load(HERE/'cache_q2_forecasts.npz'))
    lp=np.full((365,4,N),np.nan); pv=np.full_like(lp,np.nan); actualL=np.full_like(lp,np.nan); actualP=np.full_like(lp,np.nan)
    hist=f['pv_kw']; off0=np.empty((365,N))
    for d in range(365): off0[d]=interp24(data.pv_kw[d-1,-1] if d else 0,off[d,0])
    ef=(data.pv_kw[14:31]-off0[14:31]).ravel(); eh=(data.pv_kw[14:31]-hist[14:31]).ravel()
    w=float(np.clip(np.sum(eh*(eh-ef))/np.sum((ef-eh)**2),0,1))
    weights=np.zeros(365)
    # Historical OOS residuals use a weight estimated only from dates before that issue.
    for d in range(2,31):
        a=slice(1,d); efd=(data.pv_kw[a]-off0[a]).ravel(); ehd=(data.pv_kw[a]-hist[a]).ravel()
        weights[d]=np.clip(np.sum(ehd*(ehd-efd))/np.sum((efd-ehd)**2),0,1)
    weights[31:]=w
    for d in range(1,365):
        nxt=next_day_load(d,data,f) if d<364 else next_day_load(d,data,f)
        L2=np.r_[f['load_kw'][d],nxt]
        A2=np.r_[data.load_kw[d], data.load_kw[d+1] if d<364 else np.full(N,np.nan)]
        P2=np.r_[data.pv_kw[d], data.pv_kw[d+1] if d<364 else np.full(N,np.nan)]
        for j,k in enumerate(ISSUES):
            lp[d,j]=L2[k:k+N]; actualL[d,j]=A2[k:k+N]; actualP[d,j]=P2[k:k+N]
            anchor=data.pv_kw[d,k-1] if k else data.pv_kw[d-1,-1]
            po=interp24(anchor,off[d,j]); pv[d,j]=weights[d]*po+(1-weights[d])*hist[d] if k==0 else po
    np.savez_compressed(CACHE,load=lp,pv=pv,actual_load=actualL,actual_pv=actualP,weight=w,weights=weights)
    mae=float(np.nanmean(np.abs(data.pv_kw[31:]-pv[31:,0])))
    (RESULTS/'paper_q3_forecast.json').write_text(json.dumps({'weight':w,'pv_mae_kw':mae},indent=2),encoding='utf8')
    return data,dict(np.load(CACHE))

def scenarios(day,j,a,frozen=False):
    candidates=[i for i in range(max(1,day-35),day) if np.isfinite(a['actual_load'][i,j]).all()][-30:]
    idx=np.asarray(candidates); el=a['actual_load'][idx,j]-a['load'][idx,j]
    if frozen and j:
        k=ISSUES[j]; hist=np.load(HERE/'cache_q2_forecasts.npz')['pv_kw']
        base=np.r_[a['pv'][day,0,k:],hist[day,:k]]
        pred=np.stack([np.r_[a['pv'][i,0,k:],hist[i,:k]] for i in idx])
        ep=a['actual_pv'][idx,j]-pred
    else:
        base=a['pv'][day,j]; ep=a['actual_pv'][idx,j]-a['pv'][idx,j]
    return np.maximum(0,a['load'][day,j]+el),np.maximum(0,base+ep)

def solve_roll(price,g0,k,load_s,pv_s,soc):
    m=len(load_s); qn=N-k; nf=k; common=qn+(0 if k==0 else 2*qn); per=nf+5*N+(N+1); total=common+m*per
    obj=np.zeros(total); bounds=[(0,None)]*total; obj[:qn]=price[k:] 
    dp0=qn; dm0=2*qn
    if k: obj[dp0:dp0+qn]=.5*price[k:]; obj[dm0:dm0+qn]=.5*price[k:]
    def ix(s,name,t):
        base=common+s*per; off={'gf':0,'b':nf,'C':nf+N,'D':nf+2*N,'E':nf+3*N,'U':nf+4*N+1}; return base+off[name]+t
    hp=np.r_[price[k:],price[:k]]
    for s in range(m):
        for t in range(nf): obj[ix(s,'gf',t)]=price[t]/m
        for t in range(N):
            obj[ix(s,'b',t)]=5*hp[t]/m; bounds[ix(s,'C',t)]=(0,S); bounds[ix(s,'D',t)]=(0,S)
        for t in range(N+1): bounds[ix(s,'E',t)]=(EMIN,EMAX)
        bounds[ix(s,'E',0)]=(soc,soc); obj[ix(s,'E',N)]=-V/m
    rr=[];cc=[];vv=[];rhs=[]; row=0; net=(load_s-pv_s)/6
    if k:
        for t in range(qn):
            for col,val in ((t,1),(dp0+t,-1),(dm0+t,1)): rr.append(row);cc.append(col);vv.append(val)
            rhs.append(float(g0[k+t]));row+=1
    for s in range(m):
        for t in range(N):
            z=t if t<qn else ix(s,'gf',t-qn)
            for col,val in ((z,1),(ix(s,'b',t),1),(ix(s,'D',t),1),(ix(s,'C',t),-1),(ix(s,'U',t),-1)):
                rr.append(row);cc.append(col);vv.append(val)
            rhs.append(float(net[s,t]));row+=1
        for t in range(N):
            for col,val in ((ix(s,'E',t+1),1),(ix(s,'E',t),-1),(ix(s,'C',t),-ETA),(ix(s,'D',t),1/ETA)):
                rr.append(row);cc.append(col);vv.append(val)
            rhs.append(0);row+=1
    A=coo_matrix((vv,(rr,cc)),shape=(row,total)).tocsr(); r=linprog(obj,A_eq=A,b_eq=rhs,bounds=bounds,method='highs')
    if not r.success: raise RuntimeError(r.message)
    return r.x[:qn]

def infconv(f:PWL, rx, ry):
    widths=np.r_[np.diff(f.x),np.diff(rx)]; slopes=np.r_[np.diff(f.y)/np.diff(f.x),np.diff(ry)/np.diff(rx)]
    order=np.argsort(slopes,kind='stable'); xx=[f.x[0]+rx[0]]; yy=[f.y[0]+ry[0]]
    for i in order: xx.append(xx[-1]+widths[i]); yy.append(yy[-1]+slopes[i]*widths[i])
    xx=np.asarray(xx);yy=np.asarray(yy); pts=np.unique(np.r_[EMIN,xx[(xx>EMIN)&(xx<EMAX)],EMAX]); return _simplify(pts,np.interp(pts,xx,yy))

def bellman_general(f:PWL, net:float, p:float):
    # y=-x; q(-y)=p*[net+psi(-y)]+ on y in [-eta*S,S/eta].
    y=np.array([-ETA*S,0,S/ETA]);
    if net<0: y=np.r_[y,-ETA*net]
    elif net>0: y=np.r_[y,net/ETA]
    y=np.unique(np.clip(y,-ETA*S,S/ETA)); x=-y
    psi=np.where(x>=0,x/ETA,ETA*x); val=p*np.maximum(net+psi,0)
    return infconv(f,y,val)

def reserves(price,net_s,a,qn):
    m=len(net_s); fs=[PWL(np.array([EMIN,EMAX]),np.array([-V*EMIN,-V*EMAX])) for _ in range(m)]; hp=np.r_[price[N-qn:],price[:N-qn]]; R=np.empty(qn)
    for t in range(N-1,-1,-1):
        mean=average_pwl(fs)
        if t<qn: R[t]=_largest_argmin(mean,5*hp[t]*ETA)
        new=[]
        for s in range(m):
            if t<qn:
                from q2_reproduce import bellman
                new.append(bellman(fs[s],net_s[s,t]-a[t],hp[t]))
            else: new.append(bellman_general(fs[s],net_s[s,t],hp[t]))
        fs=new
    return R

def run(policy, days=334):
    data,a=prepare(); rows=[]; soc=E0; started=time.perf_counter(); trace=[]
    updates=ISSUES if policy=='frozen' else POLICIES[policy]
    for day in range(31,min(365,31+days)):
        g0=None; delivered=np.zeros(N); emerg=np.zeros(N); waste=np.zeros(N)
        for pos,k in enumerate(updates):
            j=ISSUES.index(k); ls,ps=scenarios(day,j,a,policy=='frozen'); plan=solve_roll(data.price,g0,k,ls,ps,soc)
            if k==0: g0=plan.copy()
            stop=updates[pos+1] if pos+1<len(updates) else N
            net_s=(ls-ps)/6; R=reserves(data.price,net_s,plan,N-k)
            for t in range(k,stop):
                z=t-k; r=(data.load_kw[day,t]-data.pv_kw[day,t])/6-plan[z]
                d=min(r,S,ETA*max(soc-R[z],0)) if r>0 else 0.; c=min(-r,S,(EMAX-soc)/ETA) if r<=0 else 0.
                h=max(r+c-d,0); wst=max(-r-c+d,0); soc+=ETA*c-d/ETA
                delivered[t]=plan[z];emerg[t]=h;waste[t]=wst
        up=np.maximum(delivered-g0,0);down=np.maximum(g0-delivered,0)
        normal=float(np.sum(data.price*(delivered+.5*(up+down)))); ec=float(np.sum(5*data.price*emerg))
        rows.append(dict(date=str(data.dates[day].date()),plan_cost=float(data.price@g0),settled_normal_cost=normal,
                         emergency_cost=ec,total_cost=normal+ec,planned_energy=float(g0.sum()),delivered_energy=float(delivered.sum()),
                         up_energy=float(up.sum()),down_energy=float(down.sum()),emergency_energy=float(emerg.sum()),waste_energy=float(waste.sum()),final_soc=soc))
        print(rows[-1]['date'],f"{rows[-1]['total_cost']:.2f}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(RESULTS/f'paper_q3_daily_{policy}.csv',index=False,encoding='utf-8-sig')
    sm={k:float(df[k].sum()) for k in ['plan_cost','settled_normal_cost','emergency_cost','total_cost','planned_energy','delivered_energy','up_energy','down_energy','emergency_energy','waste_energy']};sm.update(policy=policy,final_soc=float(soc),weight=float(a['weight']),runtime_seconds=time.perf_counter()-started)
    (RESULTS/f'paper_q3_summary_{policy}.json').write_text(json.dumps(sm,indent=2),encoding='utf8');print(json.dumps(sm,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=POLICIES,required=True);p.add_argument('--days',type=int,default=334);z=p.parse_args();run(z.policy,z.days)
