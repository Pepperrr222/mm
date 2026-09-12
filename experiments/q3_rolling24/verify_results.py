"""Independent CSV audit: original data, order settlement, execution and provenance."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import run as r

def verify_directory(directory):
    summary=json.loads((directory/'summary.json').read_text('utf-8'))
    detail=pd.read_csv(directory/'detail.csv')
    days=pd.read_csv(directory/'daily.csv')
    orders=pd.read_csv(directory/'adjustments.csv')
    solver=pd.read_csv(directory/'solver.csv')
    assert len(detail)==summary['days']*144 and len(days)==summary['days']
    expected=pd.date_range('2025-02-01',periods=summary['days']).strftime('%Y-%m-%d').tolist()
    assert days.date.tolist()==expected
    assert list(detail.groupby('date',sort=False).size())==[144]*len(days)
    E=detail.soc_start.to_numpy();F=detail.soc_end.to_numpy()
    assert np.max(np.abs(E[1:]-F[:-1]))<1e-6
    assert abs(E[0]-6000)<1e-6
    assert min(E.min(),F.min())>=1200-1e-6 and max(E.max(),F.max())<=10800+1e-6
    assert detail.c_exec.max()<=r.Q+1e-6 and detail.d_exec.max()<=r.Q+1e-6
    net=(detail.load_kw-detail.pv_kw)/6
    balance=detail.g_final+detail.d_exec+detail.h_real-net-detail.c_exec-detail.w_real
    state=F-E-.9*detail.c_exec+detail.d_exec/.9
    assert np.max(np.abs(balance))<1e-6 and np.max(np.abs(state))<1e-6
    expected_c=np.minimum.reduce([detail.c_plan.to_numpy(),(10800-E)/.9,np.maximum(detail.g_final-net,0)])
    expected_d=np.minimum.reduce([detail.d_plan.to_numpy(),.9*(E-1200),np.maximum(net-detail.g_final,0)])
    assert np.max(np.abs(expected_c-detail.c_exec))<1e-6
    assert np.max(np.abs(expected_d-detail.d_exec))<1e-6
    assert not((detail.c_exec>1e-6)&(detail.h_real>1e-6)).any()
    assert not((detail.c_exec>1e-6)&(detail.d_exec>1e-6)).any()
    assert solver.status.eq(0).all()
    if summary['mode']=='physical':
        assert solver.final_simultaneous_cd.eq(0).all() and solver.final_emergency_charge.eq(0).all()
    bill=math.fsum(float(p*g) for p,g in zip(detail.price,detail.g0))
    adj_up=adj_down=0.
    for date,group in detail.groupby('date',sort=False):
        commitment=group.g0.to_numpy().copy()
        gross=0.
        for event in orders[orders.date==date].itertuples(index=False):
            assert event.issue_hour*6<=event.t
            if summary['variant']=='BLOCK':assert event.t<event.issue_hour*6+36
            assert abs(commitment[event.t]-event.previous)<1e-6
            delta=event.new-event.previous
            assert abs(event.up-max(delta,0))<1e-6 and abs(event.down-max(-delta,0))<1e-6
            commitment[event.t]=event.new
            adj_up+=1.5*event.price*max(delta,0)
            adj_down-=.5*event.price*max(-delta,0)
            gross+=abs(delta)
        assert np.max(np.abs(commitment-group.g_final))<1e-6
        churn=gross-float(np.abs(commitment-group.g0).sum())
        row=days[days.date==date].iloc[0]
        assert abs(churn-row.adjustment_churn)<1e-6
        for hour in (0,6,12,18):
            assert abs(row[f'soc_{hour:02d}00']-group.iloc[hour*6].soc_start)<1e-6
        assert abs(row.soc_2400-group.iloc[-1].soc_end)<1e-6
    emergency=math.fsum(float(5*p*h) for p,h in zip(detail.price,detail.h_real))
    total=math.fsum([bill,adj_up,adj_down,emergency])
    assert abs(total-summary['cost_total'])<1e-6
    assert abs(bill-summary['cost_plan'])<1e-6 and abs(adj_up-summary['cost_adjust_up'])<1e-6 and abs(adj_down-summary['cost_adjust_down'])<1e-6
    return dict(experiment=directory.name,days=summary['days'],cost_recomputed=total,cost_difference=total-summary['cost_total'],balance_max=float(np.abs(balance).max()),state_max=float(np.abs(state).max()),status='pass')

def main():
    records=[]
    for path in sorted(r.OUT.glob('*physical_v3_*/summary.json')):
        records.append(verify_directory(path.parent))
    assert records
    r.csv(pd.DataFrame(records),r.OUT/'independent_verification.csv')
    for name,expected in json.loads((r.OUT/'protected_inputs_hashes.json').read_text('utf-8')).items():
        assert r.sha(r.ROOT/name)==expected
    provenance=pd.read_csv(r.OUT/'forecast_provenance.csv',parse_dates=['issue_time','training_cutoff','target_time','feature_cutoff'])
    assert (provenance.training_cutoff<=provenance.issue_time).all()
    assert (provenance.feature_cutoff<=provenance.issue_time).all()
    assert (provenance.issue_time<provenance.target_time).all()
    assert set(provenance.groupby(['day','k']).size())=={144}
    print(f'Independent audit passed for {len(records)} experiments; protected inputs unchanged.',flush=True)

if __name__=='__main__':main()
