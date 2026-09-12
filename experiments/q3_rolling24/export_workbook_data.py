"""Prepare model-result values for the template exporter; does not write XLSX."""
import json
from pathlib import Path
import pandas as pd
import run as r

def main():
    best=json.loads((r.OUT/'best.json').read_text('utf-8'))
    d=pd.read_csv(r.OUT/'daily_metrics_best.csv')
    trace=pd.read_csv(r.OUT/'soc_trace_best.csv')
    events=pd.read_csv(r.OUT/'emergency_events_best.csv',keep_default_na=False)
    plan=[];final=[];storage=[]
    for row in d.itertuples(index=False):
        g=trace[trace.date==row.date]
        assert len(g)==144
        plan.append([row.date,*g.g0.tolist(),row.energy_g0,row.cost_plan])
        final.append([row.date,*g.g_final.tolist(),row.energy_final_commitment,row.cost_plan+row.cost_adjust_up+row.cost_adjust_down])
        for h in range(6):
            block=g.iloc[h*24:(h+1)*24]
            storage.append([row.date if h==0 else None,f'{h*4}:00-{(h+1)*4}:00',float(block.c_exec.sum()),float(block.d_exec.sum()),'0:00' if h==0 else '24:00' if h==1 else None,row.soc_0000 if h==0 else row.soc_2400 if h==1 else None])
    headers=['日期\\时段']+[f'{t//6}:{t%6*10:02d}-{(t+1)//6}:{(t+1)%6*10:02d}' for t in range(144)]+['全天购电量','全天购电费']
    emergency=events[['date','interval','energy_kwh']].values.tolist()
    data={'headers':headers,'plan':plan,'final':final,'storage':storage,'emergency':emergency,'best':best}
    r.dump(data,r.OUT/'workbook_data.json')
    print('Workbook data prepared: 334 days, 4 original template sheets.')

if __name__=='__main__':main()
