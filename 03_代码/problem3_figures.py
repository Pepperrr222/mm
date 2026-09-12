from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"04_结果"/"问题3_10场景完整求解"
FIG=ROOT/"05_图表"/"问题3"
FIG.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,r"C:\Users\Lenovo\.codex\skills\math-modeling\tools\figure\scripts")
from export_figure import export_figure

plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Arial"],"axes.unicode_minus":False,
                     "font.size":8,"axes.spines.top":False,"axes.spines.right":False})
C={"H24":"#0072B2","H48":"#D55E00"}

daily=pd.read_csv(RES/"问题3_每日指标.csv",parse_dates=["date"])
detail=pd.read_csv(RES/"问题3_逐时段结果.csv",parse_dates=["date"])
updates=pd.read_csv(RES/"问题3_更新级验证.csv",parse_dates=["date"])

def save(fig,name,size=(6.8,3.8)):
    export_figure(fig,str(FIG/name),formats=["svg","png"],dpi=300,size_inches=size,
                  grayscale_preview=False,tight=False)
    plt.close(fig)

# raw 1: representative-day load and PV
sel=pd.Timestamp("2025-06-21"); x=np.arange(144)/6
d=detail[(detail.policy=="H24")&(detail.date==sel)]
fig,ax=plt.subplots(); ax.plot(x,d.load_kw,label="负荷",color="#333333"); ax.plot(x,d.pv_kw,label="光伏",color="#009E73")
ax.set(xlabel="时刻/h",ylabel="功率/kW"); ax.legend(frameon=False,ncol=2); save(fig,"raw_q3_load_pv_day")

# raw 2: daily load/PV energy by month
tmp=detail[detail.policy=="H24"].copy(); tmp["month"]=tmp.date.dt.month
m=tmp.groupby("month").agg(load=("load_kw","sum"),pv=("pv_kw","sum"))* (1/6)
fig,ax=plt.subplots(); ax.plot(m.index,m.load/1000,"o-",label="负荷"); ax.plot(m.index,m.pv/1000,"s--",label="光伏")
ax.set(xlabel="月份",ylabel="月电量/MWh",xticks=range(2,13)); ax.legend(frameon=False); save(fig,"raw_q3_monthly_energy")

# raw 3: emergency distribution
fig,ax=plt.subplots(); vals=[daily[daily.policy==p].emergency_kwh for p in ["H24","H48"]]
ax.boxplot(vals,tick_labels=["24 h","48 h"],showfliers=False); ax.set_ylabel("每日紧急购电量/kWh"); save(fig,"raw_q3_emergency_distribution",(4.0,3.2))

# process 1: solver time
fig,ax=plt.subplots(); vals=[daily[daily.policy==p].solver_time_s for p in ["H24","H48"]]
ax.boxplot(vals,tick_labels=["24 h","48 h"],showfliers=False); ax.set_ylabel("每日求解时间/s"); save(fig,"process_q3_solver_time",(4.0,3.2))

# process 2: adjustment by issue time
g=updates.groupby(["policy","issue_slot"]).agg(up=("up_kwh","mean"),down=("down_kwh","mean")).reset_index()
fig,axs=plt.subplots(1,2,sharey=True)
for ax,p in zip(axs,["H24","H48"]):
    z=g[g.policy==p]; xx=np.arange(len(z)); ax.bar(xx-.18,z.up,.36,label="上调",color=C["H24"]); ax.bar(xx+.18,z.down,.36,label="下调",color=C["H48"])
    ax.set_xticks(xx,[f"{int(k/6)}:00" for k in z.issue_slot]); ax.set_title(p); ax.set_xlabel("更新时刻")
axs[0].set_ylabel("平均调整量/kWh"); axs[1].legend(frameon=False); save(fig,"process_q3_adjustment_epochs")

# process 3: storage paths on four dates
dates=pd.to_datetime(["2025-03-20","2025-06-21","2025-09-23","2025-12-21"])
fig,axs=plt.subplots(2,2,sharex=True,sharey=True)
for ax,date in zip(axs.flat,dates):
    for p in ["H24","H48"]:
        z=detail[(detail.policy==p)&(detail.date==date)]; ax.plot(np.arange(len(z))/6,z.E_end_kwh,color=C[p],label=p)
    ax.set_title(str(date.date())); ax.set_ylim(1100,10900)
axs[0,0].legend(frameon=False); fig.supxlabel("时刻/h"); fig.supylabel("储电量/kWh"); save(fig,"process_q3_storage_specified",(6.8,4.8))

# result 1: cumulative cost difference
p=daily.pivot(index="date",columns="policy",values="total_cost_yuan").sort_index(); delta=p.H24-p.H48
fig,ax=plt.subplots(); ax.plot(p.index,delta.cumsum(),color="#7A0177"); ax.axhline(0,color="#777777",lw=.8); ax.set(xlabel="日期",ylabel="累计费用差：24h−48h/元"); save(fig,"result_q3_cumulative_cost_difference")

# result 2: total cost components
cols=["baseline_cost_yuan","adjustment_cost_yuan","emergency_cost_yuan"]
s=daily.groupby("policy")[cols].sum()/1e6
fig,ax=plt.subplots(); bottom=np.zeros(2)
for col,color,label in zip(cols,["#56B4E9","#E69F00","#CC79A7"],["基准购电","调整结算","紧急购电"]):
    ax.bar(["24 h","48 h"],s[col].values,bottom=bottom,label=label,color=color); bottom+=s[col].values
ax.set_ylabel("全年费用/百万元"); ax.legend(frameon=False,ncol=3,fontsize=7); save(fig,"result_q3_cost_components",(5.2,3.5))

# result 3: paired daily differences and CI
fig,ax=plt.subplots(); ax.hist(delta,bins=30,color="#56B4E9",edgecolor="white"); ax.axvline(delta.mean(),color="#D55E00",label=f"均值 {delta.mean():.1f} 元/日")
ax.axvline(0,color="#333333",lw=.8); ax.set(xlabel="每日费用差：24h−48h/元",ylabel="天数"); ax.legend(frameon=False); save(fig,"result_q3_daily_difference_distribution")

print(f"generated 9 logical figures in {FIG}")
