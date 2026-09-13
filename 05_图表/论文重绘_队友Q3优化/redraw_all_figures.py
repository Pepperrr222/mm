from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = Path(r"C:\Users\Lenovo\.codex\skills\nature-figure\scripts")
sys.path.insert(0, str(AUDIT))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .55,
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white", "savefig.pad_inches": .04,
})
C = {"blue":"#3B6EA8", "orange":"#D27B35", "green":"#2A9D8F", "gray":"#7A7A7A", "dark":"#222222", "red":"#C95C5C", "purple":"#7655A8"}

def csv(path): return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=1).columns else None)
def first_csv(folder, pattern):
    xs = list(Path(folder).glob(pattern)); assert xs, (folder, pattern); return xs[0]
def save(fig, stem, multi=False):
    base = OUT / stem
    # Nature-like finishing pass: compact ticks, restrained grid, and consistent spine treatment.
    for ax in fig.axes:
        ax.tick_params(direction="out", length=2.5, width=.55, pad=2)
        ax.grid(axis="y", color="#D9DEE5", lw=.45, alpha=.55, zorder=0)
        for sp in ax.spines.values(): sp.set_linewidth(.55)
    fig.patch.set_facecolor("white")
    fig.canvas.draw()
    if multi:
        require_matplotlib_panel_alignment(fig, json_out=str(base)+".alignment.json", overlay_svg=str(base)+".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(base)+".svg", bbox_inches="tight")
    fig.savefig(str(base)+".pdf", bbox_inches="tight")
    fig.savefig(str(base)+".png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def label(ax, s): ax.text(-.13, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q1 = pd.read_csv(ROOT/"04_结果"/"问题1_逐时段结果.csv")
    q2d = pd.read_csv(ROOT/"04_结果"/"问题2_每日指标.csv", parse_dates=["date"])
    q2t = pd.read_csv(ROOT/"04_结果"/"问题2_逐时段结果.csv", parse_dates=["date"])
    q2f = pd.read_csv(ROOT/"04_结果"/"问题2_预测指标.csv", parse_dates=["date"])
    q3dir = ROOT/"04_结果"/"问题3_10场景完整求解"
    q3d = pd.read_csv(q3dir/"问题3_每日指标.csv", parse_dates=["date"])
    q3t = pd.read_csv(q3dir/"问题3_逐时段结果.csv", parse_dates=["date"])
    q3u = pd.read_csv(q3dir/"问题3_更新级验证.csv", parse_dates=["date"])
    q4old = ROOT/"04_结果"/"问题4_2_完整求解"; q4old3 = ROOT/"04_结果"/"问题4_3_完整求解"
    q4new2 = ROOT/"04_结果"/"问题4_队友Q3优化"/"问题4_2"; q4new3 = ROOT/"04_结果"/"问题4_队友Q3优化"/"问题4_3"
    q42old = pd.read_csv(first_csv(q4old,"*两方案汇总.csv")); q43old = pd.read_csv(first_csv(q4old3,"*两方案汇总.csv"))
    q42 = pd.read_csv(q4new2/"每日指标.csv", parse_dates=["date"]); q43 = pd.read_csv(q4new3/"每日指标.csv", parse_dates=["date"])
    q42t = pd.read_csv(q4new2/"逐时段结果.csv", parse_dates=["date"]); q43t = pd.read_csv(q4new3/"逐时段结果.csv", parse_dates=["date"])
    # Q1 raw input and dispatch
    x=np.arange(len(q1))/6
    fig,ax=plt.subplots(figsize=(5.7,2.7)); ax.plot(x,q1.load_kw,color=C["dark"],lw=.85,label="Load",zorder=3); ax.plot(x,q1.pv_kw,color=C["green"],lw=.85,label="PV",zorder=3); ax.fill_between(x,q1.pv_kw,0,color=C["green"],alpha=.06,zorder=1); ax.set(xlabel="Time (h)",ylabel="Power (kW)"); ax.legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),handlelength=1.4,columnspacing=1.2); ax.grid(axis="y",alpha=.2); save(fig,"raw_q1_load_pv")
    fig,ax=plt.subplots(figsize=(5.7,2.7)); ax.plot(x,q1.grid_kwh,color=C["blue"],label="Grid purchase"); ax.plot(x,q1.charge_kwh,color=C["green"],label="Charge"); ax.plot(x,q1.discharge_kwh,color=C["orange"],label="Discharge"); ax.set(xlabel="Time (h)",ylabel="Energy (kWh)"); ax.legend(frameon=False,ncol=3,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"process_q1_dispatch")
    fig,ax=plt.subplots(figsize=(5.7,2.7)); ax.plot(x,q1.E_start_kwh,color=C["blue"],lw=1); ax.axhline(1200,color=C["gray"],ls="--",lw=.7); ax.axhline(10800,color=C["gray"],ls="--",lw=.7); ax.set(xlabel="Time (h)",ylabel="Stored energy (kWh)"); ax.grid(axis="y",alpha=.2); save(fig,"process_q1_storage")
    baseline=float(np.sum(np.maximum(q1.load_kw-q1.pv_kw,0)*q1.price/6)); opt=float((q1.grid_kwh*q1.price).sum()); fig,ax=plt.subplots(figsize=(3.8,2.7)); ax.bar(["Without storage","Optimized"],[baseline,opt],color=[C["gray"],C["blue"]],width=.55); ax.set_ylabel("Cost (CNY)"); ax.grid(axis="y",alpha=.2); save(fig,"result_q1_cost")
    # Q2
    fig,ax=plt.subplots(figsize=(4.2,2.9)); vals=[q2f.loc[q2f.signal==s,"nmae"].to_numpy()*100 for s in ["load","pv"]]; bp=ax.boxplot(vals,patch_artist=True,showfliers=False,tick_labels=["Load","PV"]); [b.set_facecolor(c) for b,c in zip(bp["boxes"],[C["blue"],C["green"]])]; ax.set_ylabel("Daily NMAE (%)"); ax.grid(axis="y",alpha=.2); save(fig,"raw_q2_forecast_error_distribution")
    fig,ax=plt.subplots(figsize=(5.8,2.9)); ax.fill_between(q2d.date,1200,10800,color=C["green"],alpha=.08,label="Feasible range"); ax.plot(q2d.date,q2d.E_start_kwh,color=C["gray"],ls="--",lw=.8,label="Start"); ax.plot(q2d.date,q2d.E_end_kwh,color=C["blue"],lw=1,label="End"); ax.set_ylabel("Stored energy (kWh)"); ax.legend(frameon=False,ncol=3,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"process_q2_storage_state")
    q2d["month"]=q2d.date.dt.month; mc=q2d.groupby("month")[["plan_cost_yuan","emergency_cost_yuan"]].sum()/1e6; fig,ax=plt.subplots(figsize=(5.8,2.9)); ax.bar(mc.index,mc.plan_cost_yuan,color=C["blue"],label="Planned"); ax.bar(mc.index,mc.emergency_cost_yuan,bottom=mc.plan_cost_yuan,color=C["red"],label="Emergency"); ax.set(xlabel="Month",ylabel="Cost (million CNY)"); ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"result_q2_monthly_cost")
    # Solver-mode audit and specified dates (original paper slots retained)
    fig,ax=plt.subplots(figsize=(4.2,2.9))
    if "solver_mode" in q2d.columns:
        counts=q2d.solver_mode.value_counts()
    elif "fallback" in q2d.columns:
        counts=q2d.fallback.map({True:"Fallback",False:"Main solver"}).value_counts()
    else:
        counts=pd.Series({"Main solver":len(q2d)})
    ax.bar(counts.index.astype(str),counts.to_numpy(),color=[C["blue"],C["gray"]][:len(counts)]); ax.set_ylabel("Number of days"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.2); save(fig,"process_q2_solver_modes")
    dates_q2=pd.to_datetime(["2025-02-01","2025-05-01","2025-08-01","2025-11-01"])
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.2),sharex=True,sharey=True)
    for ax,d,lab in zip(axs.flat,dates_q2,["a","b","c","d"]):
        z=q2t[q2t.date==d]; ax.plot(z.t/6,z.grid_kwh,color=C["blue"],lw=.9,label="Planned grid"); ax.plot(z.t/6,z.emergency_kwh,color=C["red"],lw=.8,label="Emergency"); ax.set_title(str(d.date())); label(ax,lab); ax.grid(axis="y",alpha=.2)
    axs[0,0].legend(frameon=False,fontsize=6,loc="lower left",bbox_to_anchor=(0,1.01)); fig.supxlabel("Time (h)"); fig.supylabel("Energy (kWh)"); save(fig,"result_q2_specified_dates",True)
    # Q3
    g=q3u.groupby(["policy","issue_slot"])[["up_kwh","down_kwh"]].mean().reset_index(); fig,axs=plt.subplots(1,2,figsize=(6.5,2.8),sharey=True); 
    for ax,p,lab in zip(axs,["H24","H48"],["a","b"]):
        z=g[g.policy==p]; xx=np.arange(len(z)); ax.bar(xx-.18,z.up_kwh,.36,color=C["green"],label="Upward"); ax.bar(xx+.18,z.down_kwh,.36,color=C["orange"],label="Cancellation"); ax.set_xticks(xx,[f"{int(k/6)}:00" for k in z.issue_slot]); ax.set_xlabel("Update time"); ax.set_title(p); label(ax,lab)
    axs[0].set_ylabel("Mean adjustment (kWh)"); axs[1].legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01)); save(fig,"process_q3_adjustment_epochs",True)
    s=q3d.groupby("policy")[["baseline_cost_yuan","adjustment_cost_yuan","emergency_cost_yuan"]].sum()/1e6; fig,ax=plt.subplots(figsize=(4.5,2.9)); bottom=np.zeros(2); 
    for col,c,lab in zip(s.columns,[C["blue"],C["orange"],C["red"]],["Baseline","Adjustment","Emergency"]): ax.bar(["24 h","48 h"],s[col].to_numpy(),bottom=bottom,color=c,label=lab); bottom+=s[col].to_numpy()
    ax.set_ylabel("Annual cost (million CNY)"); ax.legend(frameon=False,ncol=3,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"result_q3_cost_components")
    p=q3d.pivot(index="date",columns="policy",values="total_cost_yuan").sort_index(); delta=p.H24-p.H48; fig,ax=plt.subplots(figsize=(5.8,2.8)); ax.plot(p.index,delta.cumsum(),color=C["purple"],lw=1); ax.axhline(0,color=C["gray"],lw=.7); ax.set(xlabel="Date",ylabel="Cumulative cost difference (CNY)"); ax.grid(axis="y",alpha=.2); save(fig,"result_q3_cumulative_cost_difference")
    dates=pd.to_datetime(["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]); fig,axs=plt.subplots(2,2,figsize=(6.5,4.2),sharex=True,sharey=True); 
    for ax,d,lab in zip(axs.flat,dates,["a","b","c","d"]):
        for p0,c in [("H24",C["blue"]),("H48",C["orange"])]: z=q3t[(q3t.policy==p0)&(q3t.date==d)]; ax.plot(z.t/6,z.E_end_kwh,color=c,label=p0,lw=.9)
        ax.set_title(str(d.date())); label(ax,lab)
    axs[0,0].legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01)); fig.supxlabel("Time (h)"); fig.supylabel("Stored energy (kWh)"); save(fig,"process_q3_storage_specified",True)
    # Q4 original versus isolated migrated improvement
    old42=q42old.set_index("policy").loc[["H24","H48"],"total_cost_yuan"].to_numpy()/1e6; old43=q43old.set_index("policy").loc[["H24","H48"],"total_cost_yuan"].to_numpy()/1e6
    old=np.array([old42.mean(),old43.mean()]); new=np.array([q42.total_cost_yuan.sum(),q43.total_cost_yuan.sum()])/1e6
    fig,ax=plt.subplots(figsize=(5.2,3.0)); xx=np.arange(2); ax.bar(xx-.18,old,.36,color=C["gray"],label="Original Q4 (mean H24/H48)"); ax.bar(xx+.18,new,.36,color=C["green"],label="Q3-improved isolated"); ax.set_xticks(xx,["Q4-2","Q4-3"]); ax.set_ylabel("Annual cost (million CNY)"); ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"result_q4_original_vs_optimized")
    # Preserve the exact original Q4 comparison as a separate paper-slot figure.
    fig,axs=plt.subplots(1,2,figsize=(6.1,2.8),sharey=True)
    for ax,title,vals in zip(axs,["Q4-2","Q4-3"],[old42,old43]):
        ax.bar(["H24","H48"],vals,color=[C["blue"],C["orange"]]); ax.set_title(title); ax.grid(axis="y",alpha=.2)
    axs[0].set_ylabel("Annual cost (million CNY)"); save(fig,"result_q4_total_cost_comparison",True)
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.2),sharex=True); 
    for ax,d,lab in zip(axs.flat,dates,["a","b","c","d"]):
        z=q43t[q43t.date==d]; ax.plot(z.slot/6,z.original_plan_kwh,color=C["gray"],lw=.8,label="Day-ahead plan"); ax.plot(z.slot/6,z.final_commitment_kwh,color=C["green"],lw=1,label="Final commitment"); ax.plot(z.slot/6,z.soc_end_kwh/1000,color=C["blue"],lw=.8,label="SOC / 1000"); ax.set_title(str(d.date())); label(ax,lab); ax.grid(axis="y",alpha=.2)
    axs[0,0].legend(frameon=False,fontsize=6,loc="lower left",bbox_to_anchor=(0,1.01)); fig.supxlabel("Time (h)"); fig.supylabel("kWh / scaled SOC"); save(fig,"result_q4_optimized_specified",True)
    # Q4 bias effect, from teammate report metrics and daily costs
    fm=Path(ROOT/"paper_q3_improvements"/"load_bias_2h"/"results"/"load_bias_forecast_metrics.csv"); f=pd.read_csv(fm); z=f[(f.issue_hour.astype(str)=="combined") & (f.horizon.isin(["0-2h","2-6h","6-12h","12-24h"]))]; piv=z.pivot(index="horizon",columns="model",values="mae_kw").reindex(["0-2h","2-6h","6-12h","12-24h"]); fig,ax=plt.subplots(figsize=(5.2,3.0)); xx=np.arange(len(piv)); ax.plot(xx,piv.P0,marker="o",color=C["gray"],label="P0 baseline"); ax.plot(xx,piv.P1,marker="o",color=C["green"],label="P1 2-h bias correction"); ax.set_xticks(xx,piv.index); ax.set_ylabel("Load MAE (kW)"); ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01)); ax.grid(axis="y",alpha=.2); save(fig,"process_q4_load_bias")
    print(f"generated figures in {OUT}")

if __name__ == "__main__": main()
