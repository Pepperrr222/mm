"""Nature-style test figures for the strict 10-scenario Question 3 run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import sys

SKILL = Path(r"C:/Users/Lenovo/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(r"D:/Git/mm")
RES = ROOT / "04_结果" / "问题3_10场景完整求解"
OUT = ROOT / "05_图表" / "问题3_10场景测试"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.frameon": False,
})
BLUE, ORANGE, GREY, RED = "#356FAE", "#D9822B", "#737373", "#B24C63"

daily = pd.read_csv(RES / "问题3_每日指标.csv", encoding="utf-8-sig", parse_dates=["date"])
pair = pd.read_csv(RES / "问题3_每日配对差.csv", encoding="utf-8-sig", parse_dates=["date"])
detail = pd.read_csv(RES / "问题3_逐时段结果.csv", encoding="utf-8-sig", parse_dates=["date"])

def save(fig, stem, aligned=False):
    if aligned:
        require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{stem}.alignment.json"),
            overlay_svg=str(OUT / f"{stem}.alignment.svg"), tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

# Figure 1: one claim, four complementary panels.
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
for a, lab in zip(ax.flat, ["a", "b", "c", "d"]):
    a.text(-0.14, 1.05, lab, transform=a.transAxes, fontsize=8, fontweight="bold", va="top")
cost = daily.pivot(index="date", columns="policy", values="total_cost_yuan")
ax[0,0].plot(cost.index, cost.H24/1000, color=GREY, lw=0.8, label="H24")
ax[0,0].plot(cost.index, cost.H48/1000, color=BLUE, lw=1.0, label="H48")
ax[0,0].set_ylabel("Daily cost (10³ yuan)"); ax[0,0].set_title("Rolling daily operating cost", loc="left")
ax[0,0].legend(fontsize=6, ncol=2); ax[0,0].xaxis.set_major_locator(MonthLocator()); ax[0,0].xaxis.set_major_formatter(DateFormatter("%b"))
ax[0,1].plot(pair.date, pair.H24_minus_H48_yuan, color=ORANGE, lw=0.8)
ax[0,1].axhline(0, color="black", lw=0.5); ax[0,1].set_ylabel("H24 − H48 (yuan)"); ax[0,1].set_title("Paired daily cost difference", loc="left")
ax[0,1].xaxis.set_major_locator(MonthLocator()); ax[0,1].xaxis.set_major_formatter(DateFormatter("%b"))
cum = pair.set_index("date").H24_minus_H48_yuan.cumsum()
ax[1,0].plot(cum.index, cum/1000, color=RED, lw=1.1)
ax[1,0].axhline(0, color="black", lw=0.5); ax[1,0].set_ylabel("Cumulative saving (10³ yuan)"); ax[1,0].set_title("Accumulated advantage of H48", loc="left")
ax[1,0].xaxis.set_major_locator(MonthLocator()); ax[1,0].xaxis.set_major_formatter(DateFormatter("%b"))
storage = daily.pivot(index="date", columns="policy", values="end_storage_kwh")
ax[1,1].plot(storage.index, storage.H24/1000, color=GREY, lw=0.8, label="H24")
ax[1,1].plot(storage.index, storage.H48/1000, color=BLUE, lw=1.0, label="H48")
ax[1,1].set_ylabel("End storage (10³ kWh)"); ax[1,1].set_title("Daily terminal storage", loc="left")
ax[1,1].xaxis.set_major_locator(MonthLocator()); ax[1,1].xaxis.set_major_formatter(DateFormatter("%b"))
for a in ax.flat: a.grid(axis="y", color="#D9D9D9", lw=0.35); a.tick_params(length=2)
save(fig, "python_figure1_yearly_comparison", aligned=True)

# Figure 2: representative-day operational trajectory.
day = pd.Timestamp("2025-06-21")
sub = detail[detail.date.eq(day)].copy(); sub["hour"] = sub["t"] / 6
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True, constrained_layout=True)
ax[0].plot(sub[sub.policy.eq("H24")].hour, sub[sub.policy.eq("H24")].E_end_kwh/1000, color=GREY, lw=1.0, label="H24")
ax[0].plot(sub[sub.policy.eq("H48")].hour, sub[sub.policy.eq("H48")].E_end_kwh/1000, color=BLUE, lw=1.1, label="H48")
ax[0].set_ylabel("Storage (10³ kWh)"); ax[0].set_title("Representative day: storage trajectory", loc="left")
ax[0].set_xlim(0, 27)
ax[0].text(24.5, sub[sub.policy.eq("H24")].E_end_kwh.iloc[-1]/1000, "H24", color=GREY, va="center", fontsize=6)
ax[0].text(24.5, sub[sub.policy.eq("H48")].E_end_kwh.iloc[-1]/1000, "H48", color=BLUE, va="center", fontsize=6)
for policy, col in [("H24", GREY), ("H48", BLUE)]:
    z = sub[sub.policy.eq(policy)]
    ax[1].plot(z.hour, z.emergency_kwh, color=col, lw=0.9, label=policy)
ax[1].set_xlabel("Time of day (h)"); ax[1].set_ylabel("Emergency purchase (kWh)"); ax[1].set_title("Emergency purchase during the same day", loc="left")
for a in ax: a.grid(axis="y", color="#D9D9D9", lw=0.35); a.tick_params(length=2)
ax[1].set_xticks(np.arange(0, 25, 4)); ax[1].set_xlim(0, 27)
e24 = float(sub[sub.policy.eq("H24")].emergency_kwh.iloc[-1]); e48 = float(sub[sub.policy.eq("H48")].emergency_kwh.iloc[-1])
if abs(e24-e48) < 1e-6:
    ax[1].text(24.5, e24, "H24 = H48", color=GREY, va="bottom", fontsize=6)
else:
    ax[1].text(24.5, e24, "H24", color=GREY, va="center", fontsize=6)
    ax[1].text(24.5, e48, "H48", color=BLUE, va="center", fontsize=6)
save(fig, "python_figure2_representative_day", aligned=True)

(OUT / "python_figure_notes.json").write_text(json.dumps({
    "claim": "Across the 10-scenario rolling experiment, H48 has lower cumulative cost than H24 while maintaining a distinct storage trajectory.",
    "data": ["问题3_每日指标.csv", "问题3_每日配对差.csv", "问题3_逐时段结果.csv"],
    "uncertainty": "No replicate-level uncertainty is plotted; the paired daily difference is shown directly and the previously computed block-bootstrap interval is reported in the result JSON.",
    "representative_day": "2025-06-21", "backend": "Python/matplotlib"
}, ensure_ascii=False, indent=2), encoding="utf-8")
