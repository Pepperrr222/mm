"""Generate the Question 4 cost-comparison figure from full-year outputs."""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_图表" / "问题4"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = Path(r"C:\Users\Lenovo\.codex\skills\nature-figure\scripts")
sys.path.insert(0, str(AUDIT))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

q42 = pd.read_csv(ROOT / "04_结果" / "问题4_2_完整求解" / "问题4_2_两方案汇总.csv")
q43 = pd.read_csv(ROOT / "04_结果" / "问题4_3_完整求解" / "问题4_3_两方案汇总.csv")
for frame in (q42, q43):
    assert set(frame["policy"]) == {"H24", "H48"}

def ordered(frame):
    return frame.set_index("policy").loc[["H24", "H48"]]

q42, q43 = ordered(q42), ordered(q43)
values = np.array([
    q42["total_cost_yuan"].to_numpy(),
    q43["total_cost_yuan"].to_numpy(),
]) / 1e6

fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
x = np.arange(2)
width = 0.34
colors = ["#4C78A8", "#E07B39"]
for i, policy in enumerate(["H24", "H48"]):
    bars = ax.bar(x + (i - 0.5) * width, values[:, i], width,
                  color=colors[i], label=policy)
    for bar, val in zip(bars, values[:, i]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x, ["每日0:00决策", "每6小时再调度"])
ax.set_ylabel("全年总费用/百万元")
ax.set_ylim(15.7, 16.75)
ax.legend(ncol=2, loc="upper right")
ax.set_axisbelow(True)

base = OUT / "result_q4_total_cost_comparison"
require_matplotlib_panel_alignment(
    fig,
    json_out=str(base) + ".alignment.json",
    overlay_svg=str(base) + ".alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)
fig.savefig(str(base) + ".svg", bbox_inches="tight")
fig.savefig(str(base) + ".pdf", bbox_inches="tight")
fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
plt.close(fig)
print(base)
