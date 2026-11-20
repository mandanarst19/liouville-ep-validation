"""Stage 2 figure from results/swarm_series_*.json:  python3 make_stage2_figure.py  ->  figures/stage2_swarm.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)
DT = 0.002
data = {}
for tag in ("undriven", "driven"):
    r = json.load(open(HERE / "results" / f"swarm_series_{tag}_dt{DT}.json"))
    data[tag] = (np.array(r["series"]), {c: i for i, c in enumerate(r["columns"])})

COL = {"undriven": "#1f77b4", "driven": "#d55e00"}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for tag, (s, ix) in data.items():
    t = s[:, ix["t"]]
    m, se = -s[:, ix["Ssys"]], s[:, ix["Ssys_se"]]
    ax[0].plot(t, m, color=COL[tag], label=tag); ax[0].fill_between(t, m - se, m + se, color=COL[tag], alpha=0.3, lw=0)
    q, qse = s[:, ix["Q"]], s[:, ix["Q_se"]]
    ax[1].plot(t, q, color=COL[tag], label=tag); ax[1].fill_between(t, q - qse, q + qse, color=COL[tag], alpha=0.3, lw=0)
    ax[2].plot(t, q / m, color=COL[tag], label=tag)
ax[0].set_ylabel(r"$-\dot S_{sys}$ (Eq. 32, MC mean $\pm$ SE)")
ax[1].set_ylabel(r"$\dot Q$ (Eqs. 34/35)")
ax[2].set_ylabel(r"$\dot Q\,/\,|\dot S_{sys}|$")
for a in ax:
    a.set_yscale("log"); a.set_xlabel("time"); a.legend()
fig.suptitle("Stage 2: n = 20 particles, N = 200 ensemble, near-critically damped; initial conditions and driver are my assumptions",
             fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "figures" / "stage2_swarm.png", dpi=150)
print("saved figures/stage2_swarm.png")
