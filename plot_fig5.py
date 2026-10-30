"""Figure 5: one Robotarium-simulator mission of the proposed method.
Usage: python3 plot_fig5.py PROPOSED.npz OUT.pdf [AUCTION.npz]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

z = np.load(sys.argv[1], allow_pickle=True)
s = z["summary"][0]
traj, sites = z["traj"], z["sites"]           # (steps+1) x 3 x n, 2 x n
comp = z["completions"]                         # rows: t, robot, task
A, G, S, ep = z["ledger_A"], z["ledger_G"], z["ledger_schedule"], z["ledger_epoch"]
n = traj.shape[2]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))

a = ax[0]
a.add_patch(Rectangle((-1.6, -1.0), 3.2, 2.0, fill=False, lw=0.8, ec="#888888"))
cm = plt.get_cmap("tab10")
for i in range(n):
    a.plot(traj[:, 0, i], traj[:, 1, i], color=cm(i % 10), lw=0.9)
    a.plot(traj[0, 0, i], traj[0, 1, i], "o", ms=3.5, color=cm(i % 10))
a.plot(sites[0], sites[1], "s", ms=6, mfc="none", mec="#222222", mew=0.9, label="task site")
a.set_aspect("equal"); a.set_xlim(-1.7, 1.7); a.set_ylim(-1.1, 1.1)
a.set_xlabel("x (m)"); a.set_ylabel("y (m)")


b = ax[1]
dt = 0.033
other = np.load(sys.argv[3], allow_pickle=True) if len(sys.argv) > 3 else None
for zz, lab, c, off in [(z, "proposed, accurate", "#d1495b", 0.0), (other, "distributed auction", "#2a6fdb", -0.6)]:
    if zz is None:
        continue
    cc = zz["completions"]; ss = zz["summary"][0]
    tt = np.concatenate([[0.0], np.sort(cc[:, 0]), [ss["mission_time"]]])
    rem = np.concatenate([[n], n - np.arange(1, len(cc) + 1), [n - len(cc)]])
    b.step(tt, rem, where="post", color=c, lw=1.3, label=lab + r" ($\Pi_{\mathrm{safe}}$ = " + f"{ss['pos_safe']:.2f} s)")
    act = zz["barrier_active"]; ts = np.arange(len(act)) * dt
    b.fill_between(ts, off - 0.25, off + 0.25, where=act, color=c, alpha=0.35, lw=0, step="post")
b.set_xlabel("time (s)"); b.set_ylabel("tasks remaining")
b.set_ylim(-1.2, n + 0.5)
b.text(0.98, 0.02, "bands: barrier active", transform=b.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
b.legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(sys.argv[2])
print("proposed", s["tasks_done"], s["mission_time"], s["pos_safe"], "auction",
      other["summary"][0]["tasks_done"] if other is not None else None,
      other["summary"][0]["mission_time"] if other is not None else None,
      other["summary"][0]["pos_safe"] if other is not None else None)
