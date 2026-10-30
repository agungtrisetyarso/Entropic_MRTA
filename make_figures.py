"""Regenerate Figures f1, f6 and f9 of the revised manuscript.

Python re-implementation of Algorithm 2 (mirror step + dual damping) with
Algorithm 1 (M Sinkhorn sweeps, column sums either exact or by T rounds of
Metropolis averaging on a random geometric graph), on instances drawn from the
same generator as src/bench.cpp (uniform base cost 5-40, 20% capability
penalty 50+U(0,30), additive 0.3*U(0,30)). Seeds differ from the C++ runs.

Usage: python3 make_figures.py OUTDIR
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"f1", "f6", "f9"}
os.makedirs(OUT, exist_ok=True)

PAPER = dict(alpha=0.08, beta=0.35, lam0=0.5, laminf=0.02, Kf=200)


def gen_cost(n, rng):
    C = rng.uniform(5, 40, (n, n))
    C += np.where(rng.uniform(size=(n, n)) < 0.2, 50 + rng.uniform(0, 30, (n, n)), 0.0)
    C += 0.3 * rng.uniform(0, 30, (n, n))
    return C


def gen_graph(n, radius, rng):
    while True:
        xy = rng.uniform(size=(n, 2))
        D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        A = (D <= radius) & ~np.eye(n, dtype=bool)
        reach = np.zeros(n, bool); reach[0] = True
        for _ in range(n):
            reach |= A[reach].any(0)
        if reach.all():
            return A


def metropolis(A):
    deg = A.sum(1); n = len(deg)
    W = np.where(A, 1.0 / (1.0 + np.maximum(deg[:, None], deg[None, :])), 0.0)
    W[np.arange(n), np.arange(n)] = 1.0 - W.sum(1)
    return W


def lam(k):
    return PAPER["laminf"] if k >= PAPER["Kf"] else max(PAPER["laminf"], PAPER["lam0"] / np.log(k + 2.0))


def ent(P):
    Q = np.maximum(P, 1e-300)
    return -(P * np.log(Q)).sum()


def round_greedy(P):
    n, d = P.shape
    a = -np.ones(n, int); ur = np.zeros(n, bool); uc = np.zeros(d, bool)
    for idx in np.argsort(-P, axis=None):
        i, j = divmod(int(idx), d)
        if not ur[i] and not uc[j]:
            a[i] = j; ur[i] = uc[j] = True
    return a


def run(C, A=None, M=15, T=5, iters=500, p_loss=0.0, rng=None, events=None, noise=0.0):
    """Returns per-iteration W_k, row residual, column residual, mean purity,
    rounded-assignment cost, and the (possibly changed) cost matrices."""
    C = C.copy(); n = C.shape[0]
    P = np.full((n, n), 1.0 / n)
    alive = np.arange(n)
    out = dict(W=[], rres=[], cres=[], pur=[], cost=[], opt=[], ev=[])
    for k in range(iters):
        if events and k in events:
            kind = events[k]
            a = round_greedy(P)
            if kind == "dropout":          # robot 0 fails; the task it held is withdrawn
                i, j = 0, a[0]
                keep_r = np.delete(np.arange(P.shape[0]), i); keep_c = np.delete(np.arange(P.shape[1]), j)
                P = P[np.ix_(keep_r, keep_c)]; C = C[np.ix_(keep_r, keep_c)]
                P /= P.sum(1, keepdims=True)
                if A is not None:
                    A = A[np.ix_(keep_r, keep_r)]
            elif kind == "replace":        # one task cancelled, a new one arrives
                j = a[0]
                C[:, j] = gen_cost(C.shape[0], rng)[:, 0]
            out["ev"].append(k)
        n = P.shape[0]
        Ck = C * (1 + noise * rng.standard_normal(C.shape)) if noise > 0 else C
        lk = lam(k)
        W = (Ck * P).sum() + lk * ent(P)
        logP = np.log(np.maximum(P, 1e-300))
        t = (1 + PAPER["alpha"] * lk) * logP - PAPER["alpha"] * Ck
        t -= t.max(1, keepdims=True); Pt = np.exp(t); Pt /= Pt.sum(1, keepdims=True)
        h = (1 - PAPER["beta"]) * np.log(np.maximum(Pt, 1e-300)) + PAPER["beta"] * logP
        h -= h.max(1, keepdims=True); Q = np.exp(h); Q /= Q.sum(1, keepdims=True)
        for _ in range(M):
            Q /= Q.sum(1, keepdims=True)
            if A is None:
                Z = np.tile(Q.mean(0), (n, 1))
            else:
                Z = Q.copy()
                for _ in range(T):
                    Al = A.copy()
                    if p_loss > 0:
                        up = rng.uniform(size=A.shape) >= p_loss
                        up = np.triu(up, 1); up = up | up.T
                        Al &= up
                    Z = metropolis(Al) @ Z
            Q = Q / (n * np.maximum(Z, 1e-300))
        P = Q
        a = round_greedy(P)
        out["W"].append(W)
        out["rres"].append(np.abs(P.sum(1) - 1).max()); out["cres"].append(np.abs(P.sum(0) - 1).max())
        out["pur"].append(float((P ** 2).sum(1).mean()))
        out["cost"].append(C[np.arange(n), a].sum())
        r, c = linear_sum_assignment(C); out["opt"].append(C[r, c].sum())
    return {k: np.array(v) for k, v in out.items()}


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"exact": "#2a6fdb", "m100": "#7a4fd1", "cons": "#d1495b"}

# ---------------------------------------------------------------- f1
f1_stats = f6_stats = f9_stats = None
if "f1" in ONLY:
  rng = np.random.default_rng(11)
  C = gen_cost(10, rng); A = gen_graph(10, 0.45, rng)
  r_exact = run(C, None, M=15)
  r_m100 = run(C, None, M=100)
  r_cons = run(C, A, M=5, T=50, rng=np.random.default_rng(1))
  fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
  for r, lab, c in [(r_exact, "exact column sums, $M{=}15$", COL["exact"]),
                    (r_m100, "exact column sums, $M{=}100$", COL["m100"]),
                    (r_cons, "Alg. 1, $M{=}5$, $T{=}50$, sparse graph", COL["cons"])]:
      ax[0].plot(r["W"], color=c, lw=1.2, label=lab)
      ax[1].semilogy(np.maximum(r["rres"], r["cres"]), color=c, lw=1.2)
  ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel("$W_k$"); ax[0].legend(frameon=False, fontsize=7)
  ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel("marginal residual")
  fig.tight_layout(); fig.savefig(os.path.join(OUT, "f1.pdf")); plt.close(fig)
  f1_stats = dict(
      max_rel_increase={k: float(np.max(np.diff(r["W"]) / np.abs(r["W"][:-1]))) for k, r in
                        [("exact15", r_exact), ("exact100", r_m100), ("cons", r_cons)]},
      final_residual={k: float(max(r["rres"][-1], r["cres"][-1])) for k, r in
                      [("exact15", r_exact), ("exact100", r_m100), ("cons", r_cons)]})

# ---------------------------------------------------------------- f6
if "f6" in ONLY:
    R = 50
    res = {"exact": [], "cons": []}
    for s in range(R):
        rng = np.random.default_rng(100 + s)
        C = gen_cost(10, rng); A = gen_graph(10, 0.45, rng)
        res["exact"].append(run(C, None, M=15))
        res["cons"].append(run(C, A, M=5, T=50, rng=np.random.default_rng(500 + s)))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
    f6_stats = {}
    for key, lab in [("exact", "exact column sums, $M{=}15$"), ("cons", "Alg. 1, $M{=}5$, $T{=}50$")]:
        W = np.array([r["W"] for r in res[key]]); pur = np.array([r["pur"] for r in res[key]])
        gap = np.array([100 * (r["cost"] / r["opt"] - 1) for r in res[key]])
        x = np.arange(W.shape[1])
        for a_, Y in zip(ax, (W, pur, gap)):
            m = Y.mean(0); lo, hi = np.percentile(Y, [10, 90], axis=0)
            a_.plot(x, m, color=COL[key], lw=1.2, label=lab); a_.fill_between(x, lo, hi, color=COL[key], alpha=0.18, lw=0)
        inc = np.diff(W, axis=1) / np.abs(W[:, :-1])
        f6_stats[key] = dict(runs_with_increase=int((inc > 1e-12).any(1).sum()), max_rel_increase=float(inc.max()),
                             final_gap_mean=float(gap[:, -1].mean()), optimal=int((gap[:, -1] < 1e-6).sum()),
                             final_purity=float(pur[:, -1].mean()))
    ax[0].set_ylabel("$W_k$"); ax[1].set_ylabel("mean purity"); ax[2].set_ylabel("gap of rounded assignment (%)")
    for a_ in ax: a_.set_xlabel("iteration $k$")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "f6.pdf")); plt.close(fig)

# ---------------------------------------------------------------- f9
if "f9" in ONLY:
    R9 = 20
    classes = [("none", {}), ("packet loss 20%", dict(p_loss=0.2)), ("packet loss 40%", dict(p_loss=0.4)),
               ("cost noise $\\sigma{=}0.15$", dict(noise=0.15)),
               ("robot dropout at $k{=}150$", dict(events={150: "dropout"})),
               ("task replaced at $k{=}150$", dict(events={150: "replace"}))]
    cols = ["#444444", "#d1495b", "#edae49", "#00798c", "#30638e", "#7a4fd1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    f9_stats = {}; cache = {}
    for (lab, kw), c in zip(classes, cols):
        Ws, gaps = [], []
        for s in range(R9):
            rng = np.random.default_rng(900 + s)
            C = gen_cost(10, rng); A = gen_graph(10, 0.45, rng)
            r = run(C, A, M=5, T=50, rng=np.random.default_rng(1900 + s), **kw)
            Ws.append(r["W"]); gaps.append(100 * (r["cost"][-1] / r["opt"][-1] - 1))
        W = np.array(Ws); cache[lab] = W
        for ax_ in axs:
            ax_.plot(W.mean(0), color=c, lw=1.3, label=lab)
        inc = np.diff(W, axis=1) / np.abs(W[:, :-1])
        if "events" in kw:
            inc[:, 149] = 0  # the structural jump itself is excluded
        f9_stats[lab] = dict(max_rel_increase=float(inc.max()), final_gap_mean=float(np.mean(gaps)),
                             optimal=int(np.sum(np.array(gaps) < 1e-6)))
    for ax_ in axs:
        ax_.axvline(150, color="#999999", lw=0.8, ls=":"); ax_.set_xlabel("iteration $k$")
    axs[0].set_ylabel("$W_k$ (mean over 20 runs)")
    lo = min(v.mean(0)[100:].min() for v in cache.values()); hi = max(v.mean(0)[100:].max() for v in cache.values())
    axs[1].set_xlim(100, 500); axs[1].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    axs[1].set_title("zoom: iterations 100-500", fontsize=8)
    axs[1].legend(frameon=False, fontsize=7, loc="upper right")
    np.savez(os.path.join(OUT, "f9_curves.npz"), **{k.replace("$", "").replace("\\", ""): v for k, v in cache.items()})
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "f9.pdf")); plt.close(fig)

import json
stats_path = os.path.join(OUT, "figure_stats.json")
old = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
for k, v in dict(f1=f1_stats, f6=f6_stats, f9=f9_stats).items():
    if v is not None:
        old[k] = v
json.dump(old, open(stats_path, "w"), indent=1)
print(json.dumps(dict(f1=f1_stats, f6=f6_stats, f9=f9_stats), indent=1))
