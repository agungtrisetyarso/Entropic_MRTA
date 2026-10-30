"""Summarize mission .npz files: python3 summarize_runs.py DIR"""
import sys, glob, os
import numpy as np

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "mrta_*_s*_n*.npz"))):
    d = np.load(f, allow_pickle=True)
    s = d["summary"][0]
    A = d["ledger_A"]
    s["frac_A_pos"] = float((A > 1e-9).mean()) if A.size else np.nan
    rows.append(s)

if len(sys.argv) > 2 and sys.argv[2] == "each":
    for s in rows:
        print(f"{s['method']:18s} n{s['n']} p{s['p_loss']:.1f} s{s['seed']} done {s['tasks_done']} T {s['mission_time']:5.1f} "
              f"CF {s['first_cf']:5.2f} kB {s['total_kB']:8.1f} msgs {s['alloc_msgs']:5d} pos_mot {s['pos_motion']:7.2f} "
              f"pos_safe {s['pos_safe']:7.3f} (active {s['pos_safe_active']:7.3f}) frac {s['barrier_active_frac']:.2f} "
              f"maxA {s['max_A']:.3g} it {s['n_iters']}")
    sys.exit()

def med(v):
    v = [x for x in v if np.isfinite(x)]
    return np.median(v) if v else np.nan

def mstd(v):
    v = np.array([x for x in v if np.isfinite(x)], float)
    return (v.mean(), v.std(ddof=1) if v.size > 1 else 0.0) if v.size else (np.nan, np.nan)

keys = sorted({(s["n"], s["p_loss"], s["method"]) for s in rows}, key=lambda k: (k[0], k[1], k[2]))
order = ["greedy", "cbaa", "auction", "proposed_fast", "proposed_accurate"]
keys.sort(key=lambda k: (k[0], k[1], order.index(k[2]) if k[2] in order else 99))
print(f"{'n':>3} {'p':>4} {'method':18s} {'runs':>4} {'done':>9} {'time s':>12} {'1st CF s':>8} {'total kB':>9} "
      f"{'alloc msgs':>10} {'Pi_safe s':>13} {'active':>6} {'max A_k':>8} {'A>0':>5} {'H3 ok':>5}")
for k in keys:
    R = [s for s in rows if (s["n"], s["p_loss"], s["method"]) == k]
    done = mstd([s["tasks_done"] for s in R]); tm = mstd([s["mission_time"] for s in R])
    ps = mstd([s["pos_safe"] for s in R])
    h3 = [s["h3"]["alloc_msgs"] <= s["h3"]["budget"] for s in R if s.get("h3")]
    print(f"{k[0]:>3} {k[1]:>4.1f} {k[2]:18s} {len(R):>4} {done[0]:5.2f}/{k[0]:<3d}{tm[0]:6.1f}+/-{tm[1]:<4.1f} "
          f"{med([s['first_cf'] for s in R]):8.2f} {med([s['total_kB'] for s in R]):9.1f} "
          f"{med([s['alloc_msgs'] for s in R]):10.0f} {ps[0]:6.3f}+/-{ps[1]:<6.3f}"
          f"{np.mean([s['barrier_active_frac'] for s in R]):6.2f} {med([s['max_A'] for s in R]):8.3g} "
          f"{med([s['frac_A_pos'] for s in R]):5.2f} {(str(sum(h3))+'/'+str(len(h3))) if h3 else '-':>5}")
