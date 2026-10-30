"""Emit LaTeX rows for the Robotarium-simulator dry-run table from mission
.npz files.  Usage: python3 make_table.py DIR_n6 DIR_n10 P_LOSS"""
import sys, glob, os
import numpy as np

NAMES = {"greedy": "Greedy", "cbaa": "CBAA", "auction": "Distributed auction",
         "proposed_fast": "Proposed, fast", "proposed_accurate": "Proposed, accurate"}
ORDER = list(NAMES)


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "mrta_*_s*_n*.npz"))):
        z = np.load(f, allow_pickle=True)
        s = z["summary"][0]
        A = z["ledger_A"]
        s["frac_A_pos"] = float((A > 1e-9).mean()) if A.size else np.nan
        out.append(s)
    return out


def ms(v, fmt):
    v = np.asarray(v, float)
    return (fmt % v.mean()) + r"$\pm$" + (fmt % (v.std(ddof=1) if v.size > 1 else 0.0))


def med(v, fmt):
    return fmt % np.median(np.asarray(v, float))


p = float(sys.argv[3])
for d in sys.argv[1:3]:
    rows = load(d)
    if not rows:
        continue
    n = rows[0]["n"]
    R = len({s["seed"] for s in rows})
    first = True
    for m in ORDER:
        S = [s for s in rows if s["method"] == m and abs(s["p_loss"] - p) < 1e-9]
        if not S:
            continue
        done = ms([s["tasks_done"] for s in S], "%.1f")
        tm = ms([s["mission_time"] for s in S], "%.1f")
        cfv = np.array([s["first_cf"] for s in S], float)
        cf = ("%.2f" % np.nanmedian(cfv)) + ("" if np.isfinite(cfv).all() else "$^{a}$")
        kb = med([s["total_kB"] for s in S], "%.0f")
        msg = med([s["alloc_msgs"] for s in S], "%.0f")
        ps = ms([s["pos_safe"] for s in S], "%.2f")
        if m.startswith("proposed"):
            ak = "%.2f (%.0f\\%%)" % (np.median([s["max_A"] for s in S]),
                                      100 * np.median([s["frac_A_pos"] for s in S]))
            h3 = "%d/%d" % (sum(s["h3"]["alloc_msgs"] <= s["h3"]["budget"] for s in S), len(S))
        else:
            ak, h3 = "---", "---"
        lead = (r"\multirow{5}{*}{%d}" % n) if first else ""
        first = False
        print(f"{lead} & {NAMES[m]} & {done} & {tm} & {cf} & {kb} & {msg} & {ps} & {ak} & {h3} \\\\")
    print(r"\midrule" if d == sys.argv[1] else "", f"% n={n}, seeds={R}")
