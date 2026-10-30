import csv, statistics as st, sys, json
from scipy import stats
out={}
for r in ("0.45","1.5"):
    rows=list(csv.DictReader(open(f"results/bench_n10_r{r}.csv")))
    meth=[]; [meth.append(x["method"]) for x in rows if x["method"] not in meth]
    ref="Proposed, M=100, exact col. sums"
    refrows={x["seed"]:x for x in rows if x["method"]==ref}
    out[r]={}
    print("\nradius",r)
    for m in meth:
        R=[x for x in rows if x["method"]==m]
        f=lambda k:[float(x[k]) for x in R]
        g,it,al,kb=f("gap_pct"),f("iters"),f("alloc_bc"),f("kbytes")
        cv=100*sum(int(x["converged"]) for x in R)/len(R)
        d=dict(conv=cv,gap=(st.mean(g),st.stdev(g)),it=(st.mean(it),st.stdev(it),st.median(it)),
               al=(st.mean(al),st.stdev(al)),kb=(st.mean(kb),st.stdev(kb),st.median(kb)))
        if m!=ref:
            a=g; b=[float(refrows[x["seed"]]["gap_pct"]) for x in R]
            d["p_gap"]=stats.wilcoxon(a,b).pvalue if any(abs(x-y)>1e-9 for x,y in zip(a,b)) else None
            a=kb; b=[float(refrows[x["seed"]]["kbytes"]) for x in R]
            d["p_kb"]=stats.wilcoxon(a,b).pvalue
        out[r][m]=d
        print(f"{m:36s} conv {cv:4.0f} gap {d['gap'][0]:6.2f}±{d['gap'][1]:5.2f} it {d['it'][0]:6.1f}±{d['it'][1]:6.1f} med {d['it'][2]:5.0f} alloc {d['al'][0]:6.1f}±{d['al'][1]:6.1f} kB {d['kb'][0]:9.1f}±{d['kb'][1]:8.1f} med {d['kb'][2]:8.1f} p_gap {d.get('p_gap')} p_kb {d.get('p_kb')}")
json.dump(out,open("results/summary.json","w"),indent=1)
