import glob, pandas as pd
M = pd.concat([pd.read_csv(f) for f in glob.glob("results/full/*_s*.csv")])
G = M[M.model.isin(["tsk_prod", "tsk_min", "moe_lin"]) & M.lam.isin([0, 0.3])]
S = G.groupby(["dataset", "model", "lam", "seed"])[["cert_exact_0.02", "acc"]].mean().reset_index()
A = S.groupby(["dataset", "model", "lam"]).agg(c=("cert_exact_0.02", "mean"), cs=("cert_exact_0.02", "std"),
                                               a=("acc", "mean"), as_=("acc", "std")).reset_index()
fams = ["tsk_prod", "tsk_min", "moe_lin"]
L = [r"\begin{tabular}{@{}l" + "cc" * 3 + r"@{}}", r"\toprule",
     r"& \multicolumn{2}{c}{TSK-prod} & \multicolumn{2}{c}{TSK-min} & \multicolumn{2}{c}{MoE-lin}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Dataset & $\lambda{=}0$ & $\lambda{=}0.3$ & $\lambda{=}0$ & $\lambda{=}0.3$ & $\lambda{=}0$ & $\lambda{=}0.3$\\", r"\midrule"]
for ds in sorted(A.dataset.unique()):
    cells = []
    for f in fams:
        for lam in [0, 0.3]:
            r = A[(A.dataset == ds) & (A.model == f) & (A.lam == lam)].iloc[0]
            cells.append(f"{100*r.c:.1f}$\\pm${100*r.cs:.1f}")
    L.append(ds.replace("_", " ") + " & " + " & ".join(cells) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_appendix.tex", "w").write("\n".join(L))
print("max seed SD (pp):", round(100 * A.cs.max(), 2), "median:", round(100 * A.cs.median(), 2), "| acc SD median:", round(100 * A.as_.median(), 2))
