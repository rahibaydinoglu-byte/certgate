import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, studentized_range

D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/tuned/*_f*.csv"))], ignore_index=True)
out = []; P = lambda *a: out.append(" ".join(str(x) for x in a))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(1, run)
    return adj


T = D[D.exp == "tuned"]
P("datasets:", T.dataset.nunique())
A = T.groupby(["dataset", "model"]).acc.mean().unstack()
ranks = A.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
k, N = A.shape[1], A.shape[0]
CD = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
P(f"Friedman p={friedmanchisquare(*[A[c] for c in A.columns]).pvalue:.2e}, CD={CD:.2f}")
order = ranks.mean().sort_values()
for m in order.index:
    P(f"  {m:14s} acc {A[m].mean():.4f} rank {order[m]:.2f}")
ps, nm = [], []
for g in ["TSK-prod", "TSK-prod-cert", "TSK-min-cert", "MoE-lin-cert"]:
    for b in ["LR", "RF", "HGB", "MLP"]:
        ps.append(wilcoxon(A[g], A[b], zero_method="zsplit").pvalue); nm.append(f"{g} vs {b}: {100*(A[g]-A[b]).mean():+.2f} pp")
for a, p, h in zip(nm, ps, holm(ps)):
    P(f"  {a}; p={p:.3f}, Holm={h:.3f}")
C = T.dropna(subset=["cert_exact_0.02"]).groupby(["dataset", "model"])["cert_exact_0.02"].mean().unstack()
P("tuned gated certified (mean over datasets):", (100 * C.mean()).round(1).to_dict())
p = wilcoxon(C["TSK-prod-cert"], C["TSK-prod"], alternative="greater", zero_method="zsplit").pvalue
P(f"  TSK-prod-cert vs TSK-prod cert: {100*(C['TSK-prod-cert']-C['TSK-prod']).mean():+.1f} pp, p={p:.2e}")
lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & Mean accuracy & Mean rank & Certified ($\varepsilon{=}0.02$)\\", r"\midrule"]
for m in order.index:
    cc = f"{100*C[m].mean():.1f}\\%" if m in C.columns else "--"
    lines.append(f"{m} & {A[m].mean():.3f} & {order[m]:.2f} & {cc}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_tuned.tex", "w").write("\n".join(lines))

R8 = D[D.exp == "R8"].groupby(["dataset", "model", "lam"]).mean(numeric_only=True).reset_index()
P("\nfixed R=8:")
ps, nm = [], []
for fam in ["tsk_prod", "tsk_min", "moe_lin"]:
    a0 = R8[(R8.model == fam) & (R8.lam == 0)].set_index("dataset"); a3 = R8[(R8.model == fam) & (R8.lam == 0.3)].set_index("dataset")
    P(f"  {fam}: cert0.02 {100*a0['cert_exact_0.02'].mean():.1f} -> {100*a3['cert_exact_0.02'].mean():.1f}; acc {100*a0.acc.mean():.2f} -> {100*a3.acc.mean():.2f}; rules {a0.rules_used.mean():.2f} -> {a3.rules_used.mean():.2f}")
    ps.append(wilcoxon(a3["cert_exact_0.02"], a0["cert_exact_0.02"], alternative="greater", zero_method="zsplit").pvalue); nm.append(f"{fam} cert")
    ps.append(wilcoxon(a3.acc, a0.acc, zero_method="zsplit").pvalue); nm.append(f"{fam} acc")
for a, p, h in zip(nm, ps, holm(ps)):
    P(f"  {a}: p={p:.2e}, Holm={h:.2e}")
open("results/tuned_stats.txt", "w").write("\n".join(out)); print("\n".join(out))
