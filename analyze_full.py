"""Analysis of the full study -> results/full_stats.txt, paper/tab_*.tex, paper/fig_*.pdf"""
import glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, studentized_range
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/full/*_s*.csv"))], ignore_index=True)
FAM = ["tsk_prod", "tsk_min", "moe_lin"]
FNAME = {"tsk_prod": "TSK-prod", "tsk_min": "TSK-min", "moe_lin": "MoE-lin"}
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(1, run)
    return adj


G = M[M.model.isin(FAM)].copy()
nd = G.dataset.nunique()
P(f"datasets: {nd}, seeds: {sorted(G.seed.unique())}, gated runs: {len(G)}")
P("non-finite runs:", int((~G.finite.astype(bool)).sum()), "| MC flips on certified points (soundness):", int(G["mc_flip_on_cert_0.02"].sum()))
P("min rules used:", int(G.rules_used.min()), "| runs with rules_used < R:", int((G.rules_used < G.R).sum()))
P("selected R counts:", G[G.lam == 0].groupby("model").R.value_counts().to_dict())

# dataset-level aggregation (mean over seeds x folds)
A = G.groupby(["dataset", "model", "lam"]).mean(numeric_only=True).reset_index()

# ---- Table: certifiers ----
lines = [r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
         r"& & \multicolumn{3}{c}{$\varepsilon=0.01$} & \multicolumn{3}{c}{$\varepsilon=0.02$} & \multicolumn{3}{c}{$\varepsilon=0.05$}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Gate & $\lambda$ & Exact & IBP & Lip. & Exact & IBP & Lip. & Exact & IBP & Lip.\\", r"\midrule"]
for fam in FAM:
    for lam in [0.0, 0.3]:
        s = A[(A.model == fam) & (A.lam == lam)]
        cells = []
        for e in [0.01, 0.02, 0.05]:
            for m in ["exact", "ibp", "lip"]:
                c = f"cert_{m}_{e}"
                cells.append("--" if s[c].isna().all() else f"{100*s[c].mean():.1f}")
        lines.append(f"{FNAME[fam]} & {lam:g} & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_cert.tex", "w").write("\n".join(lines))

P("\n== exact vs baselines (dataset-level, all families and lambdas pooled per eps)")
ps, nm = [], []
for e in [0.01, 0.02, 0.05]:
    for base in ["ibp", "lip"]:
        sub = A.dropna(subset=[f"cert_{base}_{e}"])
        dlt = sub[f"cert_exact_{e}"] - sub[f"cert_{base}_{e}"]
        p = wilcoxon(sub[f"cert_exact_{e}"], sub[f"cert_{base}_{e}"], alternative="greater", zero_method="zsplit").pvalue
        ps.append(p); nm.append(f"exact>{base} eps={e}: n={len(sub)} mean gain {100*dlt.mean():.2f} pp, median {100*dlt.median():.2f}, min {100*dlt.min():.2f}, max {100*dlt.max():.2f}")
for a, p, h in zip(nm, ps, holm(ps)):
    P(f"  {a}; p={p:.2e}, Holm p={h:.2e}")
P("  per family, eps=0.02, lam=0:")
for fam in FAM:
    s = A[(A.model == fam) & (A.lam == 0)]
    for base in ["ibp", "lip"]:
        if s[f"cert_{base}_0.02"].isna().all(): continue
        d = s["cert_exact_0.02"] - s[f"cert_{base}_0.02"]
        P(f"    {fam} exact-{base}: mean {100*d.mean():.2f} pp, max {100*d.max():.2f} pp")

# ---- sampling / MC ----
P("\n== Monte-Carlo sampling check (100 uniform draws per point, eps=0.02)")
for fam in FAM:
    s = G[G.model == fam]
    P(f"  {fam}: MC-stable {100*s['mc_stable_0.02'].mean():.2f}% vs exact {100*s['cert_exact_0.02'].mean():.2f}%; "
      f"false stability claims {100*s['mc_false_claim_0.02'].mean():.2f}% of test points "
      f"(max per run {100*s['mc_false_claim_0.02'].max():.2f}%)")

# ---- runtime ----
P("\n== certification time per test fold (s), eps=0.02: median [max]")
for fam in FAM:
    s = G[G.model == fam]
    P("  " + fam + ": " + ", ".join(f"{m} {s[f't_{m}_0.02'].median():.4f} [{s[f't_{m}_0.02'].max():.3f}]"
                                   for m in ["exact", "ibp", "lip"] if not s[f"t_{m}_0.02"].isna().all()))

# ---- stability training ----
P("\n== stability training vs lam=0 (dataset-level, Wilcoxon, Holm within family)")
lam_rows = []
for fam in FAM:
    a0 = A[(A.model == fam) & (A.lam == 0)].set_index("dataset")
    ps, nm = [], []
    for lam in [0.1, 0.3, 1.0]:
        a1 = A[(A.model == fam) & (A.lam == lam)].set_index("dataset")
        for col, alt in [("cert_exact_0.02", "greater"), ("acc", "two-sided"), ("radius_median", "greater")]:
            d = a1[col] - a0[col]
            p = wilcoxon(a1[col], a0[col], alternative=alt, zero_method="zsplit").pvalue
            ps.append(p); nm.append((lam, col, d.mean(), d.median()))
    for (lam, col, mu, md), p, h in zip(nm, ps, holm(ps)):
        P(f"  {fam} lam={lam} {col}: mean {mu:+.4f} median {md:+.4f}; p={p:.2e} Holm={h:.2e}")
    for lam in [0.0, 0.1, 0.3, 1.0]:
        s = A[(A.model == fam) & (A.lam == lam)]
        lam_rows.append(dict(model=fam, lam=lam, acc=s.acc.mean(), cert=s["cert_exact_0.02"].mean(),
                             radius=s.radius_median.mean(), rules=s.rules_used.mean()))
L = pd.DataFrame(lam_rows); P(L.round(4).to_string(index=False))

# ---- accuracy ----
M2 = M.copy()
M2["name"] = np.where(M2.model.isin(FAM), M2.model.map(FNAME) + " ($\\lambda$=" + M2.lam.astype(str).str.replace(".0", "", regex=False) + ")", M2.model)
keep = M2.model.isin(["LR", "RF", "HGB", "MLP"]) | (M2.lam.isin([0.0, 0.3]))
T = M2[keep].groupby(["dataset", "name"]).acc.mean().unstack()
ranks = T.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
fr = friedmanchisquare(*[T[c] for c in T.columns]).pvalue
k, N = T.shape[1], T.shape[0]
q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * N))
P(f"\n== accuracy: Friedman p={fr:.2e}; Nemenyi CD (alpha=0.05) = {CD:.3f}")
order = ranks.mean().sort_values()
for nmn in order.index:
    P(f"  {nmn:28s} mean acc {T[nmn].mean():.4f}  mean rank {order[nmn]:.2f}")
ps, nm = [], []
for g in [c for c in T.columns if "$" in c]:
    for b in ["LR", "RF", "HGB", "MLP"]:
        p = wilcoxon(T[g], T[b], zero_method="zsplit").pvalue; ps.append(p); nm.append(f"{g} vs {b}: diff {T[g].sub(T[b]).mean():+.4f}")
for a, p, h in zip(nm, ps, holm(ps)):
    P(f"  {a}; p={p:.3f}, Holm={h:.3f}")
lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Model & Mean accuracy & Mean rank\\", r"\midrule"]
for nmn in order.index:
    lines.append(f"{nmn} & {T[nmn].mean():.3f} & {order[nmn]:.2f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_acc.tex", "w").write("\n".join(lines))

# CD diagram
fig, ax = plt.subplots(figsize=(6.5, 2.2))
lo, hi = 1, k
ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(0, 1); ax.axis("off")
ax.hlines(0.8, lo, hi, color="k", lw=0.8)
for i in range(lo, hi + 1):
    ax.vlines(i, 0.78, 0.82, color="k", lw=0.8); ax.text(i, 0.86, str(i), ha="center", fontsize=8)
for j, (nmn, rk) in enumerate(order.items()):
    yy = 0.65 - 0.07 * (j % 9)
    ax.plot([rk, rk], [0.8, yy], color="0.4", lw=0.6)
    ax.text(rk + (0.05 if j % 2 else -0.05), yy - 0.02, nmn.replace("$\\lambda$", "λ"), fontsize=6.5,
            ha="left" if j % 2 else "right", va="top")
ax.hlines(0.95, lo, lo + CD, color="k", lw=1.5); ax.text(lo + CD / 2, 0.97, f"CD = {CD:.2f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig("paper/fig_cd.pdf"); plt.close(fig)

# ---- lambda trade-off figure ----
fig, ax = plt.subplots(1, 3, figsize=(9, 2.6), sharey=False)
for i, fam in enumerate(FAM):
    s = L[L.model == fam]
    ax[i].plot(100 * s.cert, 100 * s.acc, "o-", color="k")
    for _, r in s.iterrows():
        ax[i].annotate(f"λ={r.lam:g}", (100 * r.cert, 100 * r.acc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    base = 100 * s.acc.iloc[0]
    ax[i].set_ylim(base - 2, base + 2); ax[i].set_xlim(65, 100)
    ax[i].set_title(FNAME[fam], fontsize=9); ax[i].set_xlabel("certified at ε=0.02 (%)", fontsize=8)
    ax[i].tick_params(labelsize=7)
ax[0].set_ylabel("test accuracy (%)", fontsize=8)
fig.tight_layout(); fig.savefig("paper/fig_lambda.pdf"); plt.close(fig)

# ---- per-dataset table (lam=0 and 0.3, tsk_prod) ----
meta = {}
for f in glob.glob("results/datasets_meta.json"):
    meta = json.load(open(f))
lines = [r"\begin{tabular}{@{}lrrrcccc@{}}", r"\toprule",
         r"Dataset & $n$ & $d$ & $K$ & HGB acc. & TSK-prod acc. & cert.\ $\lambda{=}0$ & cert.\ $\lambda{=}0.3$\\", r"\midrule"]
hg = M[M.model == "HGB"].groupby("dataset").acc.mean()
for ds in sorted(G.dataset.unique()):
    a0 = A[(A.dataset == ds) & (A.model == "tsk_prod") & (A.lam == 0)].iloc[0]
    a3 = A[(A.dataset == ds) & (A.model == "tsk_prod") & (A.lam == 0.3)].iloc[0]
    m = meta.get(ds, {})
    lines.append(f"{ds.replace('_', ' ')} & {m.get('n', '')} & {m.get('d', '')} & {m.get('K', '')} & {hg[ds]:.3f} & {a0.acc:.3f} & {100*a0['cert_exact_0.02']:.1f} & {100*a3['cert_exact_0.02']:.1f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_datasets.tex", "w").write("\n".join(lines))

open("results/full_stats.txt", "w").write("\n".join(out))
print("\n".join(out))
