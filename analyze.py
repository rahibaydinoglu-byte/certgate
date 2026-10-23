"""Statistics for the pilot: Wilcoxon signed-rank (Holm) and Friedman tests over datasets."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

M = pd.read_csv("results/pilot_main.csv")
R = pd.read_csv("results/pilot_R.csv")
S = pd.read_csv("results/pilot_shap.csv")
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(1, run)
    return adj


G = M[M.model.isin(["tsk_prod", "tsk_min", "moe_lin"])]
P("== sanity: soundness violations / attack-missed (must be 0 / 0):",
  int(G["sound_viol_0.02"].sum()), "/", int(G["attack_missed_0.02"].sum()),
  "| non-finite runs:", int((~G.finite.astype(bool)).sum()))
P("== exact certified fraction vs attack-stable at eps=0.02: max |diff| =",
  float((G["cert_exact_0.02"] - G["attack_stable_0.02"]).abs().max()))

# 1) certifier comparison (per dataset x family x lam, mean over folds)
A = G.groupby(["dataset", "model", "lam"]).mean(numeric_only=True).reset_index()
P("\n== Certified fraction (mean over datasets), by family, lam, eps")
for eps in [0.01, 0.02, 0.05]:
    cols = [f"cert_exact_{eps}", f"cert_ibp_{eps}", f"cert_lip_{eps}"]
    P(f"eps={eps}"); P(A.groupby(["model", "lam"])[cols].mean().round(3).to_string())
tests, names = [], []
for eps in [0.01, 0.02, 0.05]:
    for base in ["ibp", "lip"]:
        sub = A.dropna(subset=[f"cert_{base}_{eps}"])
        dlt = sub[f"cert_exact_{eps}"] - sub[f"cert_{base}_{eps}"]
        p = wilcoxon(sub[f"cert_exact_{eps}"], sub[f"cert_{base}_{eps}"], alternative="greater", zero_method="zsplit").pvalue
        tests.append(p); names.append(f"exact>{base} eps={eps} (n={len(sub)}, mean gain {dlt.mean():.3f}, min gain {dlt.min():.3f})")
for nme, p, ph in zip(names, tests, holm(tests)):
    P(f"Wilcoxon {nme}: p={p:.2e}, Holm p={ph:.2e}")

# runtime
P("\n== Median certification time per test fold (s), eps=0.02")
P(A.groupby("model")[["t_exact_0.02", "t_ibp_0.02", "t_lip_0.02"]].median().round(4).to_string())

# 2) effect of the stability penalty (lam 0 vs 0.3)
P("\n== Stability training: lam=0 vs lam=0.3 (paired over datasets)")
ps, nm = [], []
for fam in ["tsk_prod", "tsk_min", "moe_lin"]:
    a0 = A[(A.model == fam) & (A.lam == 0)].set_index("dataset")
    a1 = A[(A.model == fam) & (A.lam == 0.3)].set_index("dataset")
    for col in ["cert_exact_0.02", "acc"]:
        d = (a1[col] - a0[col])
        alt = "greater" if col.startswith("cert") else "two-sided"
        p = wilcoxon(a1[col], a0[col], alternative=alt, zero_method="zsplit").pvalue
        ps.append(p); nm.append(f"{fam} {col}: mean change {d.mean():+.3f} (median {d.median():+.3f})")
    P(f"{fam}: rules used (mean) lam0={a0.rules_used.mean():.2f} lam0.3={a1.rules_used.mean():.2f}")
for nme, p, ph in zip(nm, ps, holm(ps)):
    P(f"Wilcoxon {nme}: p={p:.2e}, Holm p={ph:.2e}")

# 3) accuracy: Friedman over models
P("\n== Accuracy (mean over folds), Friedman over datasets")
M2 = M.copy(); M2["name"] = M2.model + np.where(M2.lam.isna(), "", "_lam" + M2.lam.astype(str))
T = M2.groupby(["dataset", "name"]).acc.mean().unstack()
P("mean acc:", T.mean().round(3).sort_values(ascending=False).to_dict())
ranks = T.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
P("mean rank (1=best):", ranks.mean().round(2).sort_values().to_dict())
P("Friedman p =", friedmanchisquare(*[T[c] for c in T.columns]).pvalue)
for fam in ["tsk_prod_lam0.0", "tsk_prod_lam0.3", "moe_lin_lam0.0", "tsk_min_lam0.0"]:
    for bb in ["LR", "HGB", "RF", "MLP"]:
        p = wilcoxon(T[fam], T[bb], zero_method="zsplit").pvalue
        P(f"  Wilcoxon {fam} vs {bb}: mean diff {np.mean(T[fam]-T[bb]):+.3f}, p={p:.3f}")

# 4) Theorem 6 on real data
P("\n== Certified fraction vs number of rules (tsk_prod, lam=0, eps=0.02, fold 0)")
Rt = R.pivot(index="dataset", columns="R", values="cert_exact_0_02")
P(Rt.round(3).to_string()); P("mean:", Rt.mean().round(3).to_dict())
P("datasets where cert is non-increasing in R:", int((Rt.diff(axis=1).iloc[:, 1:] <= 1e-9).all(axis=1).sum()), "/", len(Rt))

# 5) post-hoc SHAP explanation stability (empirical upper bound) vs certified routing
P("\n== HGB+SHAP top-1 feature, empirically stable fraction under 30 random perturbations (eps=0.02, fold 0)")
P(S[["dataset", "emp_stable"]].round(3).to_string(index=False)); P("mean:", round(S.emp_stable.mean(), 3))
open("results/pilot_stats.txt", "w").write("\n".join(out))
print("\n".join(out))
