"""Pilot study (1 seed, 5-fold CV, all 18 datasets).
Outputs results/pilot_main.csv, results/pilot_R.csv, results/pilot_shap.csv
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
import jax.numpy as jnp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from certgate import data, train, certify, attack

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
EPS = [0.01, 0.02, 0.05]
FAMS = ["tsk_prod", "tsk_min", "moe_lin"]
LAMS = [0.0, 0.3]
names = sys.argv[1:] or data.ALL
main_rows, R_rows, shap_rows = [], [], []


def shap_stability(model, Xte, eps, n_pts=150, n_pert=30, seed=0):
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xte), min(n_pts, len(Xte)), replace=False)
    expl = shap.TreeExplainer(model)
    def top1(Z):
        sv = expl.shap_values(Z)
        sv = np.abs(np.asarray(sv))
        if sv.ndim == 3:          # (n, d, K) or (K, n, d)
            sv = sv.sum(-1) if sv.shape[0] == len(Z) else sv.sum(0)
        return sv.argmax(-1)
    base = top1(Xte[idx])
    stable = np.ones(len(idx), bool)
    for _ in range(n_pert):
        Z = np.clip(Xte[idx] + rng.uniform(-eps, eps, Xte[idx].shape), 0, 1)
        stable &= top1(Z) == base
    return stable.mean()


for name in names:
    X, y, meta = data.load(name)
    for fold, (tr, te) in list(enumerate(data.splits(y)))[:int(os.environ.get("MAXFOLDS", 5))]:
        Xtr, Xte = data.minmax(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        # black-box baselines (accuracy)
        bb = {"LR": LogisticRegression(max_iter=3000),
              "RF": RandomForestClassifier(300, random_state=0, n_jobs=2),
              "HGB": HistGradientBoostingClassifier(random_state=0),
              "MLP": MLPClassifier((64, 64), max_iter=1000, random_state=0)}
        for k, mdl in bb.items():
            mdl.fit(Xtr, ytr)
            main_rows.append(dict(dataset=name, fold=fold, model=k, lam=np.nan, acc=mdl.score(Xte, yte)))
        if fold == 0:
            for eps in [0.02]:
                shap_rows.append(dict(dataset=name, model="HGB+SHAP", eps=eps,
                                      emp_stable=shap_stability(bb["HGB"], Xte, eps)))
        # gated models
        for fam in FAMS:
            for lam in LAMS:
                t0 = time.time()
                p = train.fit(fam, Xtr, ytr, meta["K"], R=4, lr=0.01, steps=1500, lam=lam, eps=0.02)
                row = dict(dataset=name, fold=fold, model=fam, lam=lam,
                           acc=train.accuracy(p, fam, Xte, yte), t_train=time.time() - t0,
                           finite=train.is_finite(p), rules_used=train.effective_rules(p, fam, Xte)[0])
                for eps in EPS:
                    for meth in ["exact", "ibp", "lip"]:
                        if meth == "lip" and fam == "tsk_min":
                            continue
                        t1 = time.time()
                        c, _ = certify.certify(p, fam, Xte, eps, meth)
                        row[f"cert_{meth}_{eps}"] = c.mean()
                        row[f"t_{meth}_{eps}"] = time.time() - t1
                    if eps == 0.02:
                        fl = attack.attack_flip(p, fam, Xte, eps, steps=30, restarts=3)
                        ce, _ = certify.certify(p, fam, Xte, eps, "exact")
                        row["attack_stable_0.02"] = 1 - fl.mean()
                        row["sound_viol_0.02"] = int((ce & fl).sum())
                        row["attack_missed_0.02"] = int(((~ce) & (~fl)).sum())
                main_rows.append(row)
        # Theorem 6 on real data: certified fraction vs number of rules (fold 0, tsk_prod, lam=0)
        if fold == 0:
            for R in [2, 4, 8, 16]:
                p = train.fit("tsk_prod", Xtr, ytr, meta["K"], R=R, lr=0.01, steps=1500)
                c, _ = certify.certify(p, "tsk_prod", Xte, 0.02, "exact")
                R_rows.append(dict(dataset=name, R=R, cert_exact_0_02=c.mean(),
                                   acc=train.accuracy(p, "tsk_prod", Xte, yte)))
        print(f"{name} fold {fold} done", flush=True)
    pd.DataFrame(main_rows).to_csv("results/pilot_main.csv", index=False)
    pd.DataFrame(R_rows).to_csv("results/pilot_R.csv", index=False)
    pd.DataFrame(shap_rows).to_csv("results/pilot_shap.csv", index=False)
    json.dump({n: data.load(n)[2] for n in names}, open("results/datasets_meta.json", "w"), indent=1, default=str)
print("ALL DONE")
