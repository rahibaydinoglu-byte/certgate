"""Ablations (seed 0, 5 folds, R = 4) on six datasets.
A1 load balancing: beta in {0, 0.1} at lambda = 1.0
A2 penalty form: scale-invariant ratio vs raw margin hinge (gamma = 1), lambda = 0.3; logit-scale diagnostic
A3 training radius: eps_train in {0.01, 0.02, 0.05}, lambda = 0.3, evaluated at eps_test in {0.01, 0.02, 0.05}
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, pandas as pd, jax.numpy as jnp
from certgate import data, train, certify
from certgate.models import logits_gate

DS = ["pima", "phoneme", "winequality_red", "banknote", "wdbc", "ionosphere"]
FAMS = ["tsk_prod", "moe_lin"]
rows = []


def logit_scale(p, fam, X):
    g = np.sort(np.asarray(logits_gate(p, fam, jnp.asarray(X))), -1)
    return float(np.median(g[:, -1] - g[:, -2]))


def record(base, p, fam, Xte, yte, **kw):
    r = {**base, **kw, "acc": train.accuracy(p, fam, Xte, yte), "rules_used": train.effective_rules(p, fam, Xte)[0],
         "logit_gap_median": logit_scale(p, fam, Xte), "finite": train.is_finite(p)}
    for e in [0.01, 0.02, 0.05]:
        r[f"cert_{e}"] = certify.certify(p, fam, Xte, e)[0].mean()
    rows.append(r)


for name in DS:
    X, y, meta = data.load(name)
    for fold, (tr, te) in enumerate(data.splits(y)):
        Xtr, Xte = data.minmax(X[tr], X[te]); ytr, yte = y[tr], y[te]
        for fam in FAMS:
            base = dict(dataset=name, fold=fold, model=fam)
            fit = lambda **kw: train.fit(fam, Xtr, ytr, meta["K"], R=4, lr=0.01, steps=1500, **kw)
            record(base, fit(lam=0.0), fam, Xte, yte, ablation="reference", setting="lam=0")
            for beta in [0.0, 0.1]:
                record(base, fit(lam=1.0, beta=beta), fam, Xte, yte, ablation="A1_balance", setting=f"beta={beta}")
            record(base, fit(lam=0.3), fam, Xte, yte, ablation="A2_penalty", setting="ratio")
            record(base, fit(lam=0.3, penalty="raw", gamma=1.0), fam, Xte, yte, ablation="A2_penalty", setting="raw")
            for et in [0.01, 0.02, 0.05]:
                record(base, fit(lam=0.3, eps=et), fam, Xte, yte, ablation="A3_eps_train", setting=f"eps_train={et}")
        print(name, fold, flush=True)
    pd.DataFrame(rows).to_csv("results/ablation.csv", index=False)
print("ABLATION DONE")
