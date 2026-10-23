"""Revision experiments (seed 0, 5 folds, 18 datasets), one CSV per dataset in results/tuned/, resumable.
(1) Equal-budget random search: every model gets the same number of configurations (BUDGET), each scored on
    a stratified 80/20 split of the training fold; the best configuration is refitted on the whole training fold.
(2) Fixed R = 8 gated models with lambda in {0, 0.3} (default hyperparameters), to remove confounding by R.
usage: python run_tuned.py WORKER_ID N_WORKERS
"""
import os, sys, time, warnings
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from certgate import data, train, certify

warnings.filterwarnings("ignore")
OUT = "results/tuned"; os.makedirs(OUT, exist_ok=True)
BUDGET = 6
ORDER = ["mammography", "winequality_white", "phoneme", "digits", "winequality_red", "banknote", "oil_spill",
         "breast_w", "pima", "wdbc", "ionosphere", "ecoli", "haberman", "new_thyroid", "glass", "wheat_seeds",
         "sonar", "wine"]
wid, nw = int(sys.argv[1]), int(sys.argv[2])
if "ONLY" in os.environ:
    ORDER = os.environ["ONLY"].split(",")


def sample_cfgs(kind, rng):
    out = []
    for _ in range(BUDGET):
        if kind == "LR":
            out.append(dict(C=float(10 ** rng.uniform(-2, 2))))
        elif kind == "RF":
            out.append(dict(max_depth=[None, 4, 8, 16][rng.integers(4)], max_features=["sqrt", 0.5, 1.0][rng.integers(3)],
                            min_samples_leaf=int(rng.choice([1, 2, 5]))))
        elif kind == "HGB":
            out.append(dict(learning_rate=float(10 ** rng.uniform(-2, -0.5)), max_leaf_nodes=int(rng.choice([7, 15, 31, 63])),
                            l2_regularization=float(10 ** rng.uniform(-4, 0))))
        elif kind == "MLP":
            out.append(dict(hidden_layer_sizes=[(32,), (64, 64), (128, 64)][rng.integers(3)], alpha=float(10 ** rng.uniform(-5, -1)),
                            learning_rate_init=float(10 ** rng.uniform(-3.5, -2))))
        else:  # gated
            out.append(dict(R=int(rng.choice([2, 4, 8])), lr=float(rng.choice([0.003, 0.01, 0.03])),
                            wd=float(rng.choice([1e-4, 1e-3, 1e-2])), steps=int(rng.choice([1000, 2000]))))
    return out


def make_bb(kind, cfg, seed):
    if kind == "LR": return LogisticRegression(max_iter=3000, **cfg)
    if kind == "RF": return RandomForestClassifier(300, random_state=seed, n_jobs=1, **cfg)
    if kind == "HGB": return HistGradientBoostingClassifier(random_state=seed, **cfg)
    return MLPClassifier(max_iter=1000, random_state=seed, **cfg)


GATED = {"TSK-prod": ("tsk_prod", 0.0), "TSK-prod-cert": ("tsk_prod", 0.3),
         "TSK-min-cert": ("tsk_min", 0.3), "MoE-lin-cert": ("moe_lin", 0.3)}

for k, name in enumerate(ORDER):
    if k % nw != wid:
        continue
    X, y, meta = data.load(name); seed = 0; t_ds = time.time()
    for fold, (tr, te) in enumerate(data.splits(y, seed=seed)):
        path = f"{OUT}/{name}_f{fold}.csv"
        if os.path.exists(path):
            continue
        rows = []
        Xtr, Xte = data.minmax(X[tr], X[te]); ytr, yte = y[tr], y[te]
        a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
        if len(a) > 3000:   # hyperparameter search on a stratified subsample of at most 3000 rows (large datasets)
            a = train_test_split(a, train_size=3000, stratify=ytr[a], random_state=seed)[0]
        Xa, Xb = data.minmax(X[tr][a], X[tr][b])
        rng = np.random.default_rng(1000 + fold)
        base = dict(dataset=name, fold=fold)
        for kind in ["LR", "RF", "HGB", "MLP"]:
            cfgs = sample_cfgs(kind, rng)
            val = [make_bb(kind, c, seed).fit(Xa, ytr[a]).score(Xb, ytr[b]) for c in cfgs]
            best = cfgs[int(np.argmax(val))]
            rows.append({**base, "exp": "tuned", "model": kind, "acc": make_bb(kind, best, seed).fit(Xtr, ytr).score(Xte, yte),
                         "cfg": str(best)})
        for nm, (fam, lam) in GATED.items():
            cfgs = sample_cfgs("gated", rng)
            val = []
            for c in cfgs:
                p = train.fit(fam, Xa, ytr[a], meta["K"], R=c["R"], lam=lam, eps=0.02, lr=c["lr"], wd=c["wd"],
                              steps=c["steps"], seed=seed)
                val.append(train.accuracy(p, fam, Xb, ytr[b]) if train.is_finite(p) else -1)
            best = cfgs[int(np.argmax(val))]
            p = train.fit(fam, Xtr, ytr, meta["K"], R=best["R"], lam=lam, eps=0.02, lr=best["lr"], wd=best["wd"],
                          steps=best["steps"], seed=seed)
            rows.append({**base, "exp": "tuned", "model": nm, "acc": train.accuracy(p, fam, Xte, yte),
                         "cert_exact_0.02": certify.certify(p, fam, Xte, 0.02)[0].mean(), "cfg": str(best)})
        for fam in ["tsk_prod", "tsk_min", "moe_lin"]:
            for lam in [0.0, 0.3]:
                p = train.fit(fam, Xtr, ytr, meta["K"], R=8, lam=lam, eps=0.02, lr=0.01, steps=1500, seed=seed)
                row = {**base, "exp": "R8", "model": fam, "lam": lam, "acc": train.accuracy(p, fam, Xte, yte),
                       "rules_used": train.effective_rules(p, fam, Xte)[0]}
                for e in [0.01, 0.02, 0.05]:
                    row[f"cert_exact_{e}"] = certify.certify(p, fam, Xte, e)[0].mean()
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{name} fold {fold}: {time.time() - t_ds:.0f}s", flush=True)
print("WORKER DONE", wid)
