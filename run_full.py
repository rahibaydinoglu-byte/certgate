"""Full study. One CSV per (dataset, seed) in results/full/, resumable.
usage: python run_full.py WORKER_ID N_WORKERS
"""
import os, sys, time, warnings
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from certgate import data, train, certify, attack

warnings.filterwarnings("ignore")
OUT = "results/full"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
FAMS = ["tsk_prod", "tsk_min", "moe_lin"]
LAMS = [0.0, 0.1, 0.3, 1.0]
RS = [2, 4, 8]
EPS = [0.01, 0.02, 0.05]
FIT = dict(lr=0.01, steps=1500)
# order datasets by cost (largest first) for load balance
ORDER = ["mammography", "winequality_white", "phoneme", "digits", "winequality_red", "banknote", "oil_spill",
         "breast_w", "pima", "wdbc", "ionosphere", "ecoli", "haberman", "new_thyroid", "glass", "wheat_seeds",
         "sonar", "wine"]
ORDER = os.environ["ONLY"].split(",") if "ONLY" in os.environ else ORDER
SEEDS = [int(s) for s in os.environ["SEEDS"].split(",")] if "SEEDS" in os.environ else SEEDS
jobs = [(d, s) for s in SEEDS for d in ORDER]
wid, nw = int(sys.argv[1]), int(sys.argv[2])

for k, (name, seed) in enumerate(jobs):
    if k % nw != wid:
        continue
    path = f"{OUT}/{name}_s{seed}.csv"
    if os.path.exists(path):
        continue
    X, y, meta = data.load(name)
    rows = []
    t_ds = time.time()
    for fold, (tr, te) in enumerate(data.splits(y, seed=seed)):
        Xtr, Xte = data.minmax(X[tr], X[te]); ytr, yte = y[tr], y[te]
        base = dict(dataset=name, seed=seed, fold=fold)
        bb = {"LR": LogisticRegression(max_iter=3000),
              "RF": RandomForestClassifier(300, random_state=seed, n_jobs=1),
              "HGB": HistGradientBoostingClassifier(random_state=seed),
              "MLP": MLPClassifier((64, 64), max_iter=1000, random_state=seed)}
        for nm, mdl in bb.items():
            rows.append({**base, "model": nm, "acc": mdl.fit(Xtr, ytr).score(Xte, yte)})
        # inner model selection of R (lam = 0) on a stratified 80/20 split of the training fold
        a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
        Xa, Xb = data.minmax(X[tr][a], X[tr][b])
        for fam in FAMS:
            val = {}
            for R in RS:
                p = train.fit(fam, Xa, ytr[a], meta["K"], R=R, seed=seed, **FIT)
                val[R] = train.accuracy(p, fam, Xb, ytr[b])
            R_sel = max(RS, key=lambda r: (round(val[r], 6), -r))         # best val acc, ties -> fewer rules
            for lam in LAMS:
                t0 = time.time()
                p = train.fit(fam, Xtr, ytr, meta["K"], R=R_sel, lam=lam, eps=0.02, seed=seed, **FIT)
                row = {**base, "model": fam, "lam": lam, "R": R_sel, "t_train": time.time() - t0,
                       "acc": train.accuracy(p, fam, Xte, yte), "finite": train.is_finite(p),
                       "rules_used": train.effective_rules(p, fam, Xte)[0]}
                for eps in EPS:
                    for meth in ["exact", "ibp", "lip"]:
                        if meth == "lip" and fam == "tsk_min":
                            continue
                        t1 = time.time()
                        c, _ = certify.certify(p, fam, Xte, eps, meth)
                        row[f"cert_{meth}_{eps}"] = c.mean(); row[f"t_{meth}_{eps}"] = time.time() - t1
                ce, _ = certify.certify(p, fam, Xte, 0.02, "exact")
                mc = attack.mc_sampling(p, fam, Xte, 0.02, n_samples=100, seed=seed)
                row["mc_stable_0.02"] = mc.mean()
                row["mc_false_claim_0.02"] = (mc & ~ce).mean()        # sampling says stable, exact says not
                row["mc_flip_on_cert_0.02"] = int((~mc & ce).sum())   # must be 0 (soundness)
                rad = certify.certified_radius(p, fam, Xte, eps_max=0.25, iters=14)
                row["radius_median"] = float(np.median(rad)); row["radius_mean"] = float(rad.mean())
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"{name} seed {seed}: {time.time() - t_ds:.0f}s", flush=True)
print("WORKER DONE", wid)
