"""Re-train the configurations where PGD missed flips and check that an explicit witness exists for
every point the exact certificate rejects (and none for certified points)."""
import numpy as np, pandas as pd
from certgate import data, train, certify, attack
from certgate.witness import verify_uncertified
M = pd.read_csv("results/pilot_main.csv")
bad = M[M["attack_missed_0.02"] > 0][["dataset", "fold", "model", "lam"]]
tot_unc = tot_wit = tot_cert_wit = 0
for _, r in bad.iterrows():
    X, y, meta = data.load(r.dataset); tr, te = data.splits(y)[r.fold]
    Xtr, Xte = data.minmax(X[tr], X[te])
    p = train.fit(r.model, Xtr, y[tr], meta["K"], R=4, lr=0.01, steps=1500, lam=r.lam, eps=0.02)
    ce, _ = certify.certify(p, r.model, Xte, 0.02)
    fl = attack.attack_flip(p, r.model, Xte, 0.02, steps=30, restarts=3)
    wit = verify_uncertified(p, r.model, Xte, 0.02)
    unc = ~ce
    tot_unc += unc.sum(); tot_wit += (wit & unc).sum(); tot_cert_wit += (wit & ce).sum()
    print(f"{r.dataset:18s} f{r.fold} {r.model:8s} lam={r.lam}: uncertified={unc.sum():4d} witness={int((wit&unc).sum()):4d} "
          f"pgd-missed={int((unc&~fl).sum()):4d} witness-on-certified={int((wit&ce).sum())}")
print(f"TOTAL uncertified {tot_unc}, with explicit witness {tot_wit}, witnesses on certified points {tot_cert_wit}")
