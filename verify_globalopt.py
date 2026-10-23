"""Independent check of the exact pairwise minima with a generic global optimiser (scipy differential evolution
+ L-BFGS-B polish), on trained models. For each (point, competitor) the optimiser value must be >= the exact
minimum (exact = global min) and should match it closely."""
import numpy as np, pandas as pd, jax.numpy as jnp
from scipy.optimize import differential_evolution
from certgate import data, train, certify
from certgate.witness import gate_np
rows = []
rng = np.random.default_rng(0)
for name in ["pima", "phoneme", "banknote", "wine"]:
    X, y, meta = data.load(name); tr, te = data.splits(y)[0]
    Xtr, Xte = data.minmax(X[tr], X[te])
    for fam in ["tsk_prod", "tsk_min", "moe_lin"]:
        p = train.fit(fam, Xtr, y[tr], meta["K"], R=4, lr=0.01, steps=1500)
        idx = rng.choice(len(Xte), 20, replace=False)
        for eps in [0.02, 0.05]:
            Xs = Xte[idx]; l, u = np.clip(Xs - eps, 0, 1), np.clip(Xs + eps, 0, 1)
            t = np.asarray(jnp.argmax(jnp.asarray(gate_np(p, fam, Xs)), -1))
            m = np.asarray(certify.pair_margin_exact(p, fam, jnp.asarray(t), jnp.asarray(l), jnp.asarray(u)))
            for i in range(len(Xs)):
                for s in range(4):
                    if s == t[i]: continue
                    f = lambda z: (lambda g: g[t[i]] - g[s])(gate_np(p, fam, z[None])[0])
                    r = differential_evolution(f, list(zip(l[i], u[i])), seed=1, tol=1e-10, maxiter=150, polish=True)
                    rows.append(dict(dataset=name, family=fam, eps=eps, exact=m[i, s], opt=r.fun))
D = pd.DataFrame(rows)
D["gap"] = D.opt - D.exact
print("pairs checked:", len(D))
print("optimiser below exact (would mean exact is wrong) by > 1e-8:", int((D.gap < -1e-8).sum()))
print("sign disagreements:", int(((D.exact > 0) != (D.opt > 0)).sum()))
print(D.groupby("family").gap.describe()[["mean", "50%", "max"]].to_string())
D.to_csv("results/globalopt_check.csv", index=False)
