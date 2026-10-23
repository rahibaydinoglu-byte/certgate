"""Revision extras (seed 0, fold 0, R=4, lam=0):
(a) sampling false-claim rate vs number of samples; (b) top-2 certification; (c) tightness of Prop. 6 weight bounds."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from certgate import data, train, certify, attack
from certgate.models import logits_gate
rows, trows, wrows = [], [], []
DS = ["pima", "phoneme", "banknote", "wdbc", "winequality_red", "ionosphere"]
for name in DS:
    X, y, meta = data.load(name); tr, te = data.splits(y)[0]
    Xtr, Xte = data.minmax(X[tr], X[te]); Xte = Xte[:400]
    for fam in ["tsk_prod", "tsk_min", "moe_lin"]:
        p = train.fit(fam, Xtr, y[tr], meta["K"], R=4, lr=0.01, steps=1500)
        ce, _ = certify.certify(p, fam, Xte, 0.02)
        for n in [100, 1000, 10000]:
            mc = attack.mc_sampling(p, fam, Xte, 0.02, n_samples=n, seed=0)
            rows.append(dict(dataset=name, family=fam, n_samples=n, exact=ce.mean(), mc=mc.mean(), false_claim=(mc & ~ce).mean()))
        # (b) top-2 certification: pairs r in T2, s not in T2
        g = np.asarray(logits_gate(p, fam, jnp.asarray(Xte))); order = np.argsort(-g, 1)
        l, u = np.clip(Xte - 0.02, 0, 1), np.clip(Xte + 0.02, 0, 1)
        ok2 = np.ones(len(Xte), bool)
        for k in range(2):
            t = jnp.asarray(order[:, k])
            m = np.asarray(certify.pair_margin_exact(p, fam, t, jnp.asarray(l), jnp.asarray(u)))
            for s_ in range(4):
                outside = ~((order[:, 0] == s_) | (order[:, 1] == s_))
                ok2 &= ~(outside & (m[:, s_] <= 0))
        trows.append(dict(dataset=name, family=fam, top1=ce.mean(), top2=ok2.mean()))
        # (c) Prop. 6 bounds vs sampled range of pi_r (5000 samples per point, 100 points)
        idx = np.arange(min(100, len(Xte)))
        t0 = jnp.asarray(order[idx, 0]); L, U = jnp.asarray(l[idx]), jnp.asarray(u[idx])
        R = 4; lo_b = np.zeros((len(idx), R)); hi_b = np.zeros((len(idx), R))
        mins = np.stack([np.asarray(certify.pair_margin_exact(p, fam, jnp.full(len(idx), r), L, U)) for r in range(R)], 1)  # [i, r, s] = m_rs
        for r in range(R):
            Ms = np.array([[-mins[i, s, r] if s != r else -np.inf for s in range(R)] for i in range(len(idx))])  # M_sr = -m_rs... careful
            # M_sr(B) = max(g_s - g_r) = -min(g_r - g_s) = -m_rs ; m_sr = mins[:, s, r]
            msr = np.array([[mins[i, s, r] if s != r else np.inf for s in range(R)] for i in range(len(idx))])
            Msr = np.array([[-mins[i, r, s] if s != r else -np.inf for s in range(R)] for i in range(len(idx))])
            lo_b[:, r] = 1 / (1 + np.exp(Msr).sum(1)); hi_b[:, r] = 1 / (1 + np.exp(np.where(np.isinf(msr), -np.inf, msr)).sum(1))
        rng = np.random.default_rng(0); emp_lo = np.ones((len(idx), R)); emp_hi = np.zeros((len(idx), R))
        for _ in range(50):
            Z = l[idx] + (u[idx] - l[idx]) * rng.uniform(size=(len(idx), Xte.shape[1]))
            pi = np.asarray(jax.nn.softmax(logits_gate(p, fam, jnp.asarray(Z)), -1))
            emp_lo = np.minimum(emp_lo, pi); emp_hi = np.maximum(emp_hi, pi)
        viol = int(((emp_lo < lo_b - 1e-9) | (emp_hi > hi_b + 1e-9)).sum())
        wrows.append(dict(dataset=name, family=fam, viol=viol, width_bound=(hi_b - lo_b).mean(), width_emp=(emp_hi - emp_lo).mean()))
        print(name, fam, flush=True)
pd.DataFrame(rows).to_csv("results/extra_sampling.csv", index=False)
pd.DataFrame(trows).to_csv("results/extra_top2.csv", index=False)
pd.DataFrame(wrows).to_csv("results/extra_weights.csv", index=False)
print("EXTRA DONE")
