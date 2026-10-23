"""Theorem 7 experiment: synthetic data with known routing regimes rho*.

X ~ U[0,1]^2; rho*(x) = Voronoi cell of R*=4 fixed centres (a linear gating);
y = 1[ a_rho(x) . x + b_rho > 0 ] with 5% label noise (one linear rule per regime).
For each lambda we report: accuracy, routing accuracy vs rho* (Hungarian matching), certified fraction,
the intrinsic ceiling 1 - P(N*_eps), and a constructive check of the proof of Theorem 7:
every x in Cert & Cor & N* must have, inside its box, a point z with rho*(z) != rho*(x) and T_1(z) != rho*(z).
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, pandas as pd, jax.numpy as jnp
from scipy.optimize import linear_sum_assignment
from certgate import train, certify
from certgate.models import logits_gate
from certgate.witness import pair_witness, gate_np

rng = np.random.default_rng(0)
Cstar = np.array([[0.25, 0.25], [0.75, 0.3], [0.3, 0.75], [0.72, 0.72]])
truth = {"W": jnp.asarray(Cstar), "b": jnp.asarray(-0.5 * (Cstar ** 2).sum(1))}       # Voronoi = linear gating
A = rng.normal(0, 1, (4, 2)); A /= np.linalg.norm(A, axis=1, keepdims=True)
B = -(A * Cstar).sum(1)                                                                 # boundary through each centre


def sample(n, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, 2))
    rho = np.asarray(jnp.argmax(logits_gate(truth, "moe_lin", jnp.asarray(X)), -1))
    y = ((A[rho] * X).sum(1) + B[rho] > 0).astype(int)
    flip = r.uniform(size=n) < 0.05
    return X, np.where(flip, 1 - y, y), rho


Xtr, ytr, _ = sample(3000, 1)
Xte, yte, rho_te = sample(5000, 2)
rows = []
EPS = [0.02, 0.05]
for fam in ["tsk_prod", "moe_lin"]:
    for lam in [0.0, 0.1, 0.3, 1.0, 3.0]:
        for seed in range(3):
            p = train.fit(fam, Xtr, ytr, 2, R=4, lam=lam, eps=0.02, seed=seed, lr=0.01, steps=2000)
            T = np.asarray(jnp.argmax(logits_gate(p, fam, jnp.asarray(Xte)), -1))
            conf = np.zeros((4, 4)); np.add.at(conf, (T, rho_te), 1)
            ri, ci = linear_sum_assignment(-conf)
            perm = dict(zip(ri, ci))                                   # model rule -> true regime
            Tm = np.array([perm[t] for t in T])
            cor = Tm == rho_te
            row = dict(family=fam, lam=lam, seed=seed, acc=train.accuracy(p, fam, Xte, yte), route_acc=cor.mean())
            for eps in EPS:
                cert, _ = certify.certify(p, fam, Xte, eps)
                star, _ = certify.certify(truth, "moe_lin", Xte, eps)    # x not in N*_eps
                row[f"cert_{eps}"] = cert.mean()
                row[f"ceiling_{eps}"] = star.mean()
                row[f"cert_and_cor_{eps}"] = (cert & cor).mean()
                # constructive check of Theorem 7's proof on Cert & Cor & N*
                idx = np.where(cert & cor & ~star)[0]
                ok = 0
                for i in idx:
                    l, u = np.clip(Xte[i] - eps, 0, 1), np.clip(Xte[i] + eps, 0, 1)
                    r0 = rho_te[i]
                    found = False
                    for s in range(4):
                        if s == r0:
                            continue
                        z = pair_witness(truth, "moe_lin", r0, s, l, u)
                        gz = gate_np(truth, "moe_lin", z[None])[0]
                        if gz.argmax() != r0:                            # rho*(z) != rho*(x)
                            tz = perm[int(np.asarray(jnp.argmax(logits_gate(p, fam, jnp.asarray(z[None])), -1))[0])]
                            found = tz != gz.argmax()                    # z is misrouted
                            break
                    ok += found
                row[f"n_cert_cor_Nstar_{eps}"] = len(idx)
                row[f"thm7_witness_ok_{eps}"] = ok
            rows.append(row)
            print(row, flush=True)
os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/synth_thm7.csv", index=False)
print("SYNTH DONE")
