"""Gated models (Definition 1): pi = softmax(g(x)), F(x) = sum_r pi_r(x) f_r(x), f_r affine (K outputs).

families:
  'tsk_prod' : g_r(x) = -sum_j (x_j - c_rj)^2 / (2 s_rj^2)        (Gaussian MFs, product t-norm)
  'tsk_min'  : g_r(x) = min_j  -(x_j - c_rj)^2 / (2 s_rj^2)        (Gaussian MFs, minimum t-norm)
  'moe_lin'  : g_r(x) = w_r . x + b_r                              (linear softmax gating)
"""
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans

jax.config.update("jax_enable_x64", True)

FAMILIES = ("tsk_prod", "tsk_min", "moe_lin")


def init_params(family, X, K, R, seed=0):
    km = KMeans(R, n_init=4, random_state=seed).fit(X)
    C = km.cluster_centers_
    lab = km.labels_
    d = X.shape[1]
    S = np.stack([X[lab == r].std(0) if (lab == r).sum() > 1 else X.std(0) for r in range(R)])
    S = np.clip(S, 0.05, None)
    rng = np.random.default_rng(seed)
    p = {"theta": jnp.asarray(rng.normal(0, 0.01, (R, d, K))), "bias": jnp.zeros((R, K))}
    if family in ("tsk_prod", "tsk_min"):
        p["c"] = jnp.asarray(C)
        p["log_s"] = jnp.asarray(np.log(S))
    else:  # Voronoi-equivalent initialisation of linear gating
        tau = float(np.median(S)) ** 2
        p["W"] = jnp.asarray(C / tau)
        p["b"] = jnp.asarray(-0.5 * (C ** 2).sum(1) / tau)
    return p


def logits_gate(p, family, X):
    if family == "moe_lin":
        return X @ p["W"].T + p["b"]
    s = jnp.exp(p["log_s"])
    q = -((X[:, None, :] - p["c"][None]) ** 2) / (2 * s[None] ** 2)      # n x R x d  (log memberships)
    return q.sum(-1) if family == "tsk_prod" else q.min(-1)


def predict_logits(p, family, X):
    pi = jax.nn.softmax(logits_gate(p, family, X), axis=-1)               # n x R
    f = jnp.einsum("nd,rdk->nrk", X, p["theta"]) + p["bias"][None]       # n x R x K
    return jnp.einsum("nr,nrk->nk", pi, f)
