"""Certification of the top-1 routing explanation on boxes B = [l, u] (one box per sample).

pair_margin_exact : m_ts(B) = min_{z in B} g_t(z) - g_s(z) for all s  (Theorem 1 + Cor. 1/2, Theorem 3)
pair_margin_ibp   : L_t - U_s from exact per-logit ranges (interval / IBP-style baseline)
pair_margin_lip   : first-order (mean-value) bound with a box-sup of the gradient (Lipschitz baseline)
All functions are vectorised: X (n,d), t (n,), l,u (n,d) -> (n,R) with +inf at s = t.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _gauss_terms(p, t):
    s = jnp.exp(p["log_s"])
    ct, st = p["c"][t], s[t]                       # n x d
    return ct, st, p["c"], s                       # (n,d),(n,d),(R,d),(R,d)


def _mask_self(m, t):
    R = m.shape[1]
    return jnp.where(jax.nn.one_hot(t, R) > 0, jnp.inf, m)


def pair_margin_exact(p, family, t, l, u):
    if family == "moe_lin":
        dW = p["W"][t][:, None, :] - p["W"][None]                       # n x R x d
        z = jnp.where(dW > 0, l[:, None, :], u[:, None, :])
        m = (dW * z).sum(-1) + (p["b"][t][:, None] - p["b"][None])
        return _mask_self(m, t)
    ct, st, C, S = _gauss_terms(p, t)
    ct, st = ct[:, None, :], st[:, None, :]                              # n x 1 x d
    C, S = C[None], S[None]                                              # 1 x R x d
    L, U = l[:, None, :], u[:, None, :]
    a = 1 / (2 * S ** 2) - 1 / (2 * st ** 2)                             # coefficient of z^2 in A_t + phi_s
    b = ct / st ** 2 - C / S ** 2
    vtx = jnp.clip(-b / (2 * jnp.where(jnp.abs(a) > 1e-12, a, 1.0)), L, U)
    A = lambda z: -((z - ct) ** 2) / (2 * st ** 2)                       # log mu of rule t
    phi = lambda z: ((z - C) ** 2) / (2 * S ** 2)                        # -log mu of rule s
    if family == "tsk_prod":                                             # Cor. 2: separable, O(d)
        cand = jnp.stack([L * jnp.ones_like(a), U * jnp.ones_like(a), vtx])
        h = A(cand) + phi(cand)
        return _mask_self(h.min(0).sum(-1), t)
    # tsk_min, Theorem 3
    phimin = phi(jnp.clip(C, L, U))                                      # n x R x d : min_{z_i} phi_i
    top1 = phimin.max(-1, keepdims=True)
    arg1 = phimin.argmax(-1)[..., None]
    second = jnp.sort(phimin, axis=-1)[..., -2:-1]
    d = phimin.shape[-1]
    is_arg = jnp.arange(d)[None, None, :] == arg1
    Kj = jnp.where(is_arg, second, top1)                                 # max over i != j
    rt = S * jnp.sqrt(jnp.maximum(2 * Kj, 1e-12))   # safe gradient at K_j = 0
    cands = [L * jnp.ones_like(a), U * jnp.ones_like(a), vtx,
             jnp.clip(C - rt, L, U), jnp.clip(C + rt, L, U)]
    F = jnp.stack([A(z) + jnp.maximum(phi(z), Kj) for z in cands]).min(0)  # n x R x d
    return _mask_self(F.min(-1), t)


def logit_ranges(p, family, l, u):
    """Exact per-logit ranges [Lo, Hi] over each box (n x R)."""
    if family == "moe_lin":
        W = p["W"][None]
        lo = jnp.where(W > 0, l[:, None, :], u[:, None, :])
        hi = jnp.where(W > 0, u[:, None, :], l[:, None, :])
        return (W * lo).sum(-1) + p["b"], (W * hi).sum(-1) + p["b"]
    S = jnp.exp(p["log_s"])[None]
    C = p["c"][None]
    L, U = l[:, None, :], u[:, None, :]
    q_max = -((jnp.clip(C, L, U) - C) ** 2) / (2 * S ** 2)               # per-dim max of log mu
    q_min = jnp.minimum(-((L - C) ** 2), -((U - C) ** 2)) / (2 * S ** 2)
    if family == "tsk_prod":
        return q_min.sum(-1), q_max.sum(-1)
    return q_min.min(-1), q_max.min(-1)


def pair_margin_ibp(p, family, t, l, u):
    lo, hi = logit_ranges(p, family, l, u)
    m = lo[jnp.arange(len(t)), t][:, None] - hi
    return _mask_self(m, t)


def pair_margin_lip(p, family, X, t, l, u):
    """(g_t - g_s)(x) - sum_j max(x_j - l_j, u_j - x_j) * sup_{z in B} |d_j (g_t - g_s)|.
    Defined for the smooth families (moe_lin, tsk_prod)."""
    from .models import logits_gate
    g = logits_gate(p, family, X)
    diff = g[jnp.arange(len(t)), t][:, None] - g
    rad = jnp.maximum(X - l, u - X)[:, None, :]
    if family == "moe_lin":
        G = jnp.abs(p["W"][t][:, None, :] - p["W"][None])
    elif family == "tsk_prod":
        ct, st, C, S = _gauss_terms(p, t)
        ct, st, C, S = ct[:, None], st[:, None], C[None], S[None]
        L, U = l[:, None, :], u[:, None, :]
        grad = lambda z: -(z - ct) / st ** 2 + (z - C) / S ** 2         # affine in z -> sup|.| at endpoints
        G = jnp.maximum(jnp.abs(grad(L)), jnp.abs(grad(U)))
    else:
        raise ValueError("Lipschitz baseline not defined for the non-smooth min t-norm")
    return _mask_self(diff - (rad * G).sum(-1), t)


def boxes(X, eps):
    return np.clip(X - eps, 0, 1), np.clip(X + eps, 0, 1)


def certify(p, family, X, eps, method="exact"):
    from .models import logits_gate
    X = jnp.asarray(X)
    t = jnp.argmax(logits_gate(p, family, X), -1)
    l, u = boxes(np.asarray(X), eps)
    l, u = jnp.asarray(l), jnp.asarray(u)
    if method == "exact":
        m = pair_margin_exact(p, family, t, l, u)
    elif method == "ibp":
        m = pair_margin_ibp(p, family, t, l, u)
    elif method == "lip":
        m = pair_margin_lip(p, family, X, t, l, u)
    else:
        raise ValueError(method)
    return np.asarray(m.min(-1) > 0), np.asarray(t)


def certified_radius(p, family, X, eps_max=0.5, iters=20, method="exact"):
    lo = np.zeros(len(X))
    hi = np.full(len(X), eps_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = np.zeros(len(X), bool)
        # per-sample eps: build boxes with vector eps
        Xn = np.asarray(X)
        l, u = np.clip(Xn - mid[:, None], 0, 1), np.clip(Xn + mid[:, None], 0, 1)
        from .models import logits_gate
        t = jnp.argmax(logits_gate(p, family, jnp.asarray(Xn)), -1)
        fn = {"exact": pair_margin_exact, "ibp": pair_margin_ibp}[method]
        ok = np.asarray(fn(p, family, t, jnp.asarray(l), jnp.asarray(u)).min(-1) > 0)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo
