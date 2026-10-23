"""Explicit witnesses for non-certified points (constructive side of Theorem 1).

For a point x with top rule t and box B, return z* in B and the competitor s with g_t(z*) - g_s(z*) = m_ts(B) <= 0.
Implemented in NumPy (float64) independently of the JAX certifier, so it also cross-checks it.
"""
import numpy as np


def _logmu(z, c, s):
    return -((z - c) ** 2) / (2 * s ** 2)


def gate_np(p, family, Z):
    if family == "moe_lin":
        return Z @ np.asarray(p["W"]).T + np.asarray(p["b"])
    C, S = np.asarray(p["c"]), np.exp(np.asarray(p["log_s"]))
    q = _logmu(Z[:, None, :], C[None], S[None])
    return q.sum(-1) if family == "tsk_prod" else q.min(-1)


def pair_witness(p, family, t, s, l, u):
    """Minimiser z* of g_t - g_s over the box [l,u] (single point, 1-D arrays)."""
    if family == "moe_lin":
        dW = np.asarray(p["W"])[t] - np.asarray(p["W"])[s]
        return np.where(dW > 0, l, u)
    C, S = np.asarray(p["c"]), np.exp(np.asarray(p["log_s"]))
    ct, st, cs, ss = C[t], S[t], C[s], S[s]
    A = lambda z: _logmu(z, ct, st)
    phi = lambda z: -_logmu(z, cs, ss)
    a = 1 / (2 * ss ** 2) - 1 / (2 * st ** 2)
    b = ct / st ** 2 - cs / ss ** 2
    vtx = np.clip(-b / (2 * np.where(np.abs(a) > 1e-12, a, 1.0)), l, u)
    if family == "tsk_prod":
        cand = np.stack([l, u, vtx])
        h = A(cand) + phi(cand)
        return cand[h.argmin(0), np.arange(len(l))]
    # tsk_min (Theorem 3): choose the best j, set z_j = t*, other coordinates at their phi-minimisers
    zphi = np.clip(cs, l, u)
    phimin = phi(zphi)
    best, zbest = np.inf, None
    for j in range(len(l)):
        K = np.max(np.delete(phimin, j)) if len(l) > 1 else -np.inf
        cands = [l[j], u[j], vtx[j]]
        if K > 0:
            r = ss[j] * np.sqrt(2 * K)
            cands += [np.clip(cs[j] - r, l[j], u[j]), np.clip(cs[j] + r, l[j], u[j])]
        cands = np.array(cands)
        F = _logmu(cands, ct[j], st[j]) + np.maximum(-_logmu(cands, cs[j], ss[j]), K)
        k = F.argmin()
        if F[k] < best:
            best = F[k]
            zbest = zphi.copy()
            zbest[j] = cands[k]
    return zbest


def verify_uncertified(p, family, X, eps, tol=1e-9):
    """For every point, find the most violating competitor and its witness.
    Returns (is_certified_by_witness_search, witness_found) arrays."""
    G = gate_np(p, family, X)
    n, R = G.shape
    found = np.zeros(n, bool)
    for i in range(n):
        t = G[i].argmax()
        l, u = np.clip(X[i] - eps, 0, 1), np.clip(X[i] + eps, 0, 1)
        for s in range(R):
            if s == t:
                continue
            z = pair_witness(p, family, t, s, l, u)
            assert np.all(z >= l - 1e-12) and np.all(z <= u + 1e-12)
            gz = gate_np(p, family, z[None])[0]
            if gz[s] >= gz[t] - tol:
                found[i] = True
                break
    return found
