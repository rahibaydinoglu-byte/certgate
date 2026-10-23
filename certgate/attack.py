"""Empirical search for routing flips inside the box (upper bound on the true stable set).
Projected gradient descent on the pairwise margin g_t - g_s for every competitor s, with random restarts.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .models import logits_gate

jax.config.update("jax_enable_x64", True)


def attack_flip(p, family, X, eps, steps=40, restarts=4, seed=0):
    X = jnp.asarray(X)
    n, d = X.shape
    g0 = logits_gate(p, family, X)
    t = jnp.argmax(g0, -1)
    R = g0.shape[1]
    l, u = jnp.clip(X - eps, 0, 1), jnp.clip(X + eps, 0, 1)

    def margin(Z, s):
        g = logits_gate(p, family, Z)
        return (g[jnp.arange(n), t] - g[jnp.arange(n), s]).sum()

    grad = jax.jit(jax.grad(margin))
    key = jax.random.PRNGKey(seed)
    flipped = np.zeros(n, bool)
    step = 2.5 * eps / steps
    for s_ in range(R):
        s = jnp.full(n, s_)
        for r in range(restarts):
            key, k = jax.random.split(key)
            Z = jnp.clip(X + jax.random.uniform(k, (n, d), minval=-eps, maxval=eps), l, u)
            for _ in range(steps):
                Z = jnp.clip(Z - step * jnp.sign(grad(Z, s)), l, u)
            g = logits_gate(p, family, Z)
            flipped |= np.asarray((jnp.argmax(g, -1) != t) & (t != s_))
    return flipped


def mc_sampling(p, family, X, eps, n_samples=100, seed=0):
    """Sampling-based (probabilistic) stability check: a point is declared stable if none of n_samples
    uniform draws from the box changes the top-1 rule. With n=100 and no flip, the flip probability
    under uniform sampling is < 3% at 95% confidence (rule of three) — a statistical, not a worst-case, claim."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    t = np.asarray(jnp.argmax(logits_gate(p, family, jnp.asarray(X)), -1))
    l, u = np.clip(X - eps, 0, 1), np.clip(X + eps, 0, 1)
    flip = np.zeros(len(X), bool)
    for _ in range(n_samples):
        Z = l + (u - l) * rng.uniform(size=X.shape)
        flip |= np.asarray(jnp.argmax(logits_gate(p, family, jnp.asarray(Z)), -1)) != t
    return ~flip
