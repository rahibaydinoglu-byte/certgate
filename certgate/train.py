"""Training with an optional scale-invariant certified-stability penalty (Section 4).

penalty_i = ReLU( gamma - min_s  m_ts(B_eps(x_i)) / (g_t(x_i) - g_s(x_i) + delta) ),  t = argmax g(x_i)
The ratio is invariant to logit scaling (Corollary 3), so the penalty cannot be met by sharpening the gates.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .certify import pair_margin_exact
from .models import init_params, logits_gate, predict_logits

jax.config.update("jax_enable_x64", True)


def make_loss(family, lam, eps, gamma, beta, penalty="ratio"):
    def loss(p, X, y, l, u):
        out = predict_logits(p, family, X)
        ce = optax.softmax_cross_entropy_with_integer_labels(out, y).mean()
        g = logits_gate(p, family, X)
        R = g.shape[1]
        # load balancing: KL(mean routing || uniform); prevents the trivial "one rule everywhere" solution
        usage = jax.nn.softmax(g, -1).mean(0)
        bal = (usage * jnp.log(usage * R + 1e-12)).sum()
        if lam == 0:
            return ce + beta * bal, (ce, 0.0)
        t = jax.lax.stop_gradient(jnp.argmax(g, -1))
        self_mask = jax.nn.one_hot(t, R) > 0
        m = pair_margin_exact(p, family, t, l, u)
        m = jnp.where(self_mask, 0.0, m)                                 # remove +inf before arithmetic
        if penalty == "raw":                                              # ablation: scale-dependent hinge
            mm = jnp.where(self_mask, jnp.inf, m).min(-1)
            pen = jax.nn.relu(gamma - jnp.clip(mm, -10.0, 10.0)).mean()
        else:
            gap = g[jnp.arange(len(t)), t][:, None] - g
            ratio = jnp.clip(m / (jnp.where(self_mask, 1.0, gap) + 1e-6), -1.0, 1.0)
            ratio = jnp.where(self_mask, 1.0, ratio)
            pen = jax.nn.relu(gamma - ratio.min(-1)).mean()
        return ce + beta * bal + lam * pen, (ce, pen)
    return loss


def fit(family, Xtr, ytr, K, R=8, lam=0.0, eps=0.02, gamma=0.5, steps=1500, lr=0.03, seed=0, wd=1e-4,
        beta=0.1, penalty="ratio"):
    p = init_params(family, Xtr, K, R, seed)
    loss = make_loss(family, lam, eps, gamma, beta, penalty)
    opt = optax.adamw(lr, weight_decay=wd)
    st = opt.init(p)
    X, y = jnp.asarray(Xtr), jnp.asarray(ytr)
    l, u = jnp.asarray(np.clip(Xtr - eps, 0, 1)), jnp.asarray(np.clip(Xtr + eps, 0, 1))

    def step(carry, _):
        p, st = carry
        (v, aux), gr = jax.value_and_grad(loss, has_aux=True)(p, X, y, l, u)
        upd, st = opt.update(gr, st, p)
        p = optax.apply_updates(p, upd)
        if "log_s" in p:
            p = {**p, "log_s": jnp.clip(p["log_s"], np.log(0.02), np.log(2.0))}
        return (p, st), None

    run = jax.jit(lambda p, st: jax.lax.scan(step, (p, st), None, length=steps)[0])
    p, st = run(p, st)
    return p


def effective_rules(p, family, X, thr=0.01):
    """Number of rules that are top-1 for at least `thr` of the points, and exp-entropy of top-1 usage."""
    t = np.asarray(jnp.argmax(logits_gate(p, family, jnp.asarray(X)), -1))
    R = logits_gate(p, family, jnp.asarray(X[:1])).shape[1]
    freq = np.bincount(t, minlength=R) / len(t)
    nz = freq[freq > 0]
    return int((freq >= thr).sum()), float(np.exp(-(nz * np.log(nz)).sum()))


def is_finite(p):
    return all(bool(np.isfinite(np.asarray(v)).all()) for v in p.values())


def accuracy(p, family, X, y):
    return float((np.asarray(jnp.argmax(predict_logits(p, family, jnp.asarray(X)), -1)) == y).mean())
