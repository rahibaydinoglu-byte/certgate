"""Linguistic case study: TSK with shared, labelled fuzzy partitions (Low / Medium / High per feature).

Antecedents are fixed by snapping k-means centroids to the nearest label; duplicate rules are merged. The shared
centres and widths and the consequents are trained (product t-norm, optional certified-stability penalty).
Per-rule parameters are materialised so that the certifier of Corollary 2 applies unchanged.
Outputs: results/casestudy.json, results/casestudy.txt
"""
import json, os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np, jax, jax.numpy as jnp, optax
from sklearn.cluster import KMeans
from certgate import data, certify
from certgate.train import make_loss
from certgate.models import logits_gate, predict_logits
from certgate.witness import pair_witness, gate_np

jax.config.update("jax_enable_x64", True)
LABELS = ["Low", "Medium", "High"]
NAMES = {"banknote": ["variance", "skewness", "curtosis", "entropy"],
         "haberman": ["age", "year of operation", "positive nodes"],
         "new_thyroid": ["T3-resin uptake", "total thyroxin", "total triiodothyronine", "basal TSH", "max. TSH difference"]}
CLASSES = {"banknote": ["class 0", "class 1"], "haberman": ["survived >= 5 y", "died < 5 y"],
           "new_thyroid": ["normal", "hyper", "hypo"]}


GAP = 0.1   # minimum distance between neighbouring label centres (scaled units) -> keeps Low < Medium < High


def centres(q):
    c0 = q["c_raw"][:, 0]
    c1 = c0 + GAP + jax.nn.softplus(q["c_raw"][:, 1])
    c2 = c1 + GAP + jax.nn.softplus(q["c_raw"][:, 2])
    return jnp.stack([c0, c1, c2], 1)                       # d x 3, strictly ordered


def materialise(q, A):
    d = A.shape[1]
    c = centres(q)[jnp.arange(d)[None, :], A]              # R x d
    ls = q["log_ss"][jnp.arange(d)[None, :], A]
    return {"c": c, "log_s": ls, "theta": q["theta"], "bias": q["bias"]}


def fit_shared(Xtr, ytr, K, R0=6, lam=0.0, eps=0.02, steps=2000, lr=0.01, seed=0, zeta=0.0, tau=0.3):
    d = Xtr.shape[1]
    qs = np.quantile(Xtr, [1 / 6, 1 / 2, 5 / 6], axis=0).T                 # d x 3 label centres
    km = KMeans(R0, n_init=8, random_state=seed).fit(Xtr)
    A = np.abs(km.cluster_centers_[:, :, None] - qs[None]).argmin(-1)      # R0 x d label indices
    A = np.unique(A, axis=0)
    R = len(A)
    width = np.maximum(np.diff(qs, axis=1).mean(1, keepdims=True) / 1.5, 0.05) * np.ones((d, 3))
    rng = np.random.default_rng(seed)
    qs = np.stack([qs[:, 0], np.maximum(qs[:, 1], qs[:, 0] + GAP + 0.01), np.maximum(qs[:, 2], np.maximum(qs[:, 1], qs[:, 0] + GAP + 0.01) + GAP + 0.01)], 1)
    inv_sp = lambda v: np.log(np.expm1(np.maximum(v, 1e-3)))
    c_raw = np.stack([qs[:, 0], inv_sp(qs[:, 1] - qs[:, 0] - GAP), inv_sp(qs[:, 2] - qs[:, 1] - GAP)], 1)
    q = {"c_raw": jnp.asarray(c_raw), "log_ss": jnp.asarray(np.log(np.minimum(width, 0.25))),
         "theta": jnp.asarray(rng.normal(0, 0.01, (R, d, K))), "bias": jnp.zeros((R, K))}
    base = make_loss("tsk_prod", lam, eps, 0.5, 0.1)
    Aj = jnp.asarray(A)
    GRID = jnp.linspace(0.0, 1.0, 401)

    def similarity(q):
        """Jaccard similarity of neighbouring terms (Low-Medium, Medium-High) per feature, on a grid."""
        c, s = centres(q), jnp.exp(q["log_ss"])
        mu = jnp.exp(-(GRID[None, None, :] - c[:, :, None]) ** 2 / (2 * s[:, :, None] ** 2))     # d x 3 x G
        sim = lambda a, b: jnp.minimum(mu[:, a], mu[:, b]).sum(-1) / jnp.maximum(mu[:, a], mu[:, b]).sum(-1)
        return jnp.stack([sim(0, 1), sim(1, 2)], 1)                                              # d x 2

    def loss(q, X, y, l, u):
        v, aux = base(materialise(q, Aj), X, y, l, u)
        if zeta > 0:   # distinguishability constraint: penalise neighbouring-term similarity above tau
            v = v + zeta * jnp.mean(jax.nn.relu(similarity(q) - tau) ** 2) * 100.0
        return v, aux
    opt = optax.adamw(lr, weight_decay=1e-4); st = opt.init(q)
    X, y = jnp.asarray(Xtr), jnp.asarray(ytr)
    l, u = jnp.asarray(np.clip(Xtr - eps, 0, 1)), jnp.asarray(np.clip(Xtr + eps, 0, 1))

    def step(carry, _):
        q, st = carry
        (v, aux), g = jax.value_and_grad(loss, has_aux=True)(q, X, y, l, u)
        up, st = opt.update(g, st, q); q = optax.apply_updates(q, up)
        q = {**q, "log_ss": jnp.clip(q["log_ss"], np.log(0.02), np.log(0.25))}
        return (q, st), None
    q, st = jax.jit(lambda q, st: jax.lax.scan(step, (q, st), None, length=steps)[0])(q, st)
    return q, A


def distinguishability(q):
    """Max Jaccard similarity between labels of the same feature (lower = better separated)."""
    grid = np.linspace(0, 1, 2001)
    cs, ss = np.asarray(centres(q)), np.exp(np.asarray(q["log_ss"]))
    sims = []
    for j in range(cs.shape[0]):
        mu = np.exp(-(grid[None] - cs[j][:, None]) ** 2 / (2 * ss[j][:, None] ** 2))
        s = [np.minimum(mu[a], mu[b]).sum() / np.maximum(mu[a], mu[b]).sum() for a in range(3) for b in range(a + 1, 3)]
        sims.append(max(s))
    return float(np.mean(sims)), float(np.max(sims))


def rule_text(A, r, names, q, lo, hi, cls_names, p, K):
    parts = [f"{names[j]} is {LABELS[A[r, j]]}" for j in range(A.shape[1])]
    return "IF " + " AND ".join(parts)


out, txt = {}, []
for name in ["banknote", "haberman", "new_thyroid"]:
    X, y, meta = data.load(name)
    tr, te = data.splits(y)[0]
    lo, hi = X[tr].min(0), X[tr].max(0)
    Xtr, Xte = data.minmax(X[tr], X[te])
    res = {}
    for lam, zeta in [(0.0, 0.0), (0.3, 0.0), (0.3, 1.0)]:
        q, A = fit_shared(Xtr, y[tr], meta["K"], lam=lam, zeta=zeta)
        key = f"lam={lam}" + (f",sep" if zeta > 0 else "")
        p = materialise(q, jnp.asarray(A))
        p = {k: np.asarray(v) for k, v in p.items()}
        pj = {k: jnp.asarray(v) for k, v in p.items()}
        acc = float((np.asarray(jnp.argmax(predict_logits(pj, "tsk_prod", jnp.asarray(Xte)), -1)) == y[te]).mean())
        cert, top = certify.certify(pj, "tsk_prod", Xte, 0.02)
        rad = certify.certified_radius(pj, "tsk_prod", Xte, eps_max=0.25, iters=14)
        dmean, dmax = distinguishability(q)
        res[key] = dict(R=int(len(A)), acc=acc, cert_0_02=float(cert.mean()), radius_median=float(np.median(rad)),
                                 distinguishability_mean=dmean, distinguishability_max=dmax)
        if lam == 0.3 and zeta > 0:
            # one certified point with large radius and one refused point with its counterexample
            i_c = int(np.argmax(np.where(cert, rad, -1)))
            refused = np.where(~cert)[0]
            i_r = None
            if len(refused):
                order_r = refused[np.argsort(-rad[refused])]          # largest radius first: closest to certified
                fallback = None
                for cand in order_r:
                    l_, u_ = np.clip(Xte[cand] - 0.02, 0, 1), np.clip(Xte[cand] + 0.02, 0, 1)
                    t_ = int(top[cand]); p0 = int(np.argmax(np.asarray(predict_logits(pj, "tsk_prod", jnp.asarray(Xte[cand:cand + 1])))[0]))
                    for s_ in range(len(A)):
                        if s_ == t_: continue
                        z_ = pair_witness(p, "tsk_prod", t_, s_, l_, u_)
                        snew_ = int(gate_np(p, "tsk_prod", z_[None])[0].argmax())
                        if snew_ != t_:
                            same_pred = int(np.argmax(np.asarray(predict_logits(pj, "tsk_prod", jnp.asarray(z_[None])))[0])) == p0
                            one_term = int((A[t_] != A[snew_]).sum()) == 1
                            if same_pred and one_term:
                                i_r = int(cand)
                            elif same_pred and fallback is None:
                                fallback = int(cand)
                            break
                    if i_r is not None: break
                if i_r is None: i_r = fallback
                if i_r is None: i_r = int(order_r[0])
            K = meta["K"]
            def describe(i):
                t = int(top[i]); xo = X[te][i]
                pred = int(np.argmax(np.asarray(predict_logits(pj, "tsk_prod", jnp.asarray(Xte[i:i + 1])))[0]))
                return t, xo, pred
            t, xo, pred = describe(i_c)
            ex = dict(certified=dict(rule=rule_text(A, t, NAMES[name], q, lo, hi, CLASSES[name], p, K),
                                     x=[round(float(v), 3) for v in xo], prediction=CLASSES[name][pred],
                                     radius=float(rad[i_c]), radius_original_units=[round(float(rad[i_c] * (hi[j] - lo[j])), 3) for j in range(len(lo))]))
            if i_r is not None:
                t, xo, pred = describe(i_r)
                l, u = np.clip(Xte[i_r] - 0.02, 0, 1), np.clip(Xte[i_r] + 0.02, 0, 1)
                best = None
                for s in range(len(A)):
                    if s == t: continue
                    z = pair_witness(p, "tsk_prod", t, s, l, u)
                    gz = gate_np(p, "tsk_prod", z[None])[0]
                    if gz.argmax() != t:
                        best = (s, z, int(gz.argmax())); break
                s, z, snew = best
                zo = lo + z * (hi - lo)
                predz = int(np.argmax(np.asarray(predict_logits(pj, "tsk_prod", jnp.asarray(z[None])))[0]))
                ex["refused"] = dict(rule=rule_text(A, t, NAMES[name], q, lo, hi, CLASSES[name], p, K),
                                     x=[round(float(v), 3) for v in xo], prediction=CLASSES[name][pred],
                                     counterexample=[round(float(v), 3) for v in zo],
                                     change=[round(float(v), 3) for v in (zo - xo)],
                                     new_rule=rule_text(A, snew, NAMES[name], q, lo, hi, CLASSES[name], p, K),
                                     prediction_at_counterexample=CLASSES[name][predz])
            res["examples"] = ex
            res["rules"] = [rule_text(A, r, NAMES[name], q, lo, hi, CLASSES[name], p, K) for r in range(len(A))]
            res["partition"] = {NAMES[name][j]: {LABELS[k]: [round(float(lo[j] + centres(q)[j, k] * (hi[j] - lo[j])), 3),
                                                             round(float(np.exp(q["log_ss"][j, k]) * (hi[j] - lo[j])), 3)]
                                                 for k in range(3)} for j in range(len(lo))}
    out[name] = res
    txt.append(f"== {name}: {json.dumps({k: v for k, v in res.items() if k.startswith('lam')}, indent=1)}")
    txt.append(json.dumps(res.get("examples"), indent=1))
json.dump(out, open("results/casestudy.json", "w"), indent=1)
open("results/casestudy.txt", "w").write("\n".join(txt))
print("\n".join(txt))
