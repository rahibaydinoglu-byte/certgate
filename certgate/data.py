"""Dataset loading for CertGate experiments.

Sources: classic UCI datasets from a public GitHub mirror (jbrownlee/Datasets),
plus datasets bundled with scikit-learn. Every file is fingerprinted (SHA-256)
so that the exact data can be re-identified; original references are the UCI
repository entries.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold

MIRROR = os.environ.get("CERTGATE_DATA", "/home/claude/jb_datasets")

# name -> (file, label column (int index, -1 = last), has header, positive-class rule)
SPECS = {
    "banknote": ("banknote_authentication.csv", -1),
    "breast_w": ("breast-cancer-wisconsin.csv", -1),
    "ecoli": ("ecoli.csv", -1),
    "glass": ("glass.csv", -1),
    "haberman": ("haberman.csv", -1),
    "ionosphere": ("ionosphere.csv", -1),
    "mammography": ("mammography.csv", -1),
    "new_thyroid": ("new-thyroid.csv", -1),
    "oil_spill": ("oil-spill.csv", -1),
    "phoneme": ("phoneme.csv", -1),
    "pima": ("pima-indians-diabetes.csv", -1),
    "sonar": ("sonar.csv", -1),
    "wheat_seeds": ("wheat-seeds.csv", -1),
    "wine": ("wine.csv", -1),
    "winequality_red": ("winequality-red.csv", -1),
    "winequality_white": ("winequality-white.csv", -1),
}
BUNDLED = {"wdbc": load_breast_cancer, "digits": load_digits}
ALL = list(SPECS) + list(BUNDLED)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load(name):
    """Return X (float64, n x d), y (int, 0..K-1), meta dict."""
    if name in BUNDLED:
        b = BUNDLED[name]()
        X, y = b.data.astype(float), b.target
        meta = {"source": f"sklearn.datasets.{BUNDLED[name].__name__}", "sha256": None}
    else:
        fname, lab = SPECS[name]
        path = os.path.join(MIRROR, fname)
        df = pd.read_csv(path, header=None, na_values=["?"])
        df = df.dropna()
        y_raw = df.iloc[:, lab].astype(str).str.strip("'\" ")
        X = df.drop(columns=df.columns[lab]).apply(pd.to_numeric, errors="coerce")
        keep = ~X.isna().any(axis=1)
        X, y_raw = X[keep].to_numpy(float), y_raw[keep.to_numpy()]
        if name in ("winequality_red", "winequality_white"):
            y_raw = (y_raw.astype(float) >= 6).astype(int).astype(str)   # good (>=6) vs not
        if name == "oil_spill":
            X = X[:, 1:]                                                  # first column is a batch id
        y = pd.factorize(y_raw, sort=True)[0]
        meta = {"source": f"UCI via github.com/jbrownlee/Datasets/{fname}", "sha256": sha256(path)}
    # drop constant features
    keep = X.std(0) > 1e-12
    X = X[:, keep]
    # merge tiny classes (< 10 samples) into the nearest-larger "other" class to keep CV stratifiable
    counts = np.bincount(y)
    small = np.where(counts < 10)[0]
    if len(small):
        rare = small[0]
        y = np.where(np.isin(y, small), rare, y)                       # pool rare classes
        if (y == rare).sum() < 10:
            y = np.where(y == rare, counts.argmax(), y)               # still too small: merge into majority
        y = pd.factorize(y, sort=True)[0]
    meta.update(n=X.shape[0], d=X.shape[1], K=int(y.max() + 1))
    return X, y, meta


def splits(y, n_splits=5, seed=0):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def minmax(Xtr, Xte):
    lo, hi = Xtr.min(0), Xtr.max(0)
    sc = np.where(hi - lo > 0, hi - lo, 1.0)
    return (Xtr - lo) / sc, np.clip((Xte - lo) / sc, 0.0, 1.0)
