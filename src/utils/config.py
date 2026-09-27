import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = None) -> dict:
    path = path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def subsample_stratified(X, y, max_n, seed):
    """Stratified subsample used to keep SVM-in-VotingClassifier training
    tractable on full-size real benchmark datasets (see config.yaml
    ml_branch.max_training_samples for rationale). No-op if already small
    enough or max_n is None."""
    import numpy as np
    if max_n is None or len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx_by_class = {}
    for cls in np.unique(y):
        idx_by_class[cls] = np.where(y == cls)[0]
    frac = max_n / len(y)
    chosen = []
    for cls, idxs in idx_by_class.items():
        n_take = max(1, int(round(len(idxs) * frac)))
        chosen.append(rng.choice(idxs, size=min(n_take, len(idxs)), replace=False))
    chosen = np.sort(np.concatenate(chosen))
    if hasattr(X, "iloc"):
        return X.iloc[chosen].reset_index(drop=True), y[chosen]
    return X[chosen], y[chosen]
