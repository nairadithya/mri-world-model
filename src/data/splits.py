"""Patient-level train/val/test splits, stratified by survival tertile."""
from __future__ import annotations

import numpy as np
import pandas as pd


def patient_splits(
    demographics_csv: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split patient IDs into train/val/test at the patient level.

    Stratifies by survival-time tertile (missing survival -> own stratum) so
    each split covers the prognosis range.
    """
    demo = pd.read_csv(demographics_csv)
    patients = demo["Patient"].tolist()

    surv = pd.to_numeric(demo["Survival time (weeks)"], errors="coerce")
    strata = pd.Series("missing", index=demo.index)
    valid = surv.dropna()
    if len(valid) >= 3:
        try:
            bins = pd.qcut(valid, 3, labels=["low", "mid", "high"], duplicates="drop")
            strata.loc[valid.index] = bins.astype(str)
        except ValueError:
            pass

    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for _, group in demo.groupby(strata):
        ids = group["Patient"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train += ids[:n_train]
        val += ids[n_train:n_train + n_val]
        test += ids[n_train + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
