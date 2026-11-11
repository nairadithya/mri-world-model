"""Compatible lesion compartments, retaining dataset-specific unsupported regions."""
import numpy as np


def harmonize_labels(values, dataset):
    if dataset == "LUMIERE":
        mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    elif dataset == "MU":
        # MU: 1 nonenhancing core, 2 FLAIR abnormality, 3 enhancing, 4 cavity.
        # Cavity has no counterpart in LUMIERE: ignore it for shared segmentation.
        mapping = {0: 0, 1: 2, 2: 3, 3: 1, 4: -1}
    else:
        raise ValueError("Unknown label definition")
    if not np.isfinite(values).all() or not np.isin(values, list(mapping)).all():
        raise ValueError("Invalid source labels")
    output = np.empty(values.shape, dtype=np.int8)
    for source, target in mapping.items():
        output[values == source] = target
    return output
