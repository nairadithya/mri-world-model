"""Explicit anatomical views of one observation decoder's logits.

Conditioning excludes the cavity channel before argmax; it does not merge a
native cavity prediction into background or any tumor compartment.
"""
from .state import LABELS


def anatomy_readouts(logits, support=None):
    if logits.ndim != 5 or logits.shape[1] not in (4, 5):
        raise ValueError("Expected four- or five-category spatial logits")
    native = logits.argmax(1)
    conditional = logits[:, :4].argmax(1)
    if logits.shape[1] == 4 and support is not None:
        # Preserve the original model's foreground-masked inference exactly.
        native = native.masked_fill(~support.bool(), 0)
        conditional = conditional.masked_fill(~support.bool(), 0)
    return {"native": native, "conditional_legacy": conditional}


def readout_contract(num_classes, dataset=None):
    if num_classes not in (4, 5):
        raise ValueError("Unsupported anatomy categories")
    labels = {0: "background", **LABELS}
    if num_classes == 5:
        labels[4] = "resection_cavity"
    return {
        "native": {"label": f"Native {num_classes}-category anatomy", "labels": labels,
                   "cavity_status": ("UNVALIDATED: LUMIERE has no cavity reference annotation"
                                     if dataset == "LUMIERE" else "Predicted; validation is dataset-specific")
                                    if num_classes == 5 else "Not modeled"},
        "conditional_legacy": {"label": "Conditional legacy 4-category anatomy",
                               "labels": {0: "background", **LABELS},
                               "definition": "Argmax over background, enhancing, necrotic/non-enhancing and edema/FLAIR logits; cavity excluded before argmax. Same model and saved state."},
    }
