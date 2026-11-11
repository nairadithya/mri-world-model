"""Joint supervised anatomy, hidden-image reconstruction and local gradients."""
import torch
from torch.nn import functional as F


def joint_loss(logits, reconstruction, target, labels, support, hidden, supervised_classes=None, allow_all_observed=False):
    logits, reconstruction = logits.float(), reconstruction.float()
    full_field = logits.shape[1] == 5
    # Conditional likelihood over annotated classes. No absent-cavity target is
    # invented on LUMIERE, and its loss has zero derivative on the cavity logit.
    if supervised_classes is not None:
        if logits.shape[0] != 1 or supervised_classes not in (4, 5) or logits.shape[1] < supervised_classes:
            raise ValueError("Explicit per-visit supervision requires batch size one and four/five classes")
        logits = logits[:, :supervised_classes]
    classes = logits.shape[1]
    if ((labels >= classes) & (torch.ones_like(labels, dtype=torch.bool) if full_field else support[:, 0].bool())).any():
        raise ValueError("Label outside declared supervised classes")
    valid = (labels >= 0) if full_field else support[:, 0].bool() & (labels >= 0)
    if not valid.any():
        raise ValueError("Patch contains no valid supported labels")
    ce_values = F.cross_entropy(logits, labels.clamp_min(0), reduction="none")
    ce_terms = []
    for c in range(classes):
        selected = valid & (labels == c)
        if not selected.any():
            continue
        if full_field and c == 0:
            # Prevent the newly supervised outside-brain background from drowning
            # out ordinary-brain negatives in the class-balanced background term.
            regions = [selected & support[:, 0].bool(), selected & ~support[:, 0].bool()]
            ce_terms.append(torch.stack([ce_values[r].mean() for r in regions if r.any()]).mean())
        else:
            ce_terms.append(ce_values[selected].mean())
    ce = torch.stack(ce_terms).mean()
    probabilities = logits.softmax(1)
    dice_terms = []
    for c in range(1, classes):
        truth = (labels == c) & valid
        predicted = probabilities[:, c] * valid
        if truth.any():
            dice_terms.append(1 - (2 * predicted[truth].sum() + 1e-5) / (predicted.sum() + truth.sum() + 1e-5))
    dice = torch.stack(dice_terms).mean() if dice_terms else logits.sum() * 0
    mask = support.expand_as(target).bool()
    hidden_mask = hidden.bool() & mask
    if not hidden_mask.any() and not allow_all_observed:
        raise ValueError("Reconstruction needs an actually hidden supported target")
    error = (reconstruction - target).abs()
    reconstruction_loss = error[hidden_mask].mean() if hidden_mask.any() else reconstruction.sum()*0
    visible_loss = error[mask & ~hidden_mask].mean() if (mask & ~hidden_mask).any() else reconstruction.sum()*0
    gradients = []
    for axis in (2, 3, 4):
        first, second = [slice(None)]*5, [slice(None)]*5
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        first, second = tuple(first), tuple(second)
        valid_gradient = mask[first] & mask[second]
        gradients.append(((reconstruction[first]-reconstruction[second]) - (target[first]-target[second])).abs()[valid_gradient].mean())
    gradient = torch.stack(gradients).mean()
    segmentation = .5 * (ce + dice)
    loss = segmentation + .5 * reconstruction_loss + .1 * visible_loss + .1 * gradient
    return loss, {"segmentation_loss": float(segmentation.detach()), "hidden_mae": float(reconstruction_loss.detach()),
                  "visible_mae": float(visible_loss.detach()), "gradient_mae": float(gradient.detach())}


def segmentation_metrics(pred, target, support, classes=(1, 2, 3)):
    support = support & (target >= 0)
    result = {}
    for c in classes:
        a, b = (pred == c) & support, (target == c) & support
        n, m = int(a.sum()), int(b.sum())
        result[str(c)] = {"dice": 2*int((a & b).sum())/(n+m) if n+m else 1.,
                          "nonempty_dice": 2*int((a & b).sum())/(n+m) if m else None,
                          "volume_error_ml": (n-m)/1000., "empty_reference_fp_ml": n/1000. if not m else None}
    return result
