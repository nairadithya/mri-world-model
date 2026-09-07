"""Pure-PyTorch training loop: AdamW + cosine/warmup + AMP + EMA + wandb.

Large-batch dynamics at batch-1 memory via gradient accumulation
(training.accumulation_steps, default 1 = legacy behavior, bit-identical
code path). With accumulation_steps > 1 the default weighting is
pair-weighted (training.accumulate_pair_weighted, default true): micro-batch
i contributes loss_i * (P_i / P_NORM) and the accumulated gradient is divided
by (P_total / P_NORM) at step time, which equals the exact pooled-batch
gradient (linearity of differentiation; P_NORM cancels and exists only for
fp16-scaler conditioning). Set accumulate_pair_weighted: false for classic
1/K micro-averaging. EMA target update, LR schedule step, gradient clip and
logging all happen once per *optimizer* step, so accumulation_steps=K matches
a true batch-K run in update count, EMA cadence and schedule.
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def lr_for_step(base_lr: float, step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def _pair_count(out: dict) -> int:
    """JEPA pairs behind this micro-batch's loss (for pair-weighted scaling).

    Horizon/dynamics modes train on all (t, u>t) pairs (info["n"] length);
    1-step mode trains on the valid (t -> t+1) mask (already requires pixels
    on both sides). Aux pairs inherit the JEPA-pair weight (documented
    approximation: aux pairs are a subset of JEPA pairs).
    """
    hz = out.get("horizon")
    if hz is not None:
        return int(hz["n"].numel())
    valid = out.get("valid")
    return int(valid.sum().item()) if valid is not None else 0


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tot, n = 0.0, 0
    stds, ranks = [], []
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch)
        tot += out["loss"].item()
        n += 1
        stds.append(out["target_std"])
        ranks.append(out["target_eff_rank"])
    return {
        "loss": tot / max(n, 1),
        "target_std": sum(stds) / max(len(stds), 1),
        "target_eff_rank": sum(ranks) / max(len(ranks), 1),
    }


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: dict,
    device: torch.device,
    resume_opt_state: dict | None = None,
) -> dict[str, float]:
    tr = cfg["training"]
    ckpt_dir = tr.get("checkpoint_dir", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=tr.get("lr", 1e-4),
        weight_decay=tr.get("weight_decay", 0.01),
    )
    if resume_opt_state is not None:
        # G8/R11: continue with the resumed momentum instead of fresh.
        # State tensors are moved to the training device explicitly
        # (load_state_dict does not remap them); group mismatch raises LOUD.
        for s in resume_opt_state.get("state", {}).values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
        opt.load_state_dict(resume_opt_state)
        print(f"loaded optimizer state "
              f"({len(resume_opt_state.get('state', {}))} tensors; "
              f"schedule still restarts per config)")
    use_amp = tr.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_epochs = tr.get("max_epochs", 100)
    warmup_epochs = tr.get("warmup_epochs", 5)
    accum = max(1, int(tr.get("accumulation_steps", 1)))
    pair_w = bool(tr.get("accumulate_pair_weighted", True))
    # Conditioning constant only (cancels exactly in step math): keeps
    # micro-loss magnitudes O(1) for the AMP scaler when pair-weighting.
    p_norm = max(float(tr.get("accumulate_pair_norm", 32.0)), 1e-9)
    # Schedule counts *optimizer* steps, so accumulation stretches it to the
    # true batch-K cadence automatically.
    steps_per_epoch = max((len(train_loader) + accum - 1) // accum, 1)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = tr.get("lr", 1e-4)
    grad_clip = tr.get("grad_clip", 1.0)
    log_every = tr.get("log_every", 10)

    use_wandb = tr.get("log_wandb", False)
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(project=tr.get("project_name", "world-model-jepa"), config=cfg)

    global_step = 0
    best_val = float("inf")
    model.to(device)
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{max_epochs}")
        pending = 0       # micro-batches accumulated in the open window
        w_total = 0.0     # sum of micro weights (divisor at step time)
        loss_num = 0.0    # sum of weight*loss (pooled step loss numerator)
        pairs_total = 0   # JEPA pairs in the window (diagnostics only)
        last_out: dict | None = None

        def flush_step() -> None:
            """Optimizer step over the open window (EMA/schedule/clip/log)."""
            nonlocal global_step, best_val, pending, w_total, loss_num, \
                pairs_total, last_out
            if pending == 0:
                return
            assert last_out is not None
            for pg in opt.param_groups:
                pg["lr"] = lr_for_step(base_lr, global_step, warmup_steps, total_steps)
            scaler.unscale_(opt)
            if accum > 1 and w_total > 0:
                f = w_total
                for p in model.trainable_parameters():
                    if p.grad is not None:
                        p.grad.div_(f)
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            model.update_target()
            global_step += 1
            step_loss = loss_num / w_total if w_total > 0 else 0.0
            if global_step % log_every == 0:
                msg: dict = {"loss": step_loss,
                             "target_std": last_out["target_std"],
                             "target_eff_rank": last_out["target_eff_rank"],
                             "lr": opt.param_groups[0]["lr"],
                             "gnorm": gnorm.item(),
                             "pairs": float(pairs_total),
                             "micro": float(pending)}
                if "velocity_norm" in last_out:
                    msg["velocity_norm"] = last_out["velocity_norm"]
                aux = last_out.get("aux") or {}
                if aux.get("n_aux", 0):
                    msg["aux_flat"] = aux["flat"].item()
                    msg["aux_prog"] = aux["prog"].item()
                    msg["aux_resp"] = aux["resp"].item()
                pbar.set_postfix({k: f"{v:.4f}" for k, v in msg.items() if k != "lr"})
                if run is not None:
                    run.log({"train/" + k: v for k, v in msg.items()}, step=global_step)
            pending, w_total, loss_num, pairs_total = 0, 0.0, 0.0, 0

        for batch in pbar:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if pending == 0:
                opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
                loss = out["loss"]
            if pair_w:
                n_pairs = _pair_count(out)
                w_i = (n_pairs / p_norm) if accum > 1 else 1.0
            else:
                # Classic 1/K micro-averaging (uniform mode): weight 1.0 now,
                # divide by window size at step time.
                n_pairs, w_i = 0, 1.0
            scaler.scale(loss * w_i).backward()
            pending += 1
            w_total += w_i
            loss_num += w_i * loss.item()
            pairs_total += n_pairs
            last_out = out
            if pending >= accum:
                flush_step()
        flush_step()  # partial tail window at epoch end

        if val_loader is not None and (epoch + 1) % tr.get("eval_every", 1) == 0:
            stats = evaluate(model, val_loader, device)
            print(f"val epoch {epoch+1}: loss={stats['loss']:.4f} "
                  f"std={stats['target_std']:.4f} rank={stats['target_eff_rank']:.1f}")
            if run is not None:
                run.log({"val/" + k: v for k, v in stats.items()}, step=global_step)
            if stats["loss"] < best_val:
                best_val = stats["loss"]
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "opt": opt.state_dict(), "val_loss": best_val, "config": cfg},
                    os.path.join(ckpt_dir, "best.pt"),
                )
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "opt": opt.state_dict(), "config": cfg},
            os.path.join(ckpt_dir, "last.pt"),
        )

    if run is not None:
        run.finish()
    return {"best_val_loss": best_val}
