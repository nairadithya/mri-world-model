"""Pure-PyTorch training loop: AdamW + cosine/warmup + AMP + EMA + wandb."""
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
) -> dict[str, float]:
    tr = cfg["training"]
    ckpt_dir = tr.get("checkpoint_dir", "checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=tr.get("lr", 1e-4),
        weight_decay=tr.get("weight_decay", 0.01),
    )
    use_amp = tr.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_epochs = tr.get("max_epochs", 100)
    warmup_epochs = tr.get("warmup_epochs", 5)
    steps_per_epoch = max(len(train_loader), 1)
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
        for batch in pbar:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            for pg in opt.param_groups:
                pg["lr"] = lr_for_step(base_lr, global_step, warmup_steps, total_steps)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            model.update_target()
            global_step += 1
            if global_step % log_every == 0:
                msg = {"loss": loss.item(), "target_std": out["target_std"],
                       "target_eff_rank": out["target_eff_rank"], "lr": opt.param_groups[0]["lr"]}
                aux = out.get("aux") or {}
                if aux.get("n_aux", 0):
                    msg["aux_flat"] = aux["flat"].item()
                    msg["aux_prog"] = aux["prog"].item()
                    msg["aux_resp"] = aux["resp"].item()
                pbar.set_postfix({k: f"{v:.4f}" for k, v in msg.items() if k != "lr"})
                if run is not None:
                    run.log({"train/" + k: v for k, v in msg.items()}, step=global_step)

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
