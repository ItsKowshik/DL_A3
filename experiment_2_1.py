"""
experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate
DA6401 Assignment 3 — W&B Report Section 2.1

Trains the same Transformer twice:
    Run A : Noam warmup + inverse-sqrt decay  (as in paper)
    Run B : Fixed LR = 1e-4, no warmup

Both runs logged to group="noam-vs-fixed-lr" for overlay in W&B.
"""

import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
from A3.train import LabelSmoothingLoss, run_epoch, evaluate_bleu, save_checkpoint
import config


# ══════════════════════════════════════════════════════════════════════
#  SHARED TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(
    use_noam: bool,
    fixed_lr: float = 1e-4,
):
    """
    Train transformer with either Noam or fixed LR.

    Args:
        use_noam : True → Noam scheduler. False → fixed LR.
        fixed_lr : LR used when use_noam=False.
    """
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    group     = "noam-vs-fixed-lr"
    run_name  = "noam-scheduler" if use_noam else f"fixed-lr-{fixed_lr}"
    scheduler_type = "noam" if use_noam else "fixed"

    # ── W&B init ───────────────────────────────────────────────────────
    wandb.init(
        project=config.WANDB_PROJECT,
        group=group,
        name=run_name,
        config={
            "d_model":        config.D_MODEL,
            "N":              config.N_LAYERS,
            "num_heads":      config.NUM_HEADS,
            "d_ff":           config.D_FF,
            "dropout":        config.DROPOUT,
            "batch_size":     config.BATCH_SIZE,
            "num_epochs":     config.NUM_EPOCHS,
            "warmup_steps":   config.WARMUP_STEPS,
            "label_smooth":   config.LABEL_SMOOTH,
            "scheduler_type": scheduler_type,
            "fixed_lr":       None if use_noam else fixed_lr,
        },
        reinit=True,
    )

    # ── Data ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Experiment 2.1 | {run_name}")
    print(f"{'='*50}")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    # ── Model ──────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    ).to(device)

    # ── Optimizer ──────────────────────────────────────────────────────
    if use_noam:
        # Paper: base lr=1, Noam controls everything
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0,
            betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(
            optimizer,
            d_model=config.D_MODEL,
            warmup_steps=config.WARMUP_STEPS,
        )
    else:
        # Fixed LR: no warmup, constant throughout
        optimizer = torch.optim.Adam(
            model.parameters(), lr=fixed_lr,
            betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0  # constant multiplier
        )

    # ── Loss ───────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} | train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.6f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(ckpt_dir, "best_model.pt")
            )

    # ── BLEU ───────────────────────────────────────────────────────────
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU ({run_name}): {bleu:.2f}")
    wandb.run.summary["test_bleu"] = bleu
    wandb.log({"test_bleu": bleu, "epoch": config.NUM_EPOCHS})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run A: Noam scheduler (paper method)
    run_experiment(use_noam=True)

    # Run B: Fixed LR = 1e-4
    run_experiment(use_noam=False, fixed_lr=1e-4)

    print("\nExperiment 2.1 complete.")
    print("Go to W&B → group 'noam-vs-fixed-lr' → overlay train_loss + val_loss curves.")