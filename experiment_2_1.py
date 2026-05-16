"""
experiment_2_1.py — Noam Scheduler vs Fixed Learning Rate
DA6401 Assignment 3 — W&B Report Section 2.1

Trains the same Transformer twice:
    Run A : Noam warmup + inverse-sqrt decay  (as in paper)
    Run B : Fixed LR = 1e-4, no warmup

Logs per epoch:
    - train_loss / val_loss
    - train_accuracy / val_accuracy  ← token-level (correct non-pad predictions)
    - lr vs step (warmup curve vs flat line)

Group: "noam-vs-fixed-lr"
"""

import os
import torch
import torch.nn as nn
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, evaluate_bleu, save_checkpoint
import config
from tqdm import tqdm



def run_epoch_with_acc(data_iter, model, loss_fn, optimizer=None,
                       scheduler=None, is_train=True, device="cpu"):
    """
    One epoch of training or evaluation.
    Returns (avg_loss, token_accuracy).

    Token accuracy = % of non-pad decoder tokens predicted correctly.
    Fast — just argmax, no beam search.
    """
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_tokens  = 0
    correct_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for src, tgt in tqdm(data_iter,
                             desc=f"{'Train' if is_train else 'Val  '}",
                             leave=False):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_in).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [batch, tgt_len, vocab_size]

            loss = loss_fn(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_out.contiguous().view(-1)
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Token-level accuracy
            with torch.no_grad():
                preds   = logits.argmax(dim=-1)          # [batch, tgt_len]
                non_pad = tgt_out != PAD_IDX             # [batch, tgt_len]
                correct_tokens += (preds == tgt_out)[non_pad].sum().item()
                total_tokens   += non_pad.sum().item()

            non_pad_count  = non_pad.sum().item()
            total_loss    += loss.item() * non_pad_count

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct_tokens / max(total_tokens, 1)
    return avg_loss, accuracy



def run_experiment(use_noam: bool, fixed_lr: float = 1e-4):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    group    = "noam-vs-fixed-lr"
    run_name = "noam-scheduler" if use_noam else f"fixed-lr-{fixed_lr}"

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
            "scheduler_type": "noam" if use_noam else "fixed",
            "fixed_lr":       None if use_noam else fixed_lr,
        },
        reinit=True,
    )

    print(f"\n{'='*55}\nExperiment 2.1 | {run_name}\n{'='*55}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        checkpoint_path=None,
    ).to(device)

    if use_noam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = NoamScheduler(
            optimizer, d_model=config.D_MODEL,
            warmup_steps=config.WARMUP_STEPS,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=fixed_lr,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    ckpt_dir  = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch_with_acc(
            train_loader, model, loss_fn,
            optimizer, scheduler, is_train=True, device=device,
        )
        val_loss, val_acc = run_epoch_with_acc(
            val_loader, model, loss_fn,
            is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={current_lr:.6f}")

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "train_accuracy": train_acc,   # token-level accuracy
            "val_accuracy":   val_acc,     # KEY: explicit validation accuracy curve
            "lr":             current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.src_vocab  = src_vocab
            model.tgt_vocab  = tgt_vocab
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  ✓ Best saved")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                wandb.run.summary["stopped_epoch"] = epoch
                break

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU ({run_name}): {bleu:.2f}")
    wandb.run.summary["test_bleu"]  = bleu
    wandb.run.summary["final_val_accuracy"] = val_acc
    wandb.log({"test_bleu": bleu, "epoch": epoch})
    wandb.finish()


if __name__ == "__main__":
    run_experiment(use_noam=True)
    run_experiment(use_noam=False, fixed_lr=1e-4)

    print("\nExperiment 2.1 complete.")
    print("W&B → group 'noam-vs-fixed-lr'")
    print("Key plots:")
    print("  1. train_loss + val_loss overlay")
    print("  2. train_accuracy + val_accuracy overlay to validation accuracy curves")
    print("  3. lr vs step — warmup curve vs flat line")