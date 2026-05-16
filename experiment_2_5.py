"""
experiment_2_5.py — Decoder Sensitivity: Label Smoothing
DA6401 Assignment 3 — W&B Report Section 2.5

Trains two models:
    Run A : eps=0.1  (label smoothing — paper default)
    Run B : eps=0.0  (standard cross-entropy, no smoothing)

Logs per epoch:
    - train_loss / val_loss
    - train_confidence / val_confidence  softmax prob of correct token
    - train_perplexity / val_perplexity

Group: "label-smoothing"
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX
from lr_scheduler import NoamScheduler
from train import save_checkpoint, evaluate_bleu
import config



class LabelSmoothingLossTracked(nn.Module):
    """
    Label smoothing loss that also tracks prediction confidence.

    Confidence = mean softmax probability assigned to the correct token
    across all non-pad positions in the batch.

    eps=0.0 → standard cross-entropy (model becomes overconfident)
    eps=0.1 → smoothed distribution (regularised, less confident)
    """

    def __init__(self, vocab_size: int, pad_idx: int,
                 smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        self._conf_sum   = 0.0
        self._conf_count = 0

    def reset_stats(self):
        self._conf_sum   = 0.0
        self._conf_count = 0

    def get_mean_confidence(self) -> float:
        if self._conf_count == 0:
            return 0.0
        return self._conf_sum / self._conf_count

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs      = F.softmax(logits, dim=-1)
            true_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            pad_mask   = (target != self.pad_idx)
            if pad_mask.sum() > 0:
                self._conf_sum   += true_probs[pad_mask].mean().item()
                self._conf_count += 1

        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target,
                                   ignore_index=self.pad_idx,
                                   reduction="mean")

        smooth_dist = torch.full(logits.size(),
                                 self.smoothing / (self.vocab_size - 1),
                                 device=logits.device)
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0.0
        smooth_dist[(target == self.pad_idx)] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss      = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad   = (target != self.pad_idx).sum().clamp(min=1)
        return loss.sum() / non_pad




def run_epoch_tracked(data_iter, model, loss_fn, optimizer=None,
                      scheduler=None, is_train=True, device="cpu"):
    """Returns (avg_loss, perplexity, mean_confidence)."""
    model.train() if is_train else model.eval()
    loss_fn.reset_stats()

    total_loss, total_tokens = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        from tqdm import tqdm
        for src, tgt in tqdm(data_iter,
                             desc=f"{'Train' if is_train else 'Val  '}"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_in).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss   = loss_fn(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_out.contiguous().view(-1)
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            non_pad       = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 10))
    confidence = loss_fn.get_mean_confidence()
    return avg_loss, perplexity, confidence




def run_experiment(smoothing: float):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    group    = "label-smoothing"
    run_name = f"smoothing-{smoothing}"

    wandb.init(
        project=config.WANDB_PROJECT,
        group=group,
        name=run_name,
        config={
            "d_model":      config.D_MODEL,
            "N":            config.N_LAYERS,
            "num_heads":    config.NUM_HEADS,
            "d_ff":         config.D_FF,
            "dropout":      config.DROPOUT,
            "batch_size":   config.BATCH_SIZE,
            "num_epochs":   config.NUM_EPOCHS,
            "warmup_steps": config.WARMUP_STEPS,
            "smoothing":    smoothing,
        },
        reinit=True,
    )

    print(f"\n{'='*55}\nExperiment 2.5 | {run_name} (eps={smoothing})\n{'='*55}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    # Training mode — checkpoint_path=None avoids auto-download
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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config.D_MODEL,
        warmup_steps=config.WARMUP_STEPS,
    )
    loss_fn = LabelSmoothingLossTracked(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=smoothing,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_ppl, train_conf = run_epoch_tracked(
            train_loader, model, loss_fn,
            optimizer, scheduler, is_train=True, device=device,
        )
        val_loss, val_ppl, val_conf = run_epoch_tracked(
            val_loader, model, loss_fn,
            is_train=False, device=device,
        )

        print(f"Epoch {epoch:2d} | "
              f"train={train_loss:.4f} ppl={train_ppl:.2f} conf={train_conf:.4f} | "
              f"val={val_loss:.4f} ppl={val_ppl:.2f} conf={val_conf:.4f}")

        wandb.log({
            "epoch":            epoch,
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "train_perplexity": train_ppl,
            "val_perplexity":   val_ppl,
            # KEY METRIC for Section 2.5 — prediction confidence
            "train_confidence": train_conf,
            "val_confidence":   val_conf,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.src_vocab  = src_vocab
            model.tgt_vocab  = tgt_vocab
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  ✓ Best saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                wandb.run.summary["stopped_epoch"] = epoch
                break

    # Final test BLEU
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"\nTest BLEU ({run_name}): {bleu:.2f}")
    wandb.run.summary["test_bleu"]          = bleu
    wandb.run.summary["final_train_conf"]   = train_conf
    wandb.run.summary["final_val_conf"]     = val_conf
    wandb.run.summary["final_train_ppl"]    = train_ppl
    wandb.run.summary["final_val_ppl"]      = val_ppl
    wandb.log({"test_bleu": bleu, "epoch": epoch})
    wandb.finish()




if __name__ == "__main__":
    # Run A: label smoothing eps=0.1 (paper default)
    run_experiment(smoothing=0.1)

    # Run B: no smoothing — standard cross-entropy
    run_experiment(smoothing=0.0)

    print("\nExperiment 2.5 complete.")
    print("W&B → group 'label-smoothing'")
    print("Key plots:")
    print("  1. train_confidence + val_confidence overlay (both runs)")
    print("  2. train_perplexity overlay — smoothed has HIGHER perplexity")
    print("  3. val_loss + test_bleu comparison")
    print()
    print("Theory checks:")
    print("  eps=0.0  confidence ~0.7+, low perplexity, may overfit")
    print("  eps=0.1  confidence ~0.6-0.65, higher perplexity, better BLEU")