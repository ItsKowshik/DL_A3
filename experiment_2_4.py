"""
experiment_2_4.py — Positional Encoding vs Learned Embeddings
DA6401 Assignment 3 — W&B Report Section 2.4

Trains two models:
    Run A : Sinusoidal PE  (fixed, non-trainable, as in paper)
    Run B : Learned PE     (nn.Embedding, trainable)

Logs val_loss per epoch + final test BLEU.
Group: "pe-ablation"
"""

import os
import math
import torch
import torch.nn as nn
import wandb

from model import (
    Transformer, make_src_mask, make_tgt_mask,
    PositionalEncoding, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, Encoder, Decoder,
)
from dataset import get_dataloaders, PAD_IDX
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, run_epoch, save_checkpoint, evaluate_bleu
import config


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL EMBEDDING MODULE
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings via nn.Embedding.
    Trained via backprop — cannot generalise beyond max_len seen in training.
    """
    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 256) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER WITH SWAPPABLE PE
# ══════════════════════════════════════════════════════════════════════

class TransformerWithPE(Transformer):
    """Transformer where PE type is configurable. Bypasses parent __init__."""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, N=3, num_heads=8, d_ff=512,
                 dropout=0.3, use_learned_pe=False, max_len=256):
        nn.Module.__init__(self)
        self._d_model  = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        if use_learned_pe:
            self.src_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            self.src_pe = PositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model, "N": N,
            "num_heads": num_heads, "d_ff": d_ff, "dropout": dropout,
        }

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.src_pe(self.src_embed(src) * math.sqrt(self._d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self._d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(use_learned_pe: bool):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    group    = "pe-ablation"
    run_name = "learned-pe" if use_learned_pe else "sinusoidal-pe"

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
            "pe_type":        "learned" if use_learned_pe else "sinusoidal",
        },
        reinit=True,
    )

    print(f"\n{'='*55}\nExperiment 2.4 | {run_name}\n{'='*55}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    model = TransformerWithPE(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        use_learned_pe=use_learned_pe,
        max_len=config.MAX_SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pe_params    = sum(p.numel() for p in model.src_pe.parameters()) * 2
    print(f"Total params: {total_params:,}  |  PE params: {pe_params:,}")
    wandb.config.update({"total_params": total_params, "pe_params": pe_params})

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config.D_MODEL,
        warmup_steps=config.WARMUP_STEPS,
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best_val_loss    = float("inf")
    patience_counter = 0

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
        print(f"Epoch {epoch:2d} | train={train_loss:.4f} "
              f"val={val_loss:.4f} lr={current_lr:.6f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
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

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU ({run_name}): {bleu:.2f}")
    wandb.run.summary["test_bleu"] = bleu
    wandb.log({"test_bleu": bleu, "epoch": epoch})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run A: sinusoidal PE (paper method)
    run_experiment(use_learned_pe=False)

    # Run B: learned positional embeddings
    run_experiment(use_learned_pe=True)

    print("\nExperiment 2.4 complete.")
    print("W&B → group 'pe-ablation'")
    print("Key plots:")
    print("  1. val_loss overlay — both runs")
    print("  2. test_bleu bar chart")
    print("Theory: learned PE may match sinusoidal on fixed-length Multi30k,")
    print("but sinusoidal extrapolates to longer sequences at inference.")