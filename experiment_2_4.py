"""
experiment_2_4.py — Sinusoidal PE vs Learned Positional Embeddings
DA6401 Assignment 3 — W&B Report Section 2.4

Trains two models:
    Run A : Sinusoidal PE  (fixed, non-trainable, as in paper)
    Run B : Learned PE     (nn.Embedding, trainable)

Compares validation BLEU across epochs.
Group: "pe-ablation"

Theory check:
    Sinusoidal → can extrapolate beyond max training length
    Learned     → better fit on training lengths, fails to extrapolate
"""

import os
import math
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from model import (
    Transformer, make_src_mask, make_tgt_mask,
    PositionalEncoding, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, Encoder, Decoder,
    MultiHeadAttention
)
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
from train import (
    LabelSmoothingLoss, run_epoch,
    save_checkpoint, evaluate_bleu
)
import config


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL EMBEDDING MODULE
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings via nn.Embedding.

    Unlike sinusoidal PE:
      - Parameters are trained via backprop
      - Cannot generalise to lengths > max_len seen during training
      - No mathematical relationship between positions

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout after adding PE.
        max_len  (int)  : Max sequence length (must cover all train lengths).
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 256) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        # Trainable embedding table: one vector per position
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        seq_len  = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # positions: [1, seq_len]
        pe = self.pos_embed(positions)   # [1, seq_len, d_model]
        return self.dropout(x + pe)


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER WITH SWAPPABLE PE
# ══════════════════════════════════════════════════════════════════════

class TransformerWithPE(Transformer):
    """
    Transformer where PE type (sinusoidal vs learned) is configurable.
    Inherits all encode/decode/forward logic from base Transformer.
    Only overrides __init__ to swap the PE module.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        use_learned_pe: bool = False,
        max_len: int = 256,
    ) -> None:
        # Build base transformer first (sets all layers)
        nn.Module.__init__(self)

        self._d_model = d_model

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # ── Positional Encoding: sinusoidal OR learned ─────────────────
        if use_learned_pe:
            self.src_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            self.src_pe = PositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = PositionalEncoding(d_model, dropout, max_len)

        # Encoder & Decoder stacks (identical to base)
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model":        d_model,
            "N":              N,
            "num_heads":      num_heads,
            "d_ff":           d_ff,
            "dropout":        dropout,
        }

        # Xavier init
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
        return self.decode(self.encode(src, src_mask),
                           src_mask, tgt, tgt_mask)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(use_learned_pe: bool):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    group    = "pe-ablation"
    run_name = "learned-pe" if use_learned_pe else "sinusoidal-pe"
    pe_type  = "learned" if use_learned_pe else "sinusoidal"

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
            "pe_type":        pe_type,
        },
        reinit=True,
    )

    print(f"\n{'='*55}")
    print(f"Experiment 2.4 | {run_name}")
    print(f"{'='*55}")

    # ── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    # greedy_decode requires batch_size=1 — separate loader for BLEU
    _, val_loader_bleu, _, _, _ = get_dataloaders(batch_size=1)

    # ── Model ──────────────────────────────────────────────────────────
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
    print(f"Total params : {total_params:,}")
    print(f"PE params    : {pe_params:,}  ({'trainable' if use_learned_pe else 'fixed/0'})")
    wandb.config.update({"total_params": total_params, "pe_params": pe_params})

    # ── Optimizer / Scheduler / Loss ───────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config.D_MODEL,
        warmup_steps=config.WARMUP_STEPS
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────────────
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

        # ── Val BLEU every 5 epochs (expensive but informative) ────────
        val_bleu = None
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            val_bleu = evaluate_bleu(
                model, val_loader_bleu, tgt_vocab, device=device
            )
            print(f"Epoch {epoch} | train={train_loss:.4f} "
                  f"val={val_loss:.4f} val_bleu={val_bleu:.2f}")
        else:
            print(f"Epoch {epoch} | train={train_loss:.4f} "
                  f"val={val_loss:.4f}")

        log_dict = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }
        if val_bleu is not None:
            log_dict["val_bleu"] = val_bleu

        wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(ckpt_dir, "best_model.pt")
            )

    # ── Final test BLEU ────────────────────────────────────────────────
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU ({run_name}): {bleu:.2f}")
    wandb.run.summary["test_bleu"] = bleu
    wandb.log({"test_bleu": bleu, "epoch": config.NUM_EPOCHS})
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
    print("Compare: val_bleu + test_bleu for both runs")
    print("Theory: sinusoidal >= learned on this dataset (fixed length)")
    print("Extrapolation: sinusoidal handles unseen lengths, learned fails")