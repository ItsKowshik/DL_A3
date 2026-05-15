"""
experiment_2_2.py — Ablation: Scaling Factor 1/√dk
DA6401 Assignment 3 — W&B Report Section 2.2

Trains two models:
    Run A : WITH    scaling (standard: scores / √dk)
    Run B : WITHOUT scaling (raw dot products)

Logs gradient norms of Q and K weight matrices for first 1000 steps.
Group: "scaling-ablation"
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Optional, Tuple
from tqdm import tqdm

from model import (
    Transformer, make_src_mask, make_tgt_mask,
    MultiHeadAttention, EncoderLayer, DecoderLayer,
    Encoder, Decoder, PositionalEncoding, PositionwiseFeedForward
)
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, save_checkpoint, evaluate_bleu
import config


# ══════════════════════════════════════════════════════════════════════
#  PATCHED ATTENTION — optional scaling toggle
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention_ablation(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention with optional √dk scaling."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    if use_scaling:
        scores = scores / math.sqrt(d_k)   # standard paper formula

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


class MHAWithScalingToggle(nn.Module):
    """MultiHeadAttention with scaling toggle for ablation."""

    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.1, use_scaling: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.d_k          = d_model // num_heads
        self.use_scaling  = use_scaling
        self.W_q          = nn.Linear(d_model, d_model)
        self.W_k          = nn.Linear(d_model, d_model)
        self.W_v          = nn.Linear(d_model, d_model)
        self.W_o          = nn.Linear(d_model, d_model)
        self.dropout      = nn.Dropout(p=dropout)
        self.attn_weights = None

    def _split_heads(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, s, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        x, self.attn_weights = scaled_dot_product_attention_ablation(
            Q, K, V, mask, use_scaling=self.use_scaling
        )
        return self.W_o(self._merge_heads(x))


class EncoderLayerAblation(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scaling=True):
        super().__init__()
        self.self_attn = MHAWithScalingToggle(d_model, num_heads, dropout, use_scaling)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayerAblation(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scaling=True):
        super().__init__()
        self.self_attn  = MHAWithScalingToggle(d_model, num_heads, dropout, use_scaling)
        self.cross_attn = MHAWithScalingToggle(d_model, num_heads, dropout, use_scaling)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


def build_ablation_transformer(src_vocab_size, tgt_vocab_size, use_scaling=True):
    """Build Transformer with scaling toggle in every attention layer."""

    class AblationTransformer(Transformer):
        def __init__(self, src_vocab_size, tgt_vocab_size,
                     d_model, N, num_heads, d_ff, dropout, use_scaling):
            # Bypass Transformer.__init__ entirely — build manually
            # Avoids checkpoint download / inference mode setup
            nn.Module.__init__(self)
            self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
            self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
            self.src_pe    = PositionalEncoding(d_model, dropout)
            self.tgt_pe    = PositionalEncoding(d_model, dropout)
            enc_layer = EncoderLayerAblation(d_model, num_heads, d_ff, dropout, use_scaling)
            dec_layer = DecoderLayerAblation(d_model, num_heads, d_ff, dropout, use_scaling)
            self.encoder = Encoder(enc_layer, N)
            self.decoder = Decoder(dec_layer, N)
            self.fc_out  = nn.Linear(d_model, tgt_vocab_size)
            self.config  = {
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
                "d_model": d_model, "N": N,
                "num_heads": num_heads, "d_ff": d_ff, "dropout": dropout,
            }
            self._d_model = d_model
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

    return AblationTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        use_scaling=use_scaling,
    )


# ══════════════════════════════════════════════════════════════════════
#  GRADIENT NORM HELPER
# ══════════════════════════════════════════════════════════════════════

def get_qk_grad_norms(model):
    """Mean grad norm of W_q and W_k across all MHA layers."""
    q_norms, k_norms = [], []
    for module in model.modules():
        if isinstance(module, MHAWithScalingToggle):
            if module.W_q.weight.grad is not None:
                q_norms.append(module.W_q.weight.grad.norm().item())
            if module.W_k.weight.grad is not None:
                k_norms.append(module.W_k.weight.grad.norm().item())
    q = sum(q_norms) / len(q_norms) if q_norms else 0.0
    k = sum(k_norms) / len(k_norms) if k_norms else 0.0
    return q, k


GRAD_LOG_STEPS = 1000


# ══════════════════════════════════════════════════════════════════════
#  TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(use_scaling: bool):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    group    = "scaling-ablation"
    run_name = "with-scaling" if use_scaling else "without-scaling"

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
            "label_smooth": config.LABEL_SMOOTH,
            "use_scaling":  use_scaling,
        },
        reinit=True,
    )

    print(f"\n{'='*50}\nExperiment 2.2 | {run_name}\n{'='*50}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    model = build_ablation_transformer(
        len(src_vocab), len(tgt_vocab), use_scaling=use_scaling
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
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, group, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    global_step   = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for src, tgt in tqdm(train_loader, desc=f"Train {epoch} [{run_name}]"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_in).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss   = loss_fn(
                logits.contiguous().view(-1, len(tgt_vocab)),
                tgt_out.contiguous().view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            # ── Log Q/K grad norms BEFORE clipping (first 1000 steps) ─
            if global_step < GRAD_LOG_STEPS:
                q_norm, k_norm = get_qk_grad_norms(model)
                # Use step= parameter so W&B x-axis is step not epoch
                wandb.log({
                    "grad_norm_Q": q_norm,
                    "grad_norm_K": k_norm,
                }, step=global_step)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            non_pad       = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

        train_loss = total_loss / max(total_tokens, 1)

        # Validation
        model.eval()
        val_loss_total, val_tokens = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                src_mask = make_src_mask(src).to(device)
                tgt_mask = make_tgt_mask(tgt_in).to(device)
                logits = model(src, tgt_in, src_mask, tgt_mask)
                loss   = loss_fn(
                    logits.contiguous().view(-1, len(tgt_vocab)),
                    tgt_out.contiguous().view(-1)
                )
                non_pad = (tgt_out != PAD_IDX).sum().item()
                val_loss_total += loss.item() * non_pad
                val_tokens     += non_pad
        val_loss = val_loss_total / max(val_tokens, 1)

        print(f"Epoch {epoch} | train={train_loss:.4f} val={val_loss:.4f}")
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.src_vocab  = src_vocab
            model.tgt_vocab  = tgt_vocab
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
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
    run_experiment(use_scaling=True)
    run_experiment(use_scaling=False)

    print("\nExperiment 2.2 complete.")
    print("W&B → group 'scaling-ablation'")
    print("Key plots:")
    print("  1. grad_norm_Q + grad_norm_K vs step (first 1000) — overlay both runs")
    print("  2. train_loss + val_loss vs epoch — overlay both runs")
    print("  3. test_bleu bar chart")