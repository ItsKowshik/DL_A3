"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from evaluate import load as load_metric

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX,
    UNK_IDX, Vocabulary
)
from lr_scheduler import NoamScheduler
import config


# ══════════════════════════════════════════════════════════════════════
# ❶  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    PAD positions receive 0 probability (excluded from loss).

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Build smoothed distribution
        # Start with smoothing / (V-1) everywhere
        smooth_dist = torch.full(
            logits.size(), self.smoothing / (self.vocab_size - 1),
            device=logits.device
        )

        # True label position gets confidence = 1 - eps
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # PAD positions → 0 probability (not counted in loss)
        smooth_dist[:, self.pad_idx] = 0.0
        pad_mask = (target == self.pad_idx)
        smooth_dist[pad_mask] = 0.0

        # KL-divergence: sum( y_smooth * log(y_smooth / p) )
        # = -sum( y_smooth * log_softmax(logits) )  [ignoring constant]
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Mean over non-pad tokens
        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
# ❷  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss.
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, backward + scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(
            tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}")
        ):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher forcing: decoder input = tgt[:-1], target = tgt[1:]
            tgt_in  = tgt[:, :-1]   # drop last  (<eos>)
            tgt_out = tgt[:, 1:]    # drop first (<sos>)

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_in).to(device)

            # Forward pass
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Flatten for loss
            batch_size, tgt_len, vocab_size = logits.size()
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            tgt_out_flat = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_out_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Log LR to W&B every 50 steps
                if batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log({"lr": current_lr,
                               "train_loss_step": loss.item()})

            # Accumulate (weight by non-pad token count)
            non_pad = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
# ❸  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # Encode source once
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

        # Start with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)

            # Decode one step
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, cur_len, tgt_vocab]

            # Greedy: pick highest prob at last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # next_token: [1, 1]

            ys = torch.cat([ys, next_token], dim=1)

            # Stop at <eos>
            if next_token.item() == end_symbol:
                break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
# ❸b  BEAM SEARCH DECODING
# ══════════════════════════════════════════════════════════════════════

def beam_search_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
    beam_size: int = 5,
) -> torch.Tensor:
    """
    Beam search decoding — correct implementation.

    Key fix vs naive version:
        - Prune by RAW cumulative log_prob (NOT length-normalised)
        - Only apply length normalisation at FINAL selection
        - Normalising during pruning biases toward short seqs mid-search
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

        # Each beam: (raw_cumulative_log_prob, sequence_tensor)
        beams     = [(0.0, torch.tensor([[start_symbol]],
                                        dtype=torch.long, device=device))]
        completed = []

        for _ in range(max_len - 1):
            all_candidates = []

            for cum_lp, seq in beams:
                # Beam already finished
                if seq[0, -1].item() == end_symbol:
                    completed.append((cum_lp, seq))
                    continue

                tgt_mask  = make_tgt_mask(seq).to(device)
                logits    = model.decode(memory, src_mask, seq, tgt_mask)
                # log probs at last position: [vocab_size]
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)

                # Expand: top beam_size tokens per beam
                topk_lp, topk_tok = log_probs.topk(beam_size)

                for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
                    new_seq = torch.cat(
                        [seq, torch.tensor([[tok]], dtype=torch.long,
                                           device=device)], dim=1
                    )
                    new_cum_lp = cum_lp + lp
                    all_candidates.append((new_cum_lp, new_seq))

            if not all_candidates:
                break

            # ── Prune by RAW cumulative log_prob (NOT normalised) ──────
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

            # Move finished beams to completed pool
            still_active = []
            for cum_lp, seq in beams:
                if seq[0, -1].item() == end_symbol:
                    completed.append((cum_lp, seq))
                else:
                    still_active.append((cum_lp, seq))
            beams = still_active

            if not beams:
                break

        # Fallback: if nothing completed, use active beams
        pool = completed if completed else beams

        # ── Final selection: length-normalised score ───────────────────
        # Divide by (length - 1) to exclude <sos> from length count
        best = max(pool,
                   key=lambda x: x[0] / max(x[1].size(1) - 1, 1))
        return best[1]   # [1, out_len]



# ══════════════════════════════════════════════════════════════════════
# ❹  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    bleu_metric = load_metric("bleu")
    model.eval()

    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src).to(device)

            # Beam search decode (beam_size=5)
            ys = beam_search_decode(
                model, src, src_mask,
                max_len=max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
                beam_size=config.BEAM_SIZE,
            )

            # Decode predicted tokens (strip <sos>, stop at <eos>)
            pred_tokens = []
            for idx in ys[0, 1:].tolist():   # skip <sos>
                if idx == EOS_IDX:
                    break
                tok = tgt_vocab.lookup_token(idx)
                if tok not in ("<unk>", "<pad>"):
                    pred_tokens.append(tok)

            # Decode reference tokens (strip <sos> and <eos>)
            ref_tokens = []
            for idx in tgt[0].tolist():
                if idx in (SOS_IDX, PAD_IDX):
                    continue
                if idx == EOS_IDX:
                    break
                tok = tgt_vocab.lookup_token(idx)
                ref_tokens.append(tok)

            if pred_tokens:   # skip empty predictions
                predictions.append(" ".join(pred_tokens))
                references.append([" ".join(ref_tokens)])  # BLEU expects list-of-refs

    result = bleu_metric.compute(predictions=predictions, references=references)
    # result["bleu"] is 0-1; multiply by 100 for standard BLEU
    bleu_score = result["bleu"] * 100
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Saved dict keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config":         model.config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) from disk.

    Args:
        path      : Checkpoint file path.
        model     : Transformer with matching architecture.
        optimizer : Optimizer to restore (None to skip).
        scheduler : Scheduler to restore (None to skip).

    Returns:
        epoch : Epoch at which checkpoint was saved.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
# ❻  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(
    group: str = "baseline",
    run_name: str = "baseline-run",
) -> None:
    """
    Full training experiment with W&B logging.

    Args:
        group    : W&B run group (e.g. 'baseline', 'noam-vs-fixed-lr',
                   'scaling-ablation', 'pe-ablation', 'label-smoothing').
        run_name : Human-readable name for this specific run.

    Steps:
        1. Init W&B
        2. Build dataset / vocabs
        3. Create DataLoaders
        4. Instantiate Transformer
        5. Instantiate Adam (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler
        7. Instantiate LabelSmoothingLoss
        8. Training loop with checkpointing
        9. Final BLEU on test set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── 1. Init W&B ────────────────────────────────────────────────────
    wandb.init(
        project=config.WANDB_PROJECT,
        group=group,          # groups runs together in W&B UI
        name=run_name,        # label for this individual run
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
            "group":        group,
        }
    )

    # ── 2 & 3. Dataset + DataLoaders ──────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.BATCH_SIZE)

    print(f"Src vocab: {len(src_vocab)}  |  Tgt vocab: {len(tgt_vocab)}")

    # ── 4. Model ───────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")
    wandb.config.update({"total_params": total_params})

    # ── 5. Optimizer (paper: β1=0.9, β2=0.98, ε=1e-9) ────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,           # base lr=1 → Noam scale controls actual LR
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4,  # L2 regularisation — helps generalisation
    )

    # ── 6. Noam Scheduler ─────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model=config.D_MODEL,
        warmup_steps=config.WARMUP_STEPS,
    )

    # ── 7. Loss ────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config.LABEL_SMOOTH,
    )

    # ── 8. Training loop with early stopping ─────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    best_path        = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{config.NUM_EPOCHS} ──")

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
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

        # ── Save best checkpoint ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  ✓ Best saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.PATIENCE})")

        # ── Early stopping ─────────────────────────────────────────────
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            wandb.run.summary["stopped_epoch"] = epoch
            break

    # ── 9. Load best → final BLEU with beam search ────────────────────
    print("\nLoading best checkpoint...")
    load_checkpoint(best_path, model)
    model.to(device)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU (beam={config.BEAM_SIZE}): {bleu:.2f}")

    wandb.run.summary["test_bleu"] = bleu
    wandb.log({"test_bleu": bleu, "epoch": epoch})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_training_experiment(
        group="baseline",
        run_name="improved-v2",
    )