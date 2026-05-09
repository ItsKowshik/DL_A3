"""
experiment_2_3.py — Attention Rollout & Head Specialization
DA6401 Assignment 3 — W&B Report Section 2.3

No training needed — loads best baseline checkpoint.
Extracts attention weights from LAST encoder layer.
Logs one heatmap per head to W&B as images.
Analyzes head specialization patterns.

Group: "attention-analysis"
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb
import os

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from A3.train import load_checkpoint
import config


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

BASELINE_CKPT = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

# Pick a few diverse test sentences for richer analysis
# (indices into test set — we pick manually after seeing them)
NUM_SENTENCES = 3


# ══════════════════════════════════════════════════════════════════════
#  HOOK: extract attention weights from last encoder layer
# ══════════════════════════════════════════════════════════════════════

def extract_encoder_attn(model, src, src_mask, device):
    """
    Run encoder forward pass and return attention weights from
    every head in the LAST encoder layer.

    Returns:
        attn_weights : np.ndarray [num_heads, src_len, src_len]
        tokens       : list of token strings
    """
    model.eval()
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    # Last encoder layer stores attn_weights after forward
    last_layer = model.encoder.layers[-1]
    # shape: [1, num_heads, src_len, src_len]
    attn = last_layer.self_attn.attn_weights

    return attn[0].cpu().numpy()   # [num_heads, src_len, src_len]


# ══════════════════════════════════════════════════════════════════════
#  HEATMAP PLOTTING
# ══════════════════════════════════════════════════════════════════════

def plot_head_heatmaps(attn_weights, tokens, sentence_idx):
    """
    Plot one heatmap per attention head.

    Args:
        attn_weights : [num_heads, seq_len, seq_len]
        tokens       : list of token strings (length = seq_len)
        sentence_idx : int — for figure title

    Returns:
        fig : matplotlib Figure with all heads as subplots
    """
    num_heads = attn_weights.shape[0]
    ncols     = 4
    nrows     = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for h in range(num_heads):
        ax  = axes[h]
        mat = attn_weights[h]   # [seq_len, seq_len]

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=mat.max(),
                       aspect="auto")
        ax.set_title(f"Head {h+1}", fontsize=10, fontweight="bold")

        # Token labels — truncate long tokens
        short_toks = [t[:8] for t in tokens]
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(short_toks, rotation=90, fontsize=7)
        ax.set_yticklabels(short_toks, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f"Last Encoder Layer — All Heads | Sentence {sentence_idx+1}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_head_comparison(attn_weights, tokens, interesting_heads):
    """
    Side-by-side comparison of specific heads for report.
    interesting_heads: list of head indices to compare.
    """
    n   = len(interesting_heads)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]

    labels = {
        "diagonal":    "Local/Diagonal (attends to self)",
        "next_token":  "Next-token head",
        "long_range":  "Long-range head",
        "uniform":     "Redundant/Uniform head",
    }

    for i, h in enumerate(interesting_heads):
        ax  = axes[i]
        mat = attn_weights[h]
        im  = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max(),
                        aspect="auto")
        ax.set_title(f"Head {h+1}", fontsize=11, fontweight="bold")
        short = [t[:8] for t in tokens]
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(short, rotation=90, fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        plt.colorbar(im, ax=ax)

    fig.suptitle("Head Specialization Comparison", fontsize=12)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  HEAD ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze_heads(attn_weights):
    """
    Compute simple statistics to characterise each head.

    Returns dict with per-head metrics:
        - diagonal_score : mean attention on diagonal (local/self)
        - next_tok_score : mean attention on diagonal+1 (next-token)
        - entropy        : attention entropy (high = uniform/redundant)
        - max_off_diag   : max attention outside diagonal (long-range)
    """
    num_heads, seq_len, _ = attn_weights.shape
    stats = []

    for h in range(num_heads):
        mat = attn_weights[h]   # [seq_len, seq_len]

        # Diagonal = self-attention
        diag_score = np.mean([mat[i, i] for i in range(seq_len)])

        # Next-token = super-diagonal
        next_scores = [mat[i, i+1] for i in range(seq_len-1)]
        next_score  = np.mean(next_scores) if next_scores else 0.0

        # Entropy (higher = more uniform = more redundant)
        eps  = 1e-9
        ent  = -np.sum(mat * np.log(mat + eps), axis=-1).mean()

        # Long-range: max attention weight > 3 positions away
        long_range = 0.0
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > 3:
                    long_range = max(long_range, mat[i, j])

        stats.append({
            "head":          h + 1,
            "diagonal":      round(float(diag_score), 4),
            "next_token":    round(float(next_score),  4),
            "entropy":       round(float(ent),         4),
            "long_range":    round(float(long_range),  4),
        })

    return stats


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=config.WANDB_PROJECT,
        group="attention-analysis",
        name="head-specialization",
        config={"checkpoint": BASELINE_CKPT},
    )

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    _, _, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=1)

    # ── Load baseline model ────────────────────────────────────────────
    print("Loading checkpoint...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        N=config.N_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    ).to(device)
    load_checkpoint(BASELINE_CKPT, model)
    model.eval()

    print(f"Model loaded. Num heads = {config.NUM_HEADS}")

    # ── Process test sentences ─────────────────────────────────────────
    all_head_stats = []

    for sent_idx, (src, tgt) in enumerate(test_loader):
        if sent_idx >= NUM_SENTENCES:
            break

        src = src.to(device)
        src_mask = make_src_mask(src).to(device)

        # Decode source tokens (strip <sos>/<eos>/<pad>)
        src_tokens = []
        for idx in src[0].tolist():
            if idx in (SOS_IDX, PAD_IDX):
                continue
            if idx == EOS_IDX:
                break
            src_tokens.append(src_vocab.lookup_token(idx))

        if len(src_tokens) < 4:
            continue   # skip very short sentences

        print(f"\nSentence {sent_idx+1}: {' '.join(src_tokens)}")

        # ── Extract attention ──────────────────────────────────────────
        attn_weights = extract_encoder_attn(model, src, src_mask, device)
        # attn_weights: [num_heads, full_src_len, full_src_len]

        # Trim to actual token positions (excluding padding)
        n_tok = len(src_tokens)
        # +1 for <sos> offset (src includes <sos>)
        attn_trimmed = attn_weights[:, 1:n_tok+1, 1:n_tok+1]

        # Recheck shape — handle edge cases
        actual_len = min(attn_trimmed.shape[1], len(src_tokens))
        attn_trimmed = attn_trimmed[:, :actual_len, :actual_len]
        tokens_used  = src_tokens[:actual_len]

        # ── Heatmap: all heads ─────────────────────────────────────────
        fig_all = plot_head_heatmaps(attn_trimmed, tokens_used, sent_idx)
        wandb.log({
            f"attn_all_heads/sentence_{sent_idx+1}":
                wandb.Image(fig_all,
                            caption=f"All heads | {' '.join(tokens_used[:6])}...")
        })
        plt.close(fig_all)

        # ── Head analysis ──────────────────────────────────────────────
        stats = analyze_heads(attn_trimmed)
        all_head_stats.append(stats)

        # Log per-head stats
        for s in stats:
            wandb.log({
                f"head_stats/sentence_{sent_idx+1}/head_{s['head']}/diagonal":   s["diagonal"],
                f"head_stats/sentence_{sent_idx+1}/head_{s['head']}/next_token": s["next_token"],
                f"head_stats/sentence_{sent_idx+1}/head_{s['head']}/entropy":    s["entropy"],
                f"head_stats/sentence_{sent_idx+1}/head_{s['head']}/long_range": s["long_range"],
            })

        # ── Identify interesting heads for comparison plot ─────────────
        # Most diagonal (local), highest entropy (uniform/redundant),
        # highest long-range
        diag_head     = max(stats, key=lambda x: x["diagonal"])["head"] - 1
        uniform_head  = max(stats, key=lambda x: x["entropy"])["head"] - 1
        longrange_head= max(stats, key=lambda x: x["long_range"])["head"] - 1
        next_tok_head = max(stats, key=lambda x: x["next_token"])["head"] - 1

        interesting = list(dict.fromkeys(
            [diag_head, next_tok_head, longrange_head, uniform_head]
        ))[:4]

        fig_cmp = plot_head_comparison(attn_trimmed, tokens_used, interesting)
        wandb.log({
            f"head_comparison/sentence_{sent_idx+1}":
                wandb.Image(fig_cmp,
                            caption=f"Specialised heads | sentence {sent_idx+1}")
        })
        plt.close(fig_cmp)

        # Print analysis
        print(f"  Head stats (sentence {sent_idx+1}):")
        for s in stats:
            print(f"    Head {s['head']:2d} | "
                  f"diag={s['diagonal']:.3f}  "
                  f"next={s['next_token']:.3f}  "
                  f"entropy={s['entropy']:.3f}  "
                  f"longrange={s['long_range']:.3f}")

        print(f"  → Most local   : Head {diag_head+1}")
        print(f"  → Next-token   : Head {next_tok_head+1}")
        print(f"  → Long-range   : Head {longrange_head+1}")
        print(f"  → Most uniform : Head {uniform_head+1}")

    # ── Head redundancy table (averaged over sentences) ────────────────
    if all_head_stats:
        avg_stats = []
        for h in range(config.NUM_HEADS):
            avg_diag = np.mean([s[h]["diagonal"]   for s in all_head_stats])
            avg_next = np.mean([s[h]["next_token"] for s in all_head_stats])
            avg_ent  = np.mean([s[h]["entropy"]    for s in all_head_stats])
            avg_lr   = np.mean([s[h]["long_range"] for s in all_head_stats])
            avg_stats.append({
                "Head":       h + 1,
                "Diagonal":   round(float(avg_diag), 4),
                "Next-token": round(float(avg_next), 4),
                "Entropy":    round(float(avg_ent),  4),
                "Long-range": round(float(avg_lr),   4),
            })

        wandb.log({
            "head_redundancy_table": wandb.Table(
                columns=["Head", "Diagonal", "Next-token", "Entropy", "Long-range"],
                data=[[s["Head"], s["Diagonal"], s["Next-token"],
                       s["Entropy"], s["Long-range"]] for s in avg_stats]
            )
        })

        # Heads with similar high entropy → likely redundant
        high_entropy = [s["Head"] for s in avg_stats
                        if s["Entropy"] > np.mean([x["Entropy"] for x in avg_stats])]
        print(f"\nPotentially redundant heads (above-avg entropy): {high_entropy}")
        wandb.run.summary["potentially_redundant_heads"] = str(high_entropy)

    wandb.finish()
    print("\nExperiment 2.3 complete.")
    print("W&B → group 'attention-analysis' → Images tab for heatmaps")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_experiment()