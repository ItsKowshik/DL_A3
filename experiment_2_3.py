"""
experiment_2_3.py — Attention Rollout & Head Specialization
DA6401 Assignment 3 — W&B Report Section 2.3

No training needed — loads best baseline checkpoint.
Extracts attention weights from LAST encoder layer.
Logs heatmaps as wandb.Image (matplotlib) — works with all wandb versions.
Group: "attention-analysis"
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
import os

from model import Transformer, make_src_mask
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from train import load_checkpoint
import config

BASELINE_CKPT = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
NUM_SENTENCES = 3


def extract_encoder_attn(model, src, src_mask):
    model.eval()
    with torch.no_grad():
        _ = model.encode(src, src_mask)
    attn = model.encoder.layers[-1].self_attn.attn_weights
    return attn[0].cpu().numpy()   # [num_heads, src_len, src_len]


def make_heatmap_fig(attn_matrix, tokens, title=""):
    """Single head heatmap as matplotlib figure → wandb.Image."""
    n   = len(tokens)
    fig, ax = plt.subplots(figsize=(max(4, n*0.5), max(4, n*0.5)))
    im = ax.imshow(attn_matrix, cmap="viridis", vmin=0,
                   vmax=attn_matrix.max(), aspect="auto")
    short = [t[:10] for t in tokens]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Keys (attended TO)")
    ax.set_ylabel("Queries (attending FROM)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def log_head_heatmaps(attn_weights, tokens, sentence_idx):
    """Log one heatmap per head as wandb.Image."""
    num_heads = attn_weights.shape[0]
    log_dict  = {}
    for h in range(num_heads):
        fig = make_heatmap_fig(
            attn_weights[h], tokens,
            title=f"Head {h+1} | Sentence {sentence_idx+1}"
        )
        log_dict[f"attn_heads/sentence_{sentence_idx+1}/head_{h+1}"] = \
            wandb.Image(fig)
        plt.close(fig)
    wandb.log(log_dict)
    print(f"  Logged {num_heads} heatmaps for sentence {sentence_idx+1}")


def log_specialization_heatmaps(attn_weights, tokens, interesting, sent_idx):
    """Log comparison heatmaps for the 4 most interesting heads."""
    labels = ["local", "next_tok", "long_range", "redundant"]
    log_dict = {}
    for label, h in zip(labels, interesting):
        fig = make_heatmap_fig(
            attn_weights[h], tokens,
            title=f"{label} — Head {h+1}"
        )
        log_dict[f"head_specialization/sentence_{sent_idx+1}/{label}_head{h+1}"] = \
            wandb.Image(fig)
        plt.close(fig)
    wandb.log(log_dict)


def analyze_heads(attn_weights):
    num_heads, seq_len, _ = attn_weights.shape
    stats = []
    for h in range(num_heads):
        mat  = attn_weights[h]
        diag = float(np.mean([mat[i, i] for i in range(seq_len)]))
        nxt  = float(np.mean([mat[i, i+1] for i in range(seq_len-1)])) \
               if seq_len > 1 else 0.0
        ent  = float(-np.sum(mat * np.log(mat + 1e-9), axis=-1).mean())
        lr   = float(max(
            (mat[i, j] for i in range(seq_len)
             for j in range(seq_len) if abs(i-j) > 3), default=0.0
        ))
        stats.append({"head": h+1,
                      "diagonal":   round(diag, 4),
                      "next_token": round(nxt,  4),
                      "entropy":    round(ent,  4),
                      "long_range": round(lr,   4)})
    return stats


def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=config.WANDB_PROJECT,
        group="attention-analysis",
        name="head-specialization",
        config={"checkpoint": BASELINE_CKPT},
    )

    print("Loading data...")
    _, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=1)

    print("Loading checkpoint...")
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
    load_checkpoint(BASELINE_CKPT, model)
    model.eval()
    print(f"Model loaded. Num heads = {config.NUM_HEADS}")

    all_head_stats = []

    for sent_idx, (src, tgt) in enumerate(test_loader):
        if sent_idx >= NUM_SENTENCES:
            break

        src      = src.to(device)
        src_mask = make_src_mask(src).to(device)

        src_tokens = []
        for idx in src[0].tolist():
            if idx in (SOS_IDX, PAD_IDX): continue
            if idx == EOS_IDX: break
            src_tokens.append(src_vocab.lookup_token(idx))

        if len(src_tokens) < 4:
            continue

        print(f"\nSentence {sent_idx+1}: {' '.join(src_tokens)}")

        attn = extract_encoder_attn(model, src, src_mask)
        n    = len(src_tokens)
        attn = attn[:, 1:n+1, 1:n+1]
        n    = min(attn.shape[1], n)
        attn = attn[:, :n, :n]
        toks = src_tokens[:n]

        # ── All-heads heatmaps ─────────────────────────────────────────
        log_head_heatmaps(attn, toks, sent_idx)

        # ── Head analysis ──────────────────────────────────────────────
        stats = analyze_heads(attn)
        all_head_stats.append(stats)

        for s in stats:
            wandb.log({
                f"head_stats/s{sent_idx+1}/head_{s['head']}/diagonal":   s["diagonal"],
                f"head_stats/s{sent_idx+1}/head_{s['head']}/next_token": s["next_token"],
                f"head_stats/s{sent_idx+1}/head_{s['head']}/entropy":    s["entropy"],
                f"head_stats/s{sent_idx+1}/head_{s['head']}/long_range": s["long_range"],
            })

        diag_h = max(stats, key=lambda x: x["diagonal"])["head"] - 1
        unif_h = max(stats, key=lambda x: x["entropy"])["head"] - 1
        lr_h   = max(stats, key=lambda x: x["long_range"])["head"] - 1
        nxt_h  = max(stats, key=lambda x: x["next_token"])["head"] - 1
        interesting = list(dict.fromkeys([diag_h, nxt_h, lr_h, unif_h]))[:4]

        # ── Specialization comparison heatmaps ─────────────────────────
        log_specialization_heatmaps(attn, toks, interesting, sent_idx)

        print(f"  Head stats:")
        for s in stats:
            print(f"    Head {s['head']:2d} | diag={s['diagonal']:.3f} "
                  f"next={s['next_token']:.3f} ent={s['entropy']:.3f} "
                  f"lr={s['long_range']:.3f}")
        print(f"  local={diag_h+1} next_tok={nxt_h+1} "
              f"long_range={lr_h+1} redundant={unif_h+1}")

    if all_head_stats:
        avg = []
        for h in range(config.NUM_HEADS):
            avg.append({
                "Head":       h+1,
                "Diagonal":   round(float(np.mean([s[h]["diagonal"]   for s in all_head_stats])), 4),
                "Next-token": round(float(np.mean([s[h]["next_token"] for s in all_head_stats])), 4),
                "Entropy":    round(float(np.mean([s[h]["entropy"]    for s in all_head_stats])), 4),
                "Long-range": round(float(np.mean([s[h]["long_range"] for s in all_head_stats])), 4),
            })
        wandb.log({
            "head_redundancy_table": wandb.Table(
                columns=["Head","Diagonal","Next-token","Entropy","Long-range"],
                data=[[s["Head"],s["Diagonal"],s["Next-token"],
                       s["Entropy"],s["Long-range"]] for s in avg]
            )
        })
        high_ent = [s["Head"] for s in avg
                    if s["Entropy"] > np.mean([x["Entropy"] for x in avg])]
        print(f"\nRedundant heads (above-avg entropy): {high_ent}")
        wandb.run.summary["potentially_redundant_heads"] = str(high_ent)

    wandb.finish()
    print("\nExperiment 2.3 complete.")


if __name__ == "__main__":
    run_experiment()