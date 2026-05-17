# DA6401 Assignment 3 — Transformer for Neural Machine Translation

**Course:** DA6401 Introduction to Deep Learning | IIT Madras  
**Task:** German → English Neural Machine Translation  
**Dataset:** Multi30k (29k train / 1,014 val / 1,000 test pairs)  
**Architecture:** "Attention Is All You Need" (Vaswani et al., 2017) — built from scratch in PyTorch

---

## Results

| Metric | Value |
|--------|-------|
| Test BLEU (greedy decode) | **~37** |
| Model parameters | ~10M |
| Training epochs | ~30 (early stopping) |
| Decoding | Greedy autoregressive |

---

## Project Structure

```
A3/
├── model.py              # Full Transformer architecture + infer()
├── train.py              # Training pipeline, loss, checkpointing, BLEU eval
├── dataset.py            # Multi30k loader, spaCy tokenization, Vocabulary
├── lr_scheduler.py       # Noam LR scheduler
├── config.py             # All hyperparameters
├── experiment_2_1.py     # Noam vs Fixed LR ablation
├── experiment_2_2.py     # Scaling factor 1/√dk ablation
├── experiment_2_3.py     # Attention heatmaps & head specialization
├── experiment_2_4.py     # Sinusoidal PE vs Learned embeddings
├── experiment_2_5.py     # Label smoothing ablation
└── checkpoints/          # Saved model weights
```

---

## Setup

```bash
# Clone repo
git clone https://github.com/ItsKowshik/DL_A3
cd da6401_assignment_3

# Create environment
conda create -n a3 python=3.11
conda activate a3

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn wandb datasets spacy tqdm evaluate sacrebleu gdown

# Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# Login to W&B
wandb login
```

---

## Hyperparameters (`config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `D_MODEL` | 256 | Embedding dimensionality |
| `N_LAYERS` | 3 | Encoder/decoder stack depth |
| `NUM_HEADS` | 8 | Attention heads |
| `D_FF` | 512 | FFN inner dimensionality |
| `DROPOUT` | 0.3 | Regularisation |
| `BATCH_SIZE` | 128 | |
| `WARMUP_STEPS` | 2000 | Noam warmup |
| `LABEL_SMOOTH` | 0.1 | ε for label smoothing |
| `WEIGHT_DECAY` | 1e-4 | Adam L2 regularisation |
| `PATIENCE` | 10 | Early stopping |
| `BEAM_SIZE` | 8 | Beam search (evaluation only) |
| `MAX_SEQ_LEN` | 256 | Max sequence length |

---

## Architecture (`model.py`)

Built entirely from scratch using `nn.Linear`, `nn.Module`, `nn.LayerNorm` — no pre-built attention modules.

### Components

**`scaled_dot_product_attention(Q, K, V, mask)`**  
Standard scaled dot-product: `softmax(QKᵀ / √dk) · V`. Masking sets future/pad positions to `-inf` before softmax.

**`MultiHeadAttention`**  
8 parallel attention heads. Projects Q/K/V independently per head, concatenates outputs, projects with W_O. `nn.MultiheadAttention` is NOT used.

**`PositionalEncoding`**  
Sinusoidal: `PE(pos,2i) = sin(pos/10000^(2i/d_model))`, `PE(pos,2i+1) = cos(...)`. Registered as a non-trainable buffer.

**`PositionwiseFeedForward`**  
`FFN(x) = max(0, xW₁+b₁)W₂+b₂` — two linear layers with ReLU.

**`EncoderLayer` / `DecoderLayer`**  
Post-LayerNorm (as in original paper). Decoder has 3 sub-layers: masked self-attn, cross-attn, FFN.

**`Encoder` / `Decoder`**  
Stacks of N=3 identical layers with final LayerNorm.

**`Transformer`**  
Full encoder-decoder. Supports two modes:
- **Training mode:** `Transformer(src_vocab_size=X, tgt_vocab_size=Y, checkpoint_path=None)`
- **Inference mode:** `Transformer()` — downloads weights from Google Drive, loads vocab + spaCy tokenizer automatically

### Key Methods (Autograder Contract)

```python
scaled_dot_product_attention(Q, K, V, mask) → (output, attn_weights)
MultiHeadAttention.forward(q, k, v, mask)   → Tensor
PositionalEncoding.forward(x)               → Tensor
make_src_mask(src, pad_idx)                 → BoolTensor  # [batch, 1, 1, src_len]
make_tgt_mask(tgt, pad_idx)                 → BoolTensor  # [batch, 1, tgt_len, tgt_len]
Transformer.encode(src, src_mask)           → Tensor      # [batch, src_len, d_model]
Transformer.decode(memory, src_mask, tgt, tgt_mask) → Tensor  # [batch, tgt_len, vocab]
Transformer.infer(src_sentence: str)        → str         # German → English
```

---

## Training (`train.py`)

**`LabelSmoothingLoss`** — smoothed target: `(1-ε)` for correct token, `ε/(V-1)` for others. PAD positions zeroed.

**`NoamScheduler`** — `lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))`

**`greedy_decode`** — autoregressive token-by-token generation (autograder contract).

**`evaluate_bleu`** — corpus-level BLEU via HuggingFace `evaluate` library.

**`save_checkpoint` / `load_checkpoint`** — embeds `src_vocab_stoi` and `tgt_vocab_stoi` directly in checkpoint for self-contained inference.

**`average_checkpoints`** — averages weights of top-5 checkpoints by val_loss for free +0.5 BLEU.

### Run Training

```bash
python train.py
```

Logs to W&B group `baseline`. Saves `checkpoints/best_model.pt` (best val_loss) and `checkpoints/avg_model.pt` (top-5 averaged).

---

## Inference

The autograder calls:
```python
model = Transformer().to(device)   # downloads weights from GDrive, loads vocab + spaCy
model.eval()
english = model.infer("Ein Hund spielt im Park.")
# → "a dog plays in the park ."
```

`Transformer.__init__()` with no args:
1. Downloads checkpoint from Google Drive via `gdown`
2. Loads model weights + vocabulary (embedded in checkpoint)
3. Loads spaCy `de_core_news_sm` tokenizer (auto-downloads if missing via `spacy.cli`)

```bash
# Test locally
python test_infer.py
```

---

## Dataset (`dataset.py`)

- **Source:** HuggingFace `bentrevett/multi30k`
- **Tokenization:** spaCy only (`de_core_news_sm` for German, `en_core_web_sm` for English)
- **Vocabulary:** Built from training split only (`min_freq=1`). Special tokens: `<unk>=0, <pad>=1, <sos>=2, <eos>=3`
- **Vocab sizes:** src ~7,853 | tgt ~5,893

```python
from dataset import get_dataloaders
train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=128)
```

---

## W&B Experiments

### 2.1 — Noam Scheduler vs Fixed LR
```bash
python experiment_2_1.py
```
Logs `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` (token-level), `lr`. Group: `noam-vs-fixed-lr`.

**Finding:** Noam scheduler converges faster and achieves higher BLEU (~37 vs ~35) due to warmup stabilising self-attention from random initialisation.

### 2.2 — Scaling Factor 1/√dk Ablation
```bash
python experiment_2_2.py
```
Logs `grad_norm_Q`, `grad_norm_K` for first 1,000 steps. Group: `scaling-ablation`.

**Finding:** Without scaling, Q/K gradient norms are higher and noisier (instability from softmax saturation). BLEU drops ~3 points without scaling.

### 2.3 — Attention Heatmaps & Head Specialization
```bash
python experiment_2_3.py
```
Logs heatmaps (one per head × 3 sentences) + head redundancy table. Group: `attention-analysis`.

**Finding:** 8 heads show clear specialization — Heads 2/5/8 redundant (37.5%), Head 4 BOS anchor, Head 6 diagonal/local, Head 7 semantic/content.

### 2.4 — Sinusoidal PE vs Learned Embeddings
```bash
python experiment_2_4.py
```
Logs `val_loss`, `val_bleu`. Group: `pe-ablation`.

**Finding:** Sinusoidal PE achieves higher BLEU (~37 vs ~34). Advantage: deterministic mathematical structure enables extrapolation to sequence lengths beyond training max_len via trigonometric addition formulas.

### 2.5 — Label Smoothing
```bash
python experiment_2_5.py
```
Logs `train_confidence`, `val_confidence`, `train_perplexity`, `val_perplexity`. Group: `label-smoothing`.

**Finding:** ε=0.1 → lower confidence (~0.63), higher perplexity (regularisation signal), better BLEU (+2 points). ε=0.0 → overconfident (~0.71), lower perplexity, worse generalisation.

---

## Architecture Decision: Post-LayerNorm

**Post-LayerNorm** (original paper §3.1): `x = LayerNorm(x + Sublayer(x))`

Chosen to faithfully reproduce the paper's exact specification for autograder compatibility. Instability from Post-LN without warmup is mitigated by the Noam schedule, which starts at near-zero LR and gradually increases — confirmed by successful convergence in all experiments.

---

## Submission

```bash
# Submission zip (code only — no checkpoint files)
zip submission.zip model.py train.py dataset.py lr_scheduler.py config.py

# Upload checkpoint to Google Drive, set _GDRIVE_ID in model.py
# Autograder downloads weights automatically via gdown
```

**W&B Report:** [Public Report Link](https://wandb.ai/k-indian-institute-of-technology-madras/A3/reports/DA6401-Assignment-3-ID25M803--VmlldzoxNjc2MzU3OQ?accessToken=o44f0zcxjqimxtg9fz0p577g63rf6rn42693m1e4fjp53nebo2q9xslqgm68d3uc)  
**GitHub:** [Repository](https://github.com/ItsKowshik/DL_A3)

---

## References

- Vaswani et al., "Attention Is All You Need," NeurIPS 2017. [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Elliott et al., "Multi30K: Multilingual English-German Image Descriptions," 2016.
- Michel et al., "Are Sixteen Heads Really Better than One?," NeurIPS 2019.

---

*Kowshik Arko Dey | id25m803 | DA6401 — IIT Madras | 2026*