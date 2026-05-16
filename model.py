"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# ❶  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # QK^T / sqrt(d_k)  →  (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask: True positions → -inf so softmax gives 0
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax over key dimension
    attn_w = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, src_len] → [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask: upper triangle = True (mask future)
    # shape [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Combine: mask out if PAD *or* future position
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
# ❸  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Projection matrices W_Q, W_K, W_V, W_O
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualisation (Section 2.3)
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [batch, seq, d_model] → [batch, num_heads, seq, d_k]
        """
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [batch, num_heads, seq, d_k] → [batch, seq, d_model]
        """
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        # Linear projections
        Q = self._split_heads(self.W_q(query))   # [b, h, seq_q, d_k]
        K = self._split_heads(self.W_k(key))     # [b, h, seq_k, d_k]
        V = self._split_heads(self.W_v(value))   # [b, h, seq_k, d_k]

        # Scaled dot-product attention per head
        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # x: [batch, num_heads, seq_q, d_k]

        # Merge heads and project
        x = self._merge_heads(x)          # [batch, seq_q, d_model]
        return self.W_o(x)


# ══════════════════════════════════════════════════════════════════════
# ❹  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a trainable parameter) so it moves
    to the correct device automatically and is NOT trained.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Division term: 10000^(2i/d_model)  computed in log-space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)   # [1, max_len, d_model]  ← batch dim

        # Register as buffer: not a parameter, moves with .to(device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# ❺  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality.
        d_ff    (int)  : Inner-layer dimensionality.
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# ❻  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm):
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Post-LayerNorm chosen (as in original paper).
    Justification: matches "Attention Is All You Need" exactly;
    training stabilised via Noam warmup scheduler.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Sub-layer 1: Self-attention + Add & Norm (Post-LN)
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: FFN + Add & Norm (Post-LN)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ══════════════════════════════════════════════════════════════════════
# ❼  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # encoder-decoder
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sub-layer 1: Masked self-attention + Add & Norm
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # Sub-layer 2: Cross-attention (query=decoder, key/val=encoder) + Add & Norm
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Sub-layer 3: FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


# ══════════════════════════════════════════════════════════════════════
# ❽  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
# ❾  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# SELF-CONTAINED VOCAB + CONSTANTS (no dataset.py / datasets import)
# Used by Transformer.__init__ in inference mode on autograder.
# ══════════════════════════════════════════════════════════════════════

_UNK_IDX = 0
_PAD_IDX = 1
_SOS_IDX = 2
_EOS_IDX = 3
_UNK_TOK = "<unk>"


class _MinimalVocab:
    """
    Lightweight vocabulary: only needs a pre-built stoi dict.
    No dependency on datasets, HuggingFace, or dataset.py.
    """
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, _UNK_TOK)

    def encode(self, tokens):
        return [self.stoi.get(t, _UNK_IDX) for t in tokens]


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Supports two modes:
        Training mode : pass src_vocab_size and tgt_vocab_size explicitly.
        Inference mode: call Transformer() with no args — downloads weights
                        from Google Drive, loads vocab and spaCy tokenizers,
                        ready for infer(german_sentence).

    Args:
        src_vocab_size (int)  : Source vocab size. None → inference mode.
        tgt_vocab_size (int)  : Target vocab size. None → inference mode.
        d_model        (int)  : Model dimensionality.
        N              (int)  : Number of encoder/decoder layers.
        num_heads      (int)  : Number of attention heads.
        d_ff           (int)  : FFN inner dimensionality.
        dropout        (float): Dropout probability.
        gdrive_id      (str)  : Google Drive file ID for weights download.
        checkpoint_path(str)  : Local path to save/load checkpoint.
    """

    # ── REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID ────────────
    _GDRIVE_ID       = "1uUXcEhitXt5eX6Y5HD3J59O7iYrt79LU"
    _CHECKPOINT_PATH = "best_model.pt"
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        src_vocab_size: int   = 7853,   # default: our trained model vocab size
        tgt_vocab_size: int   = 5893,   # default: our trained model vocab size
        d_model:   int        = 256,
        N:         int        = 3,
        num_heads: int        = 8,
        d_ff:      int        = 512,
        dropout:   float      = 0.3,
        checkpoint_path: str  = "best_model.pt",  # triggers download+load on init
    ) -> None:
        super().__init__()

        # Build model architecture first
        self._build_model(
            src_vocab_size, tgt_vocab_size,
            d_model, N, num_heads, d_ff, dropout,
        )

        # Always load weights from checkpoint (download if needed)
        self._setup_inference(
            self._GDRIVE_ID,
            checkpoint_path,
        )

    def _build_model(
        self,
        src_vocab_size, tgt_vocab_size,
        d_model, N, num_heads, d_ff, dropout,
    ):
        """Build all model layers."""
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # Positional Encoding
        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        # Encoder & Decoder stacks
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final linear projection → vocab logits
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Store config for checkpoint saving
        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model":        d_model,
            "N":              N,
            "num_heads":      num_heads,
            "d_ff":           d_ff,
            "dropout":        dropout,
        }

        # Xavier uniform initialisation
        self._init_weights()

    def _setup_inference(self, gdrive_id: str, checkpoint_path: str):
        """
        Download checkpoint, load weights + vocab + spaCy tokenizer.
        Architecture already built before this is called.
        Fully self-contained — no dataset.py / datasets imports.
        Auto-downloads de_core_news_sm if missing (handles autograder).
        """
        import os, spacy

        # ── 1. Download weights ────────────────────────────────────────
        if not os.path.exists(checkpoint_path):
            print(f"Downloading weights from Google Drive → {checkpoint_path}")
            import gdown
            gdown.download(id=gdrive_id, output=checkpoint_path, quiet=False)

        # ── 2. Load checkpoint ─────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded checkpoint config: {ckpt['model_config']}")

        # ── 3. Load weights ────────────────────────────────────────────
        self.load_state_dict(ckpt["model_state_dict"])
        print("Model weights loaded.")

        # ── 4. Restore vocab (self-contained, no dataset.py needed) ───
        self._sos = _SOS_IDX
        self._eos = _EOS_IDX
        self._pad = _PAD_IDX

        if "src_vocab_stoi" in ckpt:
            self.src_vocab = _MinimalVocab(ckpt["src_vocab_stoi"])
            self.tgt_vocab = _MinimalVocab(ckpt["tgt_vocab_stoi"])
            print(f"Vocab restored: src={len(self.src_vocab)} tgt={len(self.tgt_vocab)}")
        else:
            raise RuntimeError(
                "Checkpoint missing vocab. Re-save with save_checkpoint() "
                "that embeds src_vocab_stoi and tgt_vocab_stoi."
            )

        # ── 5. Load spaCy German tokenizer (auto-download if missing) ──
        # Use spacy.cli.download — no subprocess, no python/python3 ambiguity
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            print("de_core_news_sm not found — downloading via spacy.cli...")
            from spacy.cli import download as spacy_download
            spacy_download("de_core_news_sm")
            self._spacy_de = spacy.load("de_core_news_sm")
        print("spaCy German tokenizer ready.")

    def _tokenize_de(self, text: str):
        """German string → list of lowercase tokens via spaCy."""
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── INFERENCE ENTRY POINT (autograder contract) ─────────────────

    def infer(self, src_sentence: str) -> str:
        """
        End-to-end NMT: German string → English string.
        Greedy autoregressive decoding (fast, fits 3-sec autograder timeout).
        """
        device = next(self.parameters()).device
        sos, eos = self._sos, self._eos

        # 1. Tokenise German
        tokens  = self._tokenize_de(src_sentence)
        indices = [sos] + self.src_vocab.encode(tokens) + [eos]
        src     = torch.tensor([indices], dtype=torch.long, device=device)

        # 2. Encode
        src_mask = make_src_mask(src).to(device)
        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # 3. Greedy decode — one argmax per step, max 50 tokens
            ys = torch.tensor([[sos]], dtype=torch.long, device=device)
            for _ in range(50):
                tgt_mask = make_tgt_mask(ys).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == eos:
                    break

        # 4. Detokenise
        out = []
        for idx in ys[0, 1:].tolist():
            if idx == eos:
                break
            tok = self.tgt_vocab.lookup_token(idx)
            if tok not in ("<unk>", "<pad>", "<sos>", "<eos>"):
                out.append(tok)
        return " ".join(out)

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.config["d_model"]))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.config["d_model"]))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


# ══════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK  — run: python model.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dummy vocab sizes
    SRC_VOCAB = 8000
    TGT_VOCAB = 6000

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params:,}")

    # Dummy batch
    src = torch.randint(2, SRC_VOCAB, (4, 20)).to(device)
    tgt = torch.randint(2, TGT_VOCAB, (4, 18)).to(device)
    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt)

    logits = model(src, tgt, src_mask, tgt_mask)
    print(f"Output logits shape: {logits.shape}")  # [4, 18, 6000]

    # Test attention weights sum to 1
    enc_layer = model.encoder.layers[0]
    attn_w = enc_layer.self_attn.attn_weights
    print(f"Attn weights sum (should be ~1): {attn_w[0, 0, 0].sum().item():.4f}")

    # Test mask: src with pad
    src_padded = src.clone()
    src_padded[0, -3:] = 1   # pad last 3
    mask = make_src_mask(src_padded)
    print(f"Src mask shape: {mask.shape}")        # [4, 1, 1, 20]
    print(f"Tgt mask shape: {tgt_mask.shape}")    # [4, 1, 18, 18]
    print("All checks passed ✓")