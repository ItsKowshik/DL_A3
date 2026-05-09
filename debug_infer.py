"""
debug_infer.py — diagnose why infer() outputs only periods
"""
import torch
from dataset import get_dataloaders, SOS_IDX, EOS_IDX, PAD_IDX
from model import Transformer, make_src_mask, make_tgt_mask
from train import greedy_decode, load_checkpoint
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load vocab the TRAINING way (ground truth) ─────────────────────────
print("Building vocab from training data...")
_, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=1)

# ── Load model in TRAINING mode (known working) ────────────────────────
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=config.D_MODEL,
    N=config.N_LAYERS,
    num_heads=config.NUM_HEADS,
    d_ff=config.D_FF,
    dropout=config.DROPOUT,
).to(device)

CKPT = "best_model_40.pt"
load_checkpoint(CKPT, model)
model.eval()

# ── Test sentence ──────────────────────────────────────────────────────
german = "Zwei junge Männer spielen Fußball."

# ── Tokenise with spaCy (same as dataset.py) ──────────────────────────
import spacy
spacy_de = spacy.load("de_core_news_sm")
tokens = [tok.text.lower() for tok in spacy_de.tokenizer(german)]
print(f"\nGerman tokens : {tokens}")

# ── Encode with TRAINING vocab ─────────────────────────────────────────
indices = [SOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
print(f"Src indices   : {indices}")

src      = torch.tensor([indices], dtype=torch.long, device=device)
src_mask = make_src_mask(src).to(device)

# ── Greedy decode (known working) ──────────────────────────────────────
ys = greedy_decode(model, src, src_mask, max_len=50,
                   start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device)

out_indices = ys[0].tolist()
out_tokens  = [tgt_vocab.lookup_token(i) for i in out_indices]
print(f"\nGreedy indices: {out_indices}")
print(f"Greedy tokens : {out_tokens}")
translation  = " ".join(t for t in out_tokens
                        if t not in ("<sos>","<eos>","<pad>","<unk>"))
print(f"Translation   : {translation}")

# ── Now compare with inference-mode vocab ──────────────────────────────
print("\n── Inference-mode vocab check ──")
inf_model = Transformer().to(device)  # loads from GDrive checkpoint
inf_model.eval()

# Check first 10 tgt vocab entries
print("Training vocab vs Inference vocab (first 15 tgt entries):")
for i in range(15):
    train_tok = tgt_vocab.lookup_token(i)
    infer_tok = inf_model.tgt_vocab.lookup_token(i)
    match = "✓" if train_tok == infer_tok else "✗ MISMATCH"
    print(f"  idx {i:3d}: train='{train_tok}'  infer='{infer_tok}'  {match}")

# Check what "." maps to in both vocabs
dot_train = tgt_vocab.stoi.get(".", -1)
dot_infer  = inf_model.tgt_vocab.stoi.get(".", -1)
print(f"\n'.' index — training vocab: {dot_train}, inference vocab: {dot_infer}")

# Check the generated indices in inference vocab
print(f"\nGenerated indices via inference vocab: {out_indices}")
for idx in out_indices[:8]:
    print(f"  idx {idx} → training='{tgt_vocab.lookup_token(idx)}'  "
          f"inference='{inf_model.tgt_vocab.lookup_token(idx)}'")