import torch
from dataset import get_dataloaders
from model import Transformer
from train import evaluate_bleu, load_checkpoint
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocab
_, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=1)

# ── Load model config FROM checkpoint (not from config.py) ────────────
# Avoids mismatch if config.py was changed (e.g. N_LAYERS 3→6)
ckpt = torch.load("best_model_40.pt",
                  map_location=device)
model_cfg = ckpt["model_config"]
print(f"Checkpoint config: {model_cfg}")

model = Transformer(**model_cfg).to(device)
load_checkpoint("best_model_40.pt", model)
model.eval()

# ── Test beam sizes ────────────────────────────────────────────────────
for beam in [1, 3, 5, 8, 10]:
    config.BEAM_SIZE = beam
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"beam={beam:2d}  BLEU={bleu:.4f}")
