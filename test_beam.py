# test_beams.py
from train import evaluate_bleu, load_checkpoint, beam_search_decode
from dataset import get_dataloaders
from model import Transformer
import config, torch
from evaluate import load as load_metric
from model import make_src_mask
from dataset import SOS_IDX, EOS_IDX, PAD_IDX

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=1)

model = Transformer(len(src_vocab), len(tgt_vocab),
    d_model=config.D_MODEL, N=config.N_LAYERS,
    num_heads=config.NUM_HEADS, d_ff=config.D_FF,
    dropout=config.DROPOUT).to(device)

load_checkpoint("checkpoints/best_model.pt", model)

for beam in [5, 10, 15, 20]:
    config.BEAM_SIZE = beam
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"beam={beam:2d}  BLEU={bleu:.4f}")