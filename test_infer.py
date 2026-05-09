# test_infer.py
from model import Transformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer().to(device)   # downloads + loads everything
model.eval()

sentence = "Zwei junge Männer spielen Fußball."
english = model.infer(sentence)
print(f"DE: {sentence}")
print(f"EN: {english}")