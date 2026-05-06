# ══════════════════════════════════════════════════════════════════════
# config.py — Hyperparameters for DA6401 Assignment 3
# ══════════════════════════════════════════════════════════════════════
 
# Model — tuned for Multi30k (29k pairs, short sentences)
D_MODEL      = 256
N_LAYERS     = 3
NUM_HEADS    = 8
D_FF         = 512
DROPOUT      = 0.3        # was 0.1 — higher dropout fights overfitting on small data
MAX_SEQ_LEN  = 256
 
# Training
BATCH_SIZE   = 128
NUM_EPOCHS   = 40         # early stopping will kick in before this
WARMUP_STEPS = 2000       # was 4000 — smaller model converges faster
LABEL_SMOOTH = 0.1
 
# Early stopping
PATIENCE     = 10          # stop if val_loss doesn't improve for 7 epochs
 
# Beam search
BEAM_SIZE    = 5          # used in evaluate_bleu
 
# Paths
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR        = "logs/"
# W&B
WANDB_PROJECT  = "A3"