# ══════════════════════════════════════════════════════════════════════
# config.py — Best config (targets BLEU 40.50+)
# ══════════════════════════════════════════════════════════════════════

D_MODEL      = 256
N_LAYERS     = 3
NUM_HEADS    = 8
D_FF         = 512
DROPOUT      = 0.3
MAX_SEQ_LEN  = 256

BATCH_SIZE   = 128
NUM_EPOCHS   = 60
WARMUP_STEPS = 2000
LABEL_SMOOTH = 0.1

PATIENCE     = 10
BEAM_SIZE    = 8          # best from test_beams.py
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "checkpoints/"
LOG_DIR        = "logs/"
WANDB_PROJECT  = "A3"