# config.py
# ─────────────────────────────────────────────
# Central configuration for the entire project.
# Change values here — they flow everywhere.
# ─────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "saved_model")
DATA_FILE       = os.path.join(DATA_DIR, "sample_data.json")

# ── Model ──────────────────────────────────────
# distilbert is small, fast, and runs well on CPU
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH  = 256       # max token length per input

# ── Training ───────────────────────────────────
EPOCHS          = 3
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
TEST_SPLIT      = 0.2       # 20% for validation
RANDOM_SEED     = 42

# ── Labels ─────────────────────────────────────
LABEL2ID = {"safe": 0, "injection": 1}
ID2LABEL = {0: "safe", 1: "injection"}

# ── Detection threshold ─────────────────────────
# Probability above this → flagged as injection
CONFIDENCE_THRESHOLD = 0.75
