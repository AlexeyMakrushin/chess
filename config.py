"""
Configuration for the chess engine (search/training/randomization).
"""

import os

EVALUATION_MODE = os.getenv("EVALUATION_MODE", "SIMPLE")  # SIMPLE | CUSTOM_NN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "m-chess.pth")

MINIMAX_DEPTH = 4
USE_ALPHA_BETA = True
TIME_LIMIT = 5.0

DATASET_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

# Opening randomization (to avoid repetitive first moves)
RANDOM_MOVE_CHANCE = 0.2          # probability to randomize among top moves
RANDOMIZE_OPENINGS_UNTIL = 3      # fullmove threshold to allow randomization
RANDOM_TOP_K = 10                 # choose randomly among top-K moves when randomizing
RANDOM_DECAY_THRESHOLD = 4.0      # higher eval magnitude → lower randomization chance

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
