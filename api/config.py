import os

MODEL_DIR = os.environ.get("RECGEN_MODEL_DIR", "models/SmolLM2-360M")
CACHE_DIR = os.environ.get("RECGEN_CACHE_DIR", ".cache/api")
DEVICE = os.environ.get("RECGEN_DEVICE", "mps")
POOLING = os.environ.get("RECGEN_POOLING", "mean")
MAX_LENGTH = int(os.environ.get("RECGEN_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("RECGEN_BATCH_SIZE", "32"))
HEAD_DIR = os.environ.get("RECGEN_HEAD_DIR", "")
