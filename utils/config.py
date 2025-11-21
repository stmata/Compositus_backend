import os
from dotenv import load_dotenv

load_dotenv()

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v.replace("_", ""))  
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default

CTX_MAX_TOKENS    = _env_int("CTX_MAX_TOKENS", 128_000)
MAX_CHUNKS        = _env_int("MAX_CHUNKS", 6)
TARGET_CHARS      = _env_int("TARGET_CHARS", 2200)
CHUNK_PREVIEW_MAX = _env_int("CHUNK_PREVIEW_MAX", 900)
EXPLAIN_TOP_N     = _env_int("EXPLAIN_TOP_N", 20)
EMBED_BATCH_SIZE  = _env_int("EMBED_BATCH_SIZE", 64)

EMBEDDING_SCORE_MIN = _env_float("EMBEDDING_SCORE_MIN", 0.5)
HYBRID_EMB_WEIGHT   = _env_float("HYBRID_EMB_WEIGHT", 0.6)

