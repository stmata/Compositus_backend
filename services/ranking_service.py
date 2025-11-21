from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from services.embedding_service import _cosine

def rank_candidates_for_job(job_vec: np.ndarray,
                            candidates: List[Tuple[str, str, str, np.ndarray, str]],
                            top_k: int = 50) -> List[Dict[str, Any]]:
    scored = []
    for src, ck, name, v, txt in candidates:
        score = _cosine(job_vec, v)
        scored.append({
            "source": src,
            "collab_key": ck,
            "name": name,
            "score": max(0.0, min(1.0, float(score))),
            "profile_text": txt
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
