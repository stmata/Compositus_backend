from typing import Any, Dict, List, Optional
import numpy as np
from utils.db_service import MongoDBManager
from datetime import datetime, timezone
from services.embedding_service import _cosine 
from services.cluster_storage_service import _save_embedding_blob, _move_embedding_blob   # si tu veux le split ainsi
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any, Optional, Tuple

CLUSTER_META_COLL = os.getenv("VAC_CLUSTER_META_COLL", "vac_clusters")
mongo = MongoDBManager()
META = mongo.get_collection(CLUSTER_META_COLL)
MASTER_COLL = mongo.get_collection("employees")
MASTER_ID = "EAP_Employees"
CLUSTER_META = mongo.get_collection("vacataires")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _load_cluster_meta(source: str) -> Optional[Dict[str, Any]]:
    doc_id = f"{MASTER_ID}:{source}"
    return CLUSTER_META.find_one({"_id": doc_id}) or None


def _l2n(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        n = np.linalg.norm(X) + 1e-12
        return (X / n).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype(np.float32)

def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2n(a); b = _l2n(b)
    return float((a * b).sum())

def _meta_id(source: str) -> str:
    return f"meta::{source}"

def _load_meta(source: str) -> Dict[str, Any]:
    return META.find_one({"_id": _meta_id(source)}) or {}

def _save_meta(source: str, centers: np.ndarray, counts: List[int], stats: Dict[str, Any]) -> None:
    META.update_one(
        {"_id": _meta_id(source)},
        {"$set": {
            "centers": centers.astype(np.float32).tolist(),
            "counts": [int(c) for c in counts],
            "stats": stats,  
            "updated_at": _now_iso(),
        }},
        upsert=True,
    )

def assign_new_vector_to_existing_clusters(
    source: str,
    collab_key: str,
    vec: np.ndarray,
    *,
    k_sigma: float = 2.0,
    hard_cosine_threshold: float = 0.82,
) -> Dict[str, Any]:
    """
    Assigne 'vec' au meilleur centre existant s'il est 'assez proche'.
    Critères:
      - d = 1 - cos(vec, center_best) <= mu + k_sigma * sigma
      - cos(vec, center_best) >= hard_cosine_threshold
    Retour: {"assigned": bool, "cluster_id": int|None, "score": float, "reason": str}
    """
    meta = _load_cluster_meta(source)
    if not meta:
        return {"assigned": False, "cluster_id": None, "score": 0.0, "reason": "no_meta"}
    centers = np.array(meta.get("centers") or [], dtype=np.float32)
    if centers.size == 0:
        return {"assigned": False, "cluster_id": None, "score": 0.0, "reason": "no_centers"}

    best_cid, best_cos = None, -1.0
    for cid, C in enumerate(centers):
        cs = _cosine(vec, np.array(C, dtype=np.float32))
        if cs > best_cos:
            best_cos, best_cid = cs, cid

    if best_cid is None:
        return {"assigned": False, "cluster_id": None, "score": 0.0, "reason": "no_selection"}

    d = 1.0 - float(best_cos)
    stats = meta.get("stats", {}).get(str(best_cid), {"mu": 1.0, "sigma": 0.0})
    mu = float(stats.get("mu", 1.0)); sigma = float(stats.get("sigma", 0.0))
    accept_by_radius = d <= (mu + k_sigma * sigma)
    accept_by_cosine = best_cos >= hard_cosine_threshold

    if accept_by_radius and accept_by_cosine:
        return {"assigned": True, "cluster_id": int(best_cid), "score": float(best_cos), "reason": "accepted"}
    return {"assigned": False, "cluster_id": int(best_cid), "score": float(best_cos), "reason": "too_far"}

def incremental_assign_update_centroid(
    source: str,
    collab_key: str,
    vec: np.ndarray,
    *,
    create_new_if_below: float = 0.85,
    hard_cosine_threshold: float = 0.93,
    min_points_warmup: int = 2,
    warmup_min_cos: float = 0.90,
    k_sigma: float = 1.5,
    enforce_cap: bool = True,
    write_blob: bool = True,
    previous_label: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Incrémente/assigne avec cap sur le nombre de clusters.

    Méta (collection 'vac_clusters', _id = 'meta::<source>'):
      - centers: list[list[float]]
      - counts:  list[int]
      - stats:   { cid: { n, mu, M2 } }   
      - max_clusters: int                

    Retour:
      {
        "assigned": True,
        "cluster_id": int,
        "score": float,        
        "emb_blob": str|None,
        "reason": str          
      }
    """
    v = _l2n(vec)

    meta   = _load_meta(source)
    centers = np.array(meta.get("centers") or [], dtype=np.float32)
    counts  = list(meta.get("counts") or [])
    stats   = dict(meta.get("stats") or {})

    max_clusters_cap = int(meta.get("max_clusters", 1000)) if enforce_cap else 1_000_000

    if centers.size == 0:
        centers = v.reshape(1, -1)
        counts  = [1]
        stats   = {"0": {"n": 1, "mu": 0.0, "M2": 0.0}}
        emb_blob = _save_embedding_blob(source, 0, collab_key, v) if write_blob else None
        _save_meta(source, centers, counts, stats)
        return {"assigned": True, "cluster_id": 0, "score": 1.0, "emb_blob": emb_blob, "reason": "bootstrap"}

    best_id, best_cos = None, -1.0
    for cid, C in enumerate(centers):
        cs = _cosine_vec(v, C)
        if cs > best_cos:
            best_cos, best_id = cs, cid

    label = int(best_id)
    n     = int(counts[label])
    dist  = 1.0 - float(best_cos)

    s       = stats.get(str(label), {"n": 0, "mu": 0.0, "M2": 0.0})
    n_stat  = int(s.get("n", 0))
    mu      = float(s.get("mu", 0.0))
    sigma   = float(np.sqrt(s.get("M2", 0.0) / max(n_stat - 1, 1))) if n_stat > 1 else 0.0

    if best_cos < float(create_new_if_below):
        if len(centers) < max_clusters_cap:
            new_label = int(len(centers))
            centers = np.vstack([centers, v])
            counts.append(1)
            stats[str(new_label)] = {"n": 1, "mu": 0.0, "M2": 0.0}
            emb_blob = _save_embedding_blob(source, new_label, collab_key, v) if write_blob else None
            _save_meta(source, centers, counts, stats)
            return {
                "assigned": True,
                "cluster_id": new_label,
                "score": float(best_cos),
                "emb_blob": emb_blob,
                "reason": "below_cut",
            }
        else:
            pass  

    accept_by_cos    = best_cos >= float(hard_cosine_threshold)
    accept_by_warmup = (n < int(min_points_warmup)) and (best_cos >= float(warmup_min_cos))
    accept_by_radius = (n_stat >= 3) and (dist <= (mu + k_sigma * sigma))

    force_assign_due_to_cap = (best_cos < float(create_new_if_below)) and (len(centers) >= max_clusters_cap)

    if accept_by_cos or accept_by_warmup or accept_by_radius or force_assign_due_to_cap:
        new_center = _l2n((centers[label] * n + v) / (n + 1.0))
        centers[label] = new_center
        counts[label]  = n + 1

        delta  = dist - mu
        mu_new = mu + delta / (n_stat + 1)
        M2_new = s.get("M2", 0.0) + delta * (dist - mu_new)
        stats[str(label)] = {"n": n_stat + 1, "mu": mu_new, "M2": M2_new}

        if write_blob:
            if previous_label is None or previous_label == label:
                emb_blob = _save_embedding_blob(source, label, collab_key, v)
            else:
                emb_blob = _move_embedding_blob(source, previous_label, label, collab_key)
        else:
            emb_blob = None

        _save_meta(source, centers, counts, stats)
        return {
            "assigned": True,
            "cluster_id": label,
            "score": float(best_cos),
            "emb_blob": emb_blob,
            "reason": "cap_reached_force_assign" if force_assign_due_to_cap else "accepted",
        }

    if len(centers) < max_clusters_cap:
        new_label = int(len(centers))
        centers = np.vstack([centers, v])
        counts.append(1)
        stats[str(new_label)] = {"n": 1, "mu": 0.0, "M2": 0.0}
        emb_blob = _save_embedding_blob(source, new_label, collab_key, v) if write_blob else None
        _save_meta(source, centers, counts, stats)
        return {"assigned": True, "cluster_id": new_label, "score": float(best_cos), "emb_blob": emb_blob, "reason": "fallback_new"}

    new_center = _l2n((centers[label] * n + v) / (n + 1.0))
    centers[label] = new_center
    counts[label]  = n + 1

    delta  = dist - mu
    mu_new = mu + delta / (n_stat + 1)
    M2_new = s.get("M2", 0.0) + delta * (dist - mu_new)
    stats[str(label)] = {"n": n_stat + 1, "mu": mu_new, "M2": M2_new}

    if write_blob:
        if previous_label is None or previous_label == label:
            emb_blob = _save_embedding_blob(source, label, collab_key, v)
        else:
            emb_blob = _move_embedding_blob(source, previous_label, label, collab_key)
    else:
        emb_blob = None

    _save_meta(source, centers, counts, stats)
    return {
        "assigned": True,
        "cluster_id": label,
        "score": float(best_cos),
        "emb_blob": emb_blob,
        "reason": "cap_reached_force_assign",
    }

def _auto_kmeans(vectors: np.ndarray, k_min=2, k_max=30, min_cluster_size=0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = vectors.shape[0]
    if n == 0: return np.array([], dtype=int), None
    if n < 3 or vectors.shape[1] < 2: return np.zeros(n, dtype=int), None
    best_score = -1.0; best = None
    upper = min(k_max, n)
    for k in range(k_min, max(k_min + 1, upper + 1)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vectors)
            if min_cluster_size:
                counts = np.bincount(labels, minlength=k)
                if (counts < min_cluster_size).any():
                    continue
            if len(set(labels)) == 1: continue
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score; best = (labels, km.cluster_centers_)
        except Exception:
            continue
    if best is None: return np.zeros(n, dtype=int), None
    return best

def estimate_target_k(vectors: np.ndarray, k_min=2, k_max=30, headroom=0.2):
    labels, _ = _auto_kmeans(vectors, k_min=k_min, k_max=k_max, min_cluster_size=0)
    if labels is None or labels.size == 0:
        return k_min
    k_star = int(len(set(labels)))
    k_cap = int(np.ceil(k_star * (1.0 + float(headroom))))
    return max(k_min, min(k_cap, k_max))



def update_max_clusters_meta(source: str, vectors: np.ndarray, k_min=2, k_max=30, headroom=0.2):
    k_cap = estimate_target_k(vectors, k_min=k_min, k_max=k_max, headroom=headroom)
    meta = _load_meta(source) or {}            
    meta["target_k"] = int(k_cap)
    meta["max_clusters"] = int(k_cap)
    meta["updated_at"] = _now_iso()
    META.update_one({"_id": _meta_id(source)}, {"$set": meta}, upsert=True) 
    return k_cap
