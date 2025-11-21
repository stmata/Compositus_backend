from __future__ import annotations
import os, json, time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from colorama import init as colorama_init, Fore, Style
from dotenv import load_dotenv
from pymongo.errors import AutoReconnect
from utils.db_service import MongoDBManager
from services.embedding_service import _ensure_env_or_die, _embed_texts, _build_embed_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from services.embedding_service import _cosine

from services.cluster_storage_service import (
    _save_embedding_blob,
    _cleanup_source_clusters,
)
from services.cluster_meta_service import (
    _auto_kmeans,
    _load_meta
)
colorama_init(autoreset=True)

load_dotenv()

def _c(txt: str, color: str = Fore.WHITE, bright: bool = False) -> str:
    return f"{Style.BRIGHT if bright else ''}{color}{txt}{Style.RESET_ALL}"

API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
API_BASE = (os.getenv("API_BASE") or "").rstrip("/")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS", "text-embedding-3-large")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
EMB_CONTAINER = os.getenv("EMB_CONTAINER", "employee-embeddings")

mongo = MongoDBManager()
MASTER_COLL = mongo.get_collection("employees")
MASTER_ID = "EAP_Employees"
CLUSTER_META = mongo.get_collection("vacataires")

SOURCE_MAP: Dict[str, str] = {
    "professeurs": "EAP_Professeurs",
    "administratifs": "EAP_Administratif",
    "entretiens": "EAP_EntretiensProfessionnels",
}

def _source_labels(kind: Optional[str], doc: Dict[str, Any]) -> List[str]:
    if kind:
        raw = (kind or "").strip()
        if raw.replace(" ", "").startswith("EAP_"):
            return [raw.replace(" ", "")]
        k = raw.strip().lower()
        mapped = SOURCE_MAP.get(k)
        return [mapped] if mapped else [raw]
    base = []
    for key, val in (doc or {}).items():
        if isinstance(val, dict) and key.startswith("EAP_"):
            base.append(key)
    if "EAP_Autres" not in base and (doc or {}).get("EAP_Autres"):
        base.append("EAP_Autres")
    return sorted(base)

def _pick_summary(node: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(node, dict): return None
    for k in ("summrize", "summarize", "summary"):
        if k in node and node[k] is not None:
            return node[k]
    s = node.get("summary") if isinstance(node.get("summary"), dict) else None
    if s and "text" in s: return s["text"]
    return None

def _extract_summary_text(summ: Any) -> str:
    if summ is None: return ""
    if isinstance(summ, str): return summ.strip()
    try:
        return json.dumps(summ, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(summ)


def _mongo_update_with_retry(coll, filt, update, upsert=True, max_retries=5, initial_delay=0.5):
    delay = initial_delay
    for i in range(max_retries):
        try:
            return coll.update_one(filt, update, upsert=upsert)
        except AutoReconnect:
            if i == max_retries - 1: raise
            time.sleep(delay); delay = min(delay * 2, 8.0)

def _approx_field_bytes(k: str, v: Any) -> int:
    base = len(k) + 8
    if isinstance(v, (int, float)): return base + 8
    if isinstance(v, str): return base + min(len(v), 64)
    if isinstance(v, list): return base + min(len(v), 64)
    return base + 16

def _chunked_set_with_retry(coll, filt, items: List[Tuple[str, Any]], *, max_bytes: int = 800_000):
    batch: Dict[str, Any] = {}
    sz = 0
    sent = 0
    for k, v in items:
        field_sz = _approx_field_bytes(k, v)
        if batch and (sz + field_sz) > max_bytes:
            _mongo_update_with_retry(coll, filt, {"$set": batch}, upsert=True)
            sent += 1
            batch = {}; sz = 0
        batch[k] = v; sz += field_sz
    if batch:
        _mongo_update_with_retry(coll, filt, {"$set": batch}, upsert=True)
        sent += 1
    return sent

def cluster_and_upload_to_blob(
    kind: Optional[str] = None,
    embed_batch_size: int = 128,
    k_min: int = 2,
    k_max: int = 30,
    min_cluster_size: int = 0,
    cleanup_before_write: bool = True,  
) -> Dict[str, Any]:
    """
    Pour chaque SOURCE:
      1) lit summaries
      2) embeddings (Azure OpenAI)
      3) clustering auto KMeans (silhouette)
      4) écrit embeddings sous Blob: <container>/<source>/C<id>/<collab>.npy
      5) met à jour employees: cluster.semantic + embedding_blob (chunked)

    Pas de collection 'employee_clusters'. Navigation par arborescence Blob.
    """
    _ensure_env_or_die()

    doc = MASTER_COLL.find_one({"_id": MASTER_ID}) or {}
    sections = _source_labels(kind, doc)

    embed = _build_embed_model()

    global_stats: Dict[str, Any] = {
      "ok": True,
      "sources": [],
      "total_users": 0,
      "total_clusters": 0,
    }

    for src in sections:
        bucket = (doc or {}).get(src, {}) or {}
        if not isinstance(bucket, dict) or not bucket:
            continue

        entries_src: List[Tuple[str, str]] = [] 
        for collab_key, node in bucket.items():
            summ = _pick_summary(node)
            text = _extract_summary_text(summ)
            if text:
                entries_src.append((collab_key, text))

        n_src = len(entries_src)
        if n_src == 0:
            continue


        texts = [t for (_, t) in entries_src]
        try:
            vectors = _embed_texts(embed, texts, batch_size=embed_batch_size)
        except Exception as e:
            global_stats["ok"] = False
            continue

        labels, centers = _auto_kmeans(vectors, k_min=k_min, k_max=k_max, min_cluster_size=min_cluster_size)
        if labels is None or vectors is None or len(labels) != len(entries_src):
            global_stats["ok"] = False
            continue  
        n_clusters_src = len(set(labels)) if labels.size else 0

        deleted = 0
        if cleanup_before_write:
            deleted = _cleanup_source_clusters(src)
            if deleted:
                print(_c(f"{src}: deleted {deleted} old blob(s)", Fore.YELLOW))

        set_items: List[Tuple[str, Any]] = []
        for (ck, _), lbl, vec in zip(entries_src, labels.tolist(), vectors.tolist()):
            _save_embedding_blob(src, int(lbl), ck, np.array(vec))

        reqs = _chunked_set_with_retry(MASTER_COLL, {"_id": MASTER_ID}, set_items, max_bytes=800_000)

        global_stats["sources"].append({
            "source": src,
            "users": n_src,
            "clusters": int(n_clusters_src),
            "deleted_old_blobs": int(deleted),
        })
        global_stats["total_users"] += n_src
        global_stats["total_clusters"] += int(n_clusters_src)

    return global_stats

CLUSTER_META_COLL = os.getenv("VAC_CLUSTER_META_COLL", "vac_clusters")  
mongo = MongoDBManager()
META = mongo.get_collection(CLUSTER_META_COLL)


def build_cv_embedding_text(parsed: Optional[dict]) -> str:
    if not isinstance(parsed, dict): return ""
    ident = parsed.get("identity", {}) or {}
    exps  = parsed.get("experiences", []) or []
    skills = parsed.get("skills", []) or []
    edu   = parsed.get("education", []) or []

    def norm(s: Any) -> str:
        return str(s).strip().lower()

    parts: List[str] = []
    title = ident.get("current_title") or ident.get("headline") or ""
    if title: parts.append(f"TITLE: {norm(title)}")
    if skills:
        parts.append("SKILLS: " + ", ".join(sorted({norm(k) for k in skills if k})))

    if exps:
        xp = []
        for x in exps:
            t = x.get("title") or ""
            kws = ", ".join(sorted({norm(k) for k in (x.get("keywords") or []) if k}))
            line = " - ".join([p for p in [norm(t), kws] if p])
            if line: xp.append(line)
        if xp: parts.append("EXPERIENCES: " + " || ".join(xp))

    if edu:
        ed = []
        for e in edu:
            line = " - ".join([norm(e.get("degree") or ""), norm(e.get("field") or ""), norm(e.get("school") or "")]).strip(" -")
            if line: ed.append(line)
        if ed: parts.append("EDUCATION: " + " || ".join(ed))

    return "\n".join(parts).strip()

def cluster_and_upload_to_blob2(kind: Optional[str] = None, **_) -> Dict[str, Any]:
    source = kind or "Vacataires"
    meta = _load_meta(source)
    return {"ok": True, "source": source, "total_clusters": len(meta.get("centers") or [])}





def _optimal_k_vac(
    vectors: np.ndarray,
    k_max: int = 12,
    cos_threshold: float = 0.80,
) -> int:
    """
    Choisit automatiquement k avec une étape de filtrage :
    - Si toutes les similarités cosinus < cos_threshold → k = n (1 cluster par personne)
    - Sinon → silhouette classique.
    """
    n = vectors.shape[0]

    if n <= 1:
        return 1

    max_cos = -1.0
    try:
        for i in range(n):
            for j in range(i + 1, n):
                c = _cosine(vectors[i], vectors[j])
                if c > max_cos:
                    max_cos = c

        if max_cos < cos_threshold:
            return n 
    except Exception:
        pass


    if n == 2:
        return 2

    k_max_eff = min(k_max, n - 1)

    best_k = 2
    best_sil = -1.0

    for k in range(2, k_max_eff + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(vectors)

        if len(set(labels)) < 2:
            continue

        try:
            sil = silhouette_score(vectors, labels)
        except Exception:
            continue

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return best_k

def _auto_kmeans_vac(vectors: np.ndarray) -> np.ndarray:
    """
    Applique KMeans avec un k décidé automatiquement.
    - Si tous très différents -> k = n
    - Sinon -> k optimal basé sur silhouette
    """
    n = vectors.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    if n == 1:
        return np.zeros(1, dtype=int)

    k = _optimal_k_vac(vectors)

    k = min(k, n)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(vectors)
