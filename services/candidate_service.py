from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import io, os, json
import numpy as np
from utils.db_service import MongoDBManager
from azure.storage.blob import BlobServiceClient

def _dept_alias_key(s: str) -> str:
    import re
    s = (s or "").strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "", s)

DEPT_ALIASES = {
    "eap_professional_interviews": "EAP_EntretiensProfessionnels",
    "eap_professors": "EAP_Professeurs",
    "eap_administratif": "EAP_Administratif",
    "eap_entretiensprofessionnels": "EAP_EntretiensProfessionnels",
    "eap_professeurs": "EAP_Professeurs",
    "eap_administratif": "EAP_Administratif",
}

def _normalize_dept(s: str) -> str:
    key = _dept_alias_key(s); return DEPT_ALIASES.get(key, s)

def _is_all(dept: str) -> bool:
    key = _dept_alias_key(dept)
    return key in {"", "all", "tout", "tous", "toutes", "everything", "*", "eap_all"}

def _source_matches_department(src_key: str, department: str) -> bool:
    if _is_all(department):
        return isinstance(src_key, str) and src_key.startswith("EAP_")
    return _normalize_dept(src_key) == _normalize_dept(department)

def _best_text_from_node(node: Dict[str, Any]) -> str:
    if not isinstance(node, dict): return ""
    summ = node.get("summrize") or node.get("summary") or node.get("profile") or {}
    if isinstance(summ, dict):
        parts = []
        for k in ("summary_long", "summary", "embedding_text", "headline", "skills", "experience"):
            v = summ.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list):
                parts.append(", ".join([str(x).strip() for x in v if str(x).strip()]))
        if parts: return "\n".join(parts)
        try: return json.dumps(summ, ensure_ascii=False)
        except Exception: pass
    if isinstance(summ, str) and summ.strip(): return summ.strip()
    try: return json.dumps(node, ensure_ascii=False)
    except Exception: return ""

def _display_name_from_node(node: Dict[str, Any], ck: str) -> str:
    if not isinstance(node, dict): return ck
    summ = node.get("summrize") or node.get("summary") or {}
    for key in ("full_name", "name", "display_name"):
        val = summ.get(key) if isinstance(summ, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()
    ident = (summ.get("identity") if isinstance(summ, dict) else None) or {}
    if isinstance(ident, dict):
        fn = (ident.get("first_name") or "").strip()
        ln = (ident.get("last_name") or "").strip()
        if fn or ln: return f"{fn} {ln}".strip()
    return ck

mongo = MongoDBManager()
EMP_COLL = mongo.get_collection("employees")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
try:
    blob = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
except Exception as e:
    blob = None

def _download_blob_to_ndarray(path: str) -> Optional[np.ndarray]:
    if not blob:
        return None
    try:
        container, blob_path = path.split("/", 1)
        cc = blob.get_container_client(container)
        data = cc.download_blob(blob_path).readall()
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return np.asarray(arr, dtype=np.float32)
    except Exception as e:
        return None

def load_candidate_vectors(department: str) -> List[Tuple[str, str, str, Optional[np.ndarray], str]]:
    doc = EMP_COLL.find_one({"_id": "EAP_Employees"}) or {}
    out: List[Tuple[str, str, str, Optional[np.ndarray], str]] = []
    for src_key, bucket in doc.items():
        if not (isinstance(bucket, dict) and src_key.startswith("EAP_")): continue
        if not _source_matches_department(src_key, department): continue
        for ck, node in bucket.items():
            if not isinstance(node, dict): continue
            name = _display_name_from_node(node, ck)
            blob_path = (node or {}).get("embedding_blob")
            vec: Optional[np.ndarray] = None
            if blob_path:
                vec = _download_blob_to_ndarray(blob_path)
                if vec is not None: vec = vec.astype(np.float32)
            profile_text = _best_text_from_node(node)
            out.append((src_key, ck, name, vec, profile_text))
    return out
