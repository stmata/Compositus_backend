import json, math, re
import numpy as np
import unicodedata
from colorama import init as colorama_init, Fore, Style
from typing import Dict, Any, Optional, List
import pandas as pd
import os

FILENAME_KEYWORDS: Dict[str, str] = {
    "professeurs": "EAP_Professeurs",
    "administratifs": "EAP_Administratif",
    "entretiens": "EAP_EntretiensProfessionnels",
}
def collapse_whitespace(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n\s*\r?\n+", "\n\n", s)
    return s.strip()

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)

def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def colorize(txt: str, color: str, bright: bool = False) -> str:
    return f"{(Style.BRIGHT if bright else '')}{color}{txt}{Style.RESET_ALL}"

def norm_key(s: str) -> str:
    s = "" if s is None else str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.strip().lower()
    s = s.replace("|", " ").replace("’", "'")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())

def clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None

def safe_person_key(name: str) -> str:
    if not name:
        return "inconnu"
    return name.replace(".", "·").replace("$", "￩").strip()

def detect_source_label(path: str) -> str:
    """Fallback par nom de fichier (non utilisé en mode strict si kind est fourni)."""
    base = os.path.basename(path)
    base = norm_key(base)
    for kw, label in FILENAME_KEYWORDS.items():
        if kw in base:
            return label
    return "EAP_Autres"

def normalize_year(val, fallback_date=None):
    s = clean_value(val)
    if s:
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            return m.group(0)
    d = clean_value(fallback_date)
    if d:
        m = re.search(r"(19|20)\d{2}", d)
        if m:
            return m.group(0)
    return None

def build_idx(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    normed = {c: norm_key(c) for c in cols}

    def find_one(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            r = re.compile(pat)
            for c, n in normed.items():
                if r.fullmatch(n):
                    return c
        return None

    return {
        "matricule": find_one([r"matricule", r"matricule_collaborateur"]),
        "collaborateur": find_one([r"collaborateur"]),
        "annee": find_one([r"annee", r"annee_.*"]),
        "date_entretien": find_one([r"date_de_lentretien", r"date_entretien", r"date.*entretien"]),
    }

def row_to_struct(row: pd.Series, idx: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    """Compacte une ligne en structure exploitable; None si insuffisant."""
    collaborateur_raw = row.get(idx["collaborateur"])
    matricule_raw     = row.get(idx["matricule"])
    annee_raw         = row.get(idx["annee"])
    date_entretien    = row.get(idx["date_entretien"])

    collaborateur = clean_value(collaborateur_raw)
    if not collaborateur:
        return None

    collab_key = safe_person_key(collaborateur)
    year = normalize_year(annee_raw, fallback_date=date_entretien)
    if not year:
        return None

    matricule_val = None
    mv = clean_value(matricule_raw)
    if mv is not None:
        try:
            matricule_val = int(float(mv))
        except Exception:
            matricule_val = mv

    identity_cols = set(filter(None, [idx["matricule"], idx["collaborateur"], idx["annee"]]))
    identity_keys_normed = {norm_key(c) for c in identity_cols}
    identity_keys_normed.add("matricule_collaborateur")

    payload: Dict[str, Any] = {}
    for col, val in row.items():
        if col is None:
            continue
        if norm_key(col) in identity_keys_normed:
            continue
        v = clean_value(val)
        if v is None:
            continue
        col_norm = norm_key(col)
        if not col_norm:
            continue
        payload[col_norm] = v

    if not payload and matricule_val is None:
        return None

    out: Dict[str, Any] = {
        "collab_key": collab_key,
        "year": year,
        "payload": payload,
    }
    if matricule_val is not None:
        out["matricule_collaborateur"] = matricule_val
    return out



