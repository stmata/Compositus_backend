from __future__ import annotations
import os
from io import StringIO
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from colorama import init as colorama_init, Fore, Style
from utils.db_service import MongoDBManager
from services.clustering_service import cluster_and_upload_to_blob
from services.cluster_storage_service import delete_employee_embedding
from utils.job_tracker import push_event, set_stage, set_done, set_error
from utils.utils import (
    _norm,
    clean_value,
    build_idx,
    row_to_struct,
)
from services.user_flags_service import mark_all_users_update_done, mark_all_users_update_error, mark_all_users_update_start
from services.summary_service import _humanize_error_message, summarize_sources_in_batches

load_dotenv()
colorama_init(autoreset=True)

mongo = MongoDBManager()
MASTER_COLL = mongo.get_collection("employees")
USERS_COLL  = mongo.get_collection("users")
MASTER_ID = "EAP_Employees"

API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
API_BASE = os.getenv("API_BASE")
AZURE_DEPLOYMENT_SUMMARY = os.getenv("AZURE_DEPLOYMENT_SUMMARY", "gpt-4o-mini")

if not API_KEY or not API_VERSION or not API_BASE:
    print(f"{Style.BRIGHT}{Fore.YELLOW}{Style.RESET_ALL} Azure OpenAI env variables missing or incomplete. "
          f"API_KEY={bool(API_KEY)}, VERSION={API_VERSION!r}, BASE={API_BASE!r}. "
          f"LLM calls may fail.")

try:
    client = AzureOpenAI(api_key=API_KEY, api_version=API_VERSION, azure_endpoint=API_BASE)
except Exception as e:
    client = None  

VALID_SOURCES: set[str] = {
    "EAP_Professeurs",
    "EAP_Administratif",
    "EAP_EntretiensProfessionnels",
}

KIND_ALIAS: Dict[str, str] = {
    "professeurs": "EAP_Professeurs",
    "administratifs": "EAP_Administratif",
    "entretiens": "EAP_EntretiensProfessionnels",
    "eap_professeurs": "EAP_Professeurs",
    "eap_administratifs": "EAP_Administratif",
    "eap_administratif": "EAP_Administratif",
    "eap_entretiensprofessionnels": "EAP_EntretiensProfessionnels",
}

SRC_COLOR = {
    "EAP_Professeurs": Fore.MAGENTA,
    "EAP_Administratif": Fore.CYAN,
    "EAP_EntretiensProfessionnels": Fore.YELLOW,
    "EAP_Autres": Fore.WHITE,
}



def import_employee_csv(
    file_content: bytes,
    filename: str,
    kind: Optional[str] = None,
    replace: bool = True,
) -> str:
    source_label = KIND_ALIAS.get(_norm(kind))
    if not source_label or source_label not in VALID_SOURCES:
        raise ValueError(f"Kind invalide: {kind!r}. Attendu ∈ {sorted(VALID_SOURCES)}")


    try:
        content_str = file_content.decode("utf-8")
    except UnicodeDecodeError:
        content_str = file_content.decode("latin-1")

    try:
        df = pd.read_csv(StringIO(content_str), dtype=str, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"CSV parsing error for {filename}: {e}")


    if df.empty:
        MASTER_COLL.update_one({"_id": MASTER_ID}, {"$set": {source_label: {}}}, upsert=True)
        msg = f"[{filename}] Fichier vide: section '{source_label}' vidée."
        return msg

    df = df.map(clean_value)
    idx = build_idx(df)

    new_data: Dict[str, Dict[str, Any]] = {}
    dropped_rows = 0

    for _, row in df.iterrows():
        ops_struct = row_to_struct(row, idx)
        if not ops_struct:
            dropped_rows += 1
            continue

        collab_key = ops_struct["collab_key"]
        matricule_val = ops_struct.get("matricule_collaborateur")
        year = ops_struct["year"]
        payload = ops_struct["payload"]

        node = new_data.setdefault(collab_key, {"history": {}})
        if matricule_val is not None:
            node["matricule_collaborateur"] = matricule_val
        node["history"].setdefault(year, {}).update(payload)

    if not new_data:
        MASTER_COLL.update_one({"_id": MASTER_ID}, {"$set": {source_label: {}}}, upsert=True)
        msg = f"[{filename}] Aucun enregistrement exploitable: section '{source_label}' vidée."
        return msg

    if replace:
        MASTER_COLL.update_one({"_id": MASTER_ID}, {"$set": {source_label: new_data}}, upsert=True)
        write_msg = f"[{filename}] {len(new_data)} collaborateurs remplacés dans '{source_label}'."
    else:
        flat_set: Dict[str, Any] = {}
        base = f"{source_label}"
        for collab_key, node in new_data.items():
            if "matricule_collaborateur" in node:
                flat_set[f"{base}.{collab_key}.matricule_collaborateur"] = node["matricule_collaborateur"]
            for year, fields in node.get("history", {}).items():
                for k, v in fields.items():
                    flat_set[f"{base}.{collab_key}.history.{year}.{k}"] = v
        MASTER_COLL.update_one({"_id": MASTER_ID}, {"$set": flat_set}, upsert=True)
        write_msg = f"[{filename}] {len(new_data)} collaborateurs fusionnés dans '{source_label}'."

    return write_msg

def delete_employee_by_collab_key(
    *,
    full_name: Optional[str],
    matricule_collaborateur: Union[int, str],
    position: Optional[str],
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Supprime un collaborateur (employee) dans MASTER_COLL à partir de :
    - matricule_collaborateur (clé principale)
    - position (EAP_Professeurs / EAP_Administratif / EAP_EntretiensProfessionnels / Vacataires)
    - full_name (optionnel, pour contrôle / fallback)

    On supprime la *clé* correspondante (ex: "ALi GATORE") dans la section choisie,
    puis on supprime les embeddings associés.
    """

    if matricule_collaborateur is None:
        return {
            "deleted": False,
            "sources": [],
            "reason": "missing_matricule_collaborateur",
        }

    doc = MASTER_COLL.find_one({"_id": MASTER_ID}) or {}
    if not doc:
        return {"deleted": False, "sources": [], "reason": "master_not_found"}

    target_mat = str(matricule_collaborateur)

    sources_touched: List[str] = []
    collab_keys_deleted: List[str] = []
    unset_ops: Dict[str, Any] = {}

    if position and position in VALID_SOURCES:
        sources_to_check = [position]
    else:
        sources_to_check = list(VALID_SOURCES)

    for source_label in sources_to_check:
        section = doc.get(source_label) or {}
        if not isinstance(section, dict):
            continue

        for key, value in section.items():
            mat_db = value.get("matricule_collaborateur")
            mat_db_str = str(mat_db) if mat_db is not None else None

            match = False

            if mat_db_str is not None and mat_db_str == target_mat:
                match = True
            elif full_name and key == full_name:
                match = True
            elif full_name and value.get("summrize", {}).get("identity", {}).get("name") == full_name:
                match = True

            if match:
                unset_ops[f"{source_label}.{key}"] = ""
                sources_touched.append(source_label)
                collab_keys_deleted.append(key)

    if not unset_ops:
        return {
            "deleted": False,
            "sources": [],
            "reason": "not_found_in_sources_by_matricule_or_name",
        }

    MASTER_COLL.update_one(
        {"_id": MASTER_ID},
        {"$unset": unset_ops},
        upsert=False,
    )

    embedding_deleted_total = 0
    for src in sources_touched:
        for collab_key in collab_keys_deleted:
            try:
                embedding_deleted_total += delete_employee_embedding(
                    collab_key, source=src
                )
            except Exception as e:
                print(
                    f"source={src} :: {e}"
                )

    return {
        "deleted": True,
        "sources": sources_touched,
        "collab_keys": collab_keys_deleted,
        "embedding_deleted": embedding_deleted_total > 0,
        "embedding_deleted_count": embedding_deleted_total,
    }

def run_post_import_pipeline(sources: Optional[List[str]] = None, job_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        if job_id:
            push_event(job_id, {"type": "summaries_start"})
            set_stage(job_id, "summaries", 0, {})

        sum_res = summarize_sources_in_batches(sources=sources, batch_size=5, max_workers=5)
        if job_id:
            push_event(job_id, {"type": "summaries_done", "result": sum_res})
            set_stage(job_id, "summaries", 100, sum_res)

        if job_id:
            push_event(job_id, {"type": "clustering_start"})
            set_stage(job_id, "clustering", 10, {})

        clus_res = cluster_and_upload_to_blob(kind=None)

        if job_id:
            push_event(job_id, {"type": "clustering_done", "result": clus_res})
            set_stage(job_id, "clustering", 100, clus_res)

        mark_all_users_update_done()

        if job_id:
            push_event(job_id, {"type": "pipeline_done"})
            set_done(job_id, delete_after=True)

        return {"summaries": sum_res, "clustering": clus_res, "ok": True}
    except Exception as e:
        msg = _humanize_error_message(e)
        mark_all_users_update_error(msg)
        if job_id:
            push_event(job_id, {"type": "pipeline_error", "error": msg})
            set_error(job_id, msg)
        return {"ok": False, "error": msg}

def _collect_employees(doc: Dict[str, Any], sources: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fusionne les collaborateurs présents dans les sections demandées.
    Retourne un dict:
      { collab_key: { 'matricule':..., 'sources': set(...), 'history': {...}, 'summary': {...}? } }
    """
    bucket: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        section = (doc or {}).get(src, {}) or {}
        for ck, node in section.items():
            dst = bucket.setdefault(ck, {"sources": set(), "history": {}, "matricule": None, "summary": None})
            dst["sources"].add(src)
            if "matricule_collaborateur" in node and dst["matricule"] is None:
                dst["matricule"] = node.get("matricule_collaborateur")
            for year, fields in (node.get("history") or {}).items():
                dst["history"].setdefault(year, {}).update(fields)
            if "summrize" in node and not dst["summary"]:
                dst["summary"] = node["summrize"]
    return bucket

def get_all_employees() -> Dict[str, Any]:
    """
    Renvoie TOUT le contenu agrégé (sans filtre), sous forme de liste d'items.
    """
    doc = MASTER_COLL.find_one({"_id": MASTER_ID}) or {}
    bucket = _collect_employees(doc, list(VALID_SOURCES))

    items: List[Dict[str, Any]] = []
    for ck, v in bucket.items():
        years = sorted(v["history"].keys(), reverse=True)
        last_year = years[0] if years else None
        fields_count = sum(len(d) for d in v["history"].values())
        item = {
            "collab_key": ck,
            "matricule": v["matricule"],
            "sources": sorted(list(v["sources"])),
            "history": v["history"],      
            "last_year": last_year,
            "fields_count": fields_count,
            "summary": v["summary"],      
        }
        items.append(item)

    return {"ok": True, "total": len(items), "items": items}
