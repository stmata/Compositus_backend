from utils.prompts import build_summary_prompt, build_prof_summary_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import re
import json
from typing import Dict, Any, Optional, Tuple, List
from openai import AzureOpenAI
import os
from colorama import Fore, Style
from dotenv import load_dotenv
from utils.db_service import MongoDBManager
from utils.job_tracker import set_stage
from threading import Lock

load_dotenv()
mongo = MongoDBManager()
MASTER_COLL = mongo.get_collection("employees")
MASTER_ID = "EAP_Employees"
profs = mongo.get_collection("Professeurs")

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

def _extract_json(text: Optional[str]) -> Optional[dict]:
    """Tolère du bruit autour ; extrait le premier blob JSON plausible."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None

def _humanize_error_message(exc: Exception) -> str:
    """Mappe l'exception vers un message court UX (quota/technique/générique)."""
    s = str(exc).lower()
    if any(x in s for x in ["rate limit", "quota", "insufficient_quota", "overloaded"]):
        return "quota: veuillez ne pas dépasser"
    if any(x in s for x in ["connection", "timeout", "ssl", "dns", "unavailable", "service"]):
        return "technique: veuillez parler avec le dev"
    return "try later"

def llm_call(prompt: str, deployment: str) -> str:
    """Appel Azure OpenAI; log minimal; remonte si client indispo."""
    if client is None:
        raise RuntimeError("AzureOpenAI client unavailable (init failed or env missing).")
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    out = resp.choices[0].message.content
    preview = (out[:240] + "…") if isinstance(out, str) and len(out or "") > 240 else out
    return out or ""

def _iter_all_sources(doc: Dict[str, Any]) -> List[str]:
    return [k for k, v in (doc or {}).items() if isinstance(v, dict) and k.startswith("EAP_")]

def _need_summary(node: Dict[str, Any]) -> bool:
    return isinstance(node, dict) and ("summrize" not in node)

def _summarize_one(source_label: str, collab_key: str, node: Dict[str, Any]) -> Tuple[str, Optional[dict], Optional[str]]:
    user_data = {
        "collaborateur": collab_key,
        "matricule": node.get("matricule_collaborateur"),
        "history": node.get("history", {}),
    }
    raw = llm_call(build_summary_prompt(user_data), deployment=AZURE_DEPLOYMENT_SUMMARY)
    payload = _extract_json(raw)
    if payload is None:
        return collab_key, None, "json_parse_failed"
    return collab_key, payload, None

def summarize_sources_in_batches(
    sources: Optional[List[str]] = None,
    batch_size: int = 5,
    max_workers: int = 5,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    if client is None:
        return {"skipped": "llm_unavailable"}

    doc = MASTER_COLL.find_one({"_id": MASTER_ID}) or {}
    all_sources = _iter_all_sources(doc)
    todo_sources = sources or all_sources

    total_todo = 0
    for src in todo_sources:
        bucket = (doc or {}).get(src, {}) or {}
        keys = [ck for ck, node in bucket.items() if _need_summary(node)]
        total_todo += len(keys)

    processed = 0
    total_ok = 0
    total_err = 0

    if total_todo == 0 and job_id:
        set_stage(job_id, "summaries", 100, {"processed": 0, "total": 0})

    for src in todo_sources:
        bucket = (doc or {}).get(src, {}) or {}
        keys = [ck for ck, node in bucket.items() if _need_summary(node)]
        if not keys:
            continue

        for i in range(0, len(keys), batch_size):
            chunk = keys[i:i + batch_size]
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for ck in chunk:
                    futures[ex.submit(_summarize_one, src, ck, bucket[ck])] = ck

                for fut in as_completed(futures):
                    ck = futures[fut]
                    try:
                        _, payload, err = fut.result()
                        if err or payload is None:
                            MASTER_COLL.update_one(
                                {"_id": MASTER_ID},
                                {"$set": {f"{src}.{ck}.summrize": {
                                    "error": err or "unknown_error",
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                    "model": AZURE_DEPLOYMENT_SUMMARY,
                                    "source": src,
                                }}},
                                upsert=True,
                            )
                            total_err += 1
                        else:
                            MASTER_COLL.update_one(
                                {"_id": MASTER_ID},
                                {"$set": {f"{src}.{ck}.summrize": payload}},
                                upsert=True,
                            )
                            total_ok += 1
                    except Exception as e:
                        MASTER_COLL.update_one(
                            {"_id": MASTER_ID},
                            {"$set": {f"{src}.{ck}.summrize": {
                                "error": f"llm_call_failed: {e}",
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                                "source": src,
                            }}},
                            upsert=True,
                        )
                        total_err += 1

                    processed += 1
                    if job_id and total_todo > 0:
                        set_stage(
                            job_id,
                            "summaries",
                            int(processed * 100 / total_todo),
                            {"processed": processed, "total": total_todo}
                        )
    return {"ok": total_ok, "err": total_err, "total": total_todo, "processed": processed}





def _llm_summarize_prof(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Appelle le LLM avec build_summary_prompt(user_data)
    et retourne un JSON parsé.
    """
    user_data = {
        "NIP": doc.get("NIP"),
        "Campus": doc.get("Campus"),
        "Academic_Title": doc.get("Academic_Title"),
        "Discipline": doc.get("Discipline"),
        "Abstract_Derived_Competencies": doc.get("Abstract_Derived_Competencies"),
        "Teaching_Interests": doc.get("Teaching_Interests"),
        "Unified_Competencies": doc.get("Unified_Competencies"),
        "Academy": doc.get("Academy"),
        "Position_Type": doc.get("Position_Type"),
        "AACSB_Qualification": doc.get("AACSB_Qualification"),
        "CEFDG_Qualification": doc.get("CEFDG_Qualification"),
    }

    prompt = build_prof_summary_prompt(user_data)

    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_SUMMARY,
        messages=[
            {"role": "system", "content": "You are an expert HR summarizer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except Exception:
        return {"error": "invalid_json_from_llm", "raw": text}

def summarize_professors_in_batches(
    max_workers: int = 5,    
    group_size: int = 10,    
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Résume tous les profs en groupes de `group_size`,
    avec au plus `max_workers` groupes en parallèle.

    - Chaque groupe traite ~10 profs séquentiellement.
    - Au max `max_workers` groupes tournent en même temps.
    - Met à jour `summary.llm` dans Mongo.
    - Met à jour la step 'summaries' pour le frontend.
    """

    cursor = profs.find(
        {},
        {
            "_id": 1,
            "NIP": 1,
            "Campus": 1,
            "Academic_Title": 1,
            "Discipline": 1,
            "Abstract_Derived_Competencies": 1,
            "Teaching_Interests": 1,
            "Unified_Competencies": 1,
            "Academy": 1,
            "Position_Type": 1,
            "AACSB_Qualification": 1,
            "CEFDG_Qualification": 1,
        },
    )

    docs = list(cursor)
    total = len(docs)
    done = 0
    errors = 0
    lock = Lock()

    if total == 0:
        if job_id:
            set_stage(job_id, "summaries", 100, {"done": 0, "total": 0})
        return {"total": 0, "summarized": 0, "errors": 0}

    groups: List[List[dict]] = [
        docs[i : i + group_size] for i in range(0, total, group_size)
    ]

    def process_group(group_docs: List[dict]) -> None:
        nonlocal done, errors
        local_nips: List[str] = []

        for doc in group_docs:
            nip = doc.get("NIP")
            try:
                summary_json = _llm_summarize_prof(doc)

                profs.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"summary.llm": summary_json}},
                    upsert=False,
                )

                local_nips.append(str(nip))

                with lock:
                    done += 1
                    if job_id and total > 0:
                        pct = int(done * 100 / total)
                        set_stage(job_id, "summaries", pct, {
                            "done": done,
                            "total": total,
                        })

            except Exception as e:
                with lock:
                    errors += 1

        

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_group, g) for g in groups]
        for fut in as_completed(futures):
            fut.result()

    return {"total": total, "summarized": done, "errors": errors}