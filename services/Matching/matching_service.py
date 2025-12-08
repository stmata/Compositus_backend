from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np, asyncio
from colorama import Fore
from utils.console import _c
from services.Offers.job_info_service import extract_job_info, load_job_offer_assets_by_filename, save_job_offer_if_new, job_info_to_embedding_text
from services.embedding_service import _embed_texts2
from services.candidate_service import load_candidate_vectors, _is_all, _normalize_dept
from services.Vacataires.vacataires_services import load_vacataire_vectors
from services.ranking_service import rank_candidates_for_job
from services.Matching.explanation_service import explain_topN  
from services.history_service import save_to_user_upload_history
from utils.db_service import MongoDBManager
import os
from services.Professors.prof_processing import build_prof_profile_text, load_professor_vectors

mongo = MongoDBManager()
VAC_COLL = mongo.get_collection("vacataires")
MASTER_COLL = mongo.get_collection("employees")
ProgressCB = Callable[[str, int, Optional[dict]], None]  
users = mongo.get_collection("Professeurs")

EXPLAIN_TOP_N = int(os.getenv("EXPLAIN_TOP_N"))
EMBEDDING_SCORE_MIN = float(os.getenv("EMBEDDING_SCORE_MIN"))

def build_vac_profile_text(vac_doc: Dict[str, Any]) -> str:
    """
    Construit un texte profil pour un vacataire à partir du document Mongo.
    - Utilise cv.parsed si dispo (JSON complet),
    - sinon fallback sur text_excerpt.
    """
    cv = vac_doc.get("cv", {})
    parsed = cv.get("parsed") or {}
    if parsed:
        return parsed
    excerpt = cv.get("text_excerpt") or ""
    return excerpt or "(no parsed data for this candidate)"

def build_employee_profile_text(emp_profile: Dict[str, Any]) -> str:
    """
    Construit un texte profil pour un employé interne à partir du bloc:
      MASTER_COLL['_id'='EAP_Employees']['EAP_EntretiensProfessionnels'][<name>]
    - Utilise 'summrize' si dispo,
    - sinon history ou tout le bloc.
    """
    summ = emp_profile.get("summrize", {}) or emp_profile.get("summarize", {}) or {}
    return summ

async def match_pipeline_async(
    *,
    email: str,
    job_text: Optional[str] = None,
    candidate_source: str = "internal",
    department: str = "All",
    save_results: bool = True,
    top_k: int = 50,
    external_ids: Optional[List[str]] = None,
    job_filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    offer_mode: str = "new",
    on_progress: Optional[ProgressCB] = None,
) -> Dict[str, Any]:
 
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 50

    emit = on_progress or (lambda *a, **k: None)
    external_ids = external_ids or []
    dep_norm = _normalize_dept(department)
    save_meta: Optional[Dict[str, Any]] = None 
    is_new_mode = (offer_mode != "existing")

    emit("init", 5, {"candidate_source": candidate_source, "department": dep_norm, "offer_mode": offer_mode})

    job_info: Dict[str, Any] = {}
    job_vec: Optional[np.ndarray] = None
    job_raw_text: str = job_text or ""

    if offer_mode == "existing":
        emit("load_existing_offer", 10, {"filename": job_filename})
        if not job_filename:
            return {"ok": False, "error": "missing_filename_for_existing_offer"}
        job_info, job_vec, job_raw_text = await load_job_offer_assets_by_filename(job_filename)
        if job_vec is None or job_vec.ndim != 1:
            return {"ok": False, "error": "existing_offer_embedding_missing"}
    else:
        emit("parse_offer", 15, None)
        job_info = extract_job_info(job_raw_text) or {}
        if not job_info:
            return {"ok": False, "error": "job_info_extraction_failed"}

        emit("embed_offer", 25, {"fields": list(job_info.keys())})
        emb_text = job_info_to_embedding_text(job_info)
        job_vec = _embed_texts2([emb_text])[0] 
        if save_results:
            emit("save_offer", 30, {"intended_filename": job_filename or "offer.txt"})
            _bytes = file_bytes or (job_raw_text or "").encode("utf-8")
            _fname = (job_filename or "offer.txt")
            if "." not in _fname:
                _fname += ".txt"
            try:
                save_meta = await save_job_offer_if_new(
                    file_bytes=_bytes,
                    filename=_fname,
                    job_text=job_raw_text,
                    job_info=job_info,
                    emb_text=emb_text,
                    job_vec=job_vec,
                    uploader=email,
                    department=dep_norm,
                    candidate_source=candidate_source,
                )
                emit("save_offer", 35, {
                    "saved": not save_meta.get("already_exists", False),
                    "_id": save_meta.get("_id")
                })
            except Exception as e:
                emit("save_offer", 35, {"error": str(e)})
    emit("load_pool", 40, {"candidate_source": candidate_source})
    if candidate_source == "external":
        raw_pool = load_vacataire_vectors(exclude_ids=external_ids)
    elif candidate_source == "all":
        internal_pool = load_professor_vectors(dep_norm)
        external_pool = load_vacataire_vectors(exclude_ids=external_ids)
        raw_pool = internal_pool + external_pool
    else:
        raw_pool = load_professor_vectors(dep_norm)
    need_idx = [i for i, (_, _, _, v, txt) in enumerate(raw_pool) if (v is None and (txt or "").strip())]
    emit("embed_missing", 50, {"need": len(need_idx)})
    to_embed = [raw_pool[i][4] for i in need_idx]
    vecs = _embed_texts2(to_embed) if to_embed else np.zeros((0, 1536), dtype=np.float32)

    finals: List[Tuple[str, str, str, np.ndarray, str]] = []
    embed_ptr = 0

    for (src, ck, name, v, txt) in raw_pool:
        src = str(src)
        ck = str(ck)
        name = str(name or "")
        txt = txt or ""

        if v is None and txt.strip():
            v = vecs[embed_ptr].astype(np.float32)
            embed_ptr += 1

        if v is not None:
            finals.append((src, ck, name, v, txt))
    def _hist_kwargs():
        return dict(
            offer_mode=offer_mode,
            is_new_offer=is_new_mode,
        )

    if not finals:
        emit("rank", 60, {"candidates": 0})
        
        save_to_user_upload_history(
            email,
            candidate_source,
            ("All" if _is_all(dep_norm) else dep_norm),
            job_info,
            [],
            **_hist_kwargs(),
        )
        emit("done", 100, {"results": 0, "note": "no candidates with embeddings"})
        return {"ok": True, "results": [], "job_description": job_info, "note": "no candidates with embeddings"}

    emit("rank", 70, {"candidates": len(finals)})
    ranked = rank_candidates_for_job(job_vec, finals, top_k=top_k)
    filtered = [r for r in ranked if r["score"] >= EMBEDDING_SCORE_MIN]
    emit("threshold", 75, {"kept": len(filtered), "min": EMBEDDING_SCORE_MIN})
    if not filtered:
        save_to_user_upload_history(
            email,
            candidate_source,
            ("All" if _is_all(dep_norm) else dep_norm),
            job_info,
            [],
            **_hist_kwargs(),
        )
        emit("done", 100, {"results": 0, "note": "no matches exist"})
        return {"ok": True, "job_description": job_info, "results": [], "note": "no matches exist"}

    hydrated_for_llm: List[Dict[str, Any]] = []
    N = min(len(filtered), EXPLAIN_TOP_N)

    for r in filtered[:N]:
        src = r.get("source", "")              
        collab_key = r.get("collab_key") or r.get("NIP") 
        profile_text = "(no profile data)"
        extra_profile: Dict[str, Any] = {}
        try:
            if src == "vacataires":
                vac_doc = VAC_COLL.find_one({"_id": collab_key}) or {}
                if not vac_doc:
                    vac_doc = VAC_COLL.find_one({"cv.filename": r.get("name")}) or {}
                profile_text = build_vac_profile_text(vac_doc)
                extra_profile = {"vac_doc": vac_doc}

            else:
               
                prof_doc = users.find_one({"NIP": collab_key}) or {}
                profile_text = build_prof_profile_text(prof_doc)
                extra_profile = {"prof_doc": prof_doc}
        except Exception as e:
            print(_c(f"hydrate failed for {collab_key}: {e}", Fore.YELLOW))

        r_expl = dict(r)
        r_expl["profile_text"] = profile_text
        r_expl["extra_profile"] = extra_profile  

        hydrated_for_llm.append(r_expl)

    emit("explain", 85, {"N": len(hydrated_for_llm)})
    results = await asyncio.to_thread(explain_topN, job_info, hydrated_for_llm)

    emit("save_history", 92, {"N": len(results)})
    save_to_user_upload_history(
        email,
        candidate_source,
        ("All" if _is_all(dep_norm) else dep_norm),
        job_info,
        results,
        **_hist_kwargs(),
    )

    emit("done", 100, {"results": len(results)})
    return {"ok": True, "job_description": job_info, "results": results}

