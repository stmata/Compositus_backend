from __future__ import annotations
import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from services.embedding_service import _download_blob_to_ndarray
from services.parsing_service import parse_one_file_async
from services.Clustering.clustering_service import (
    _build_embed_model as build_embed_model,
    _embed_texts as embed_vectors,
    build_cv_embedding_text,
    _auto_kmeans_vac,
)
from services.Clustering.cluster_storage_service import upsert_single_embedding_and_label2
from utils.db_service import MongoDBManager
from utils.job_tracker import push_event, set_done, set_error
from services.Vacataires.vacataires_blob_service import _azure_delete_by_path, _azure_delete_pdf, _azure_upload_pdf_bytes
from services.Vacataires.vacataires_repository import _exists_collab_key, _exists_pdf_filename
from services.Vacataires.vacataires_llm_service import _llm_cv_extract, _pick_collab_key

load_dotenv()
colorama_init(autoreset=True)

mongo = MongoDBManager()
VAC_COLL = mongo.get_collection("vacataires")

API_KEY = os.getenv("API_KEY")
API_VER = os.getenv("OPENAI_API_VERSION")
API_BASE = (os.getenv("API_BASE") or "").rstrip("/")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")

PDFS_CONTAINER = os.getenv("CVS_CONTAINER", "cvs-pdfs")
VAC_EMB_CONTAINER = os.getenv("VACATAIRES_EMB_CONTAINER", os.getenv("EMB_CONTAINER", "vacataires-embeddings"))
AZURE_DEPLOYMENT_SUMMARY = os.getenv("AZURE_DEPLOYMENT_SUMMARY", "gpt-4o-mini")

ASSIGN_K_SIGMA = float(os.getenv("ASSIGN_K_SIGMA", "1.5"))
ASSIGN_MIN_COS = float(os.getenv("ASSIGN_MIN_COS", "0.92"))
RECLUSTER_NEED_COUNT = int(os.getenv("RECLUSTER_NEED_COUNT", "10"))

if not API_KEY or not API_VER or not API_BASE:
    print(
        f"{Style.BRIGHT}{Fore.YELLOW}{Style.RESET_ALL} Azure OpenAI env manquantes. "
        f"API_KEY={bool(API_KEY)} VER={API_VER!r} BASE={API_BASE!r}"
    )

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

async def ingest_vacataire_cv(
    f: UploadFile,
    uploader_email: Optional[str] = None,
    job_id: Optional[str] = None,
    source_label: str = "Vacataires",
) -> Dict[str, Any]:
    """
    1) Parsing + upload PDF (en parallèle)
    2) LLM extraction (JSON)
    3) Embedding (pour debug uniquement)
    4) Upsert Mongo (sans assignation/clustering, juste needs_recluster=True)

    Événements émis :
      vac_upload_start → vac_parse_start → vac_parsed → vac_llm_start → vac_llm_extracted
      → vac_embedding_start → (vac_embed_debug) → vac_assign_start → vac_assign_done
    """
    if not f:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu.")

    try:
        if job_id:
            push_event(job_id, {"type": "vac_upload_start", "filename": f.filename})
            await asyncio.sleep(0)

        file_bytes = await f.read()
        filename = f.filename or "cv.pdf"
        if "." not in filename:
            filename += ".pdf"

        already = _exists_pdf_filename(filename)
        if already:
            if job_id:
                push_event(job_id, {
                    "type": "vac_duplicate_pdf",
                    "filename": filename,
                    "collab_key": already.get("_id"),
                })
                await asyncio.sleep(0)
            raise HTTPException(
                status_code=409,
                detail=f"Un CV avec ce nom de fichier existe déjà (vacataire: {already.get('_id')}).",
            )

        parse_coro = parse_one_file_async(file_bytes, filename)
        upload_coro = _azure_upload_pdf_bytes(file_bytes, filename)

        if job_id:
            push_event(job_id, {"type": "vac_parse_start", "filename": filename})
            await asyncio.sleep(0)

        parse_obj, blob_name = await asyncio.gather(parse_coro, upload_coro, return_exceptions=False)

        if not isinstance(parse_obj, dict) or not parse_obj.get("ok"):
            await _azure_delete_pdf(blob_name)
            err = (parse_obj or {}).get("error", "parse_failed")
            if job_id:
                push_event(job_id, {"type": "vac_parse_giveup", "error": err, "filename": filename})
                await asyncio.sleep(0)
            raise HTTPException(status_code=502, detail=f"Échec parsing: {err}")

        text = (parse_obj.get("text") or "").strip()
        pdf_blob_path = f"{PDFS_CONTAINER}/{blob_name}"

        if job_id:
            push_event(job_id, {"type": "vac_parsed", "filename": filename})
            await asyncio.sleep(0)
            push_event(job_id, {"type": "vac_llm_start", "filename": filename})
            await asyncio.sleep(0)

        parsed = await _llm_cv_extract(text)
        collab_key = _pick_collab_key(parsed, filename)

        dup = _exists_collab_key(collab_key)
        if dup:
            try:
                await _azure_delete_pdf(blob_name)
            except Exception:
                pass
            if job_id:
                push_event(job_id, {
                    "type": "vac_duplicate_collab",
                    "collab_key": collab_key,
                    "filename": filename
                })
                await asyncio.sleep(0)
            raise HTTPException(status_code=409, detail=f"Vacataire déjà existant: {collab_key}.")

        if job_id:
            push_event(job_id, {
                "type": "vac_llm_extracted",
                "ok": bool(parsed),
                "collab_key": collab_key
            })
            await asyncio.sleep(0)
            push_event(job_id, {"type": "vac_embedding_start", "collab_key": collab_key})
            await asyncio.sleep(0)

        embed_model = build_embed_model()
        emb_text = build_cv_embedding_text(parsed)
        vec = embed_vectors(embed_model, [emb_text])[0]
        if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-6:
            raise HTTPException(status_code=500, detail="Embedding vide/NaN.")
        vec = vec.astype(np.float32)

        if job_id:
            push_event(job_id, {
                "type": "vac_embed_debug",
                "emb_norm": float(np.linalg.norm(vec)),
                "emb_preview": emb_text[:220],
            })
            await asyncio.sleep(0)
            push_event(job_id, {"type": "vac_assign_start", "collab_key": collab_key})
            await asyncio.sleep(0)

        needs_recluster = True

        VAC_COLL.update_one(
            {"_id": collab_key},
            {
                "$set": {
                    "cv": {
                        "filename": filename,
                        "pdf_blob": pdf_blob_path,
                        "uploaded_at": _now_iso(),
                        "uploaded_by": uploader_email,
                        "text_excerpt": (text[:1200] + "…") if len(text) > 1200 else text,
                        "parsed": parsed,
                    },
                    "embedding": {
                        "model": os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS", "text-embedding-3-large"),
                        "blob": None, 
                        "updated_at": _now_iso(),
                    },
                    "embedding_text": emb_text,
                    "needs_recluster": needs_recluster,
                    "updated_at": _now_iso(),
                    "cluster": {
                        "semantic": None,
                        "score": None,
                    },
                }
            },
            upsert=True,
        )

        if job_id:
            push_event(job_id, {
                "type": "vac_assign_done",
                "collab_key": collab_key,
                "assigned": False,
                "cluster": None,
                "score": None,
            })
            await asyncio.sleep(0)

        return {
            "ok": True,
            "collab_key": collab_key,
            "assigned": False,
            "cluster": None,
            "score": None,
            "pdf_blob": pdf_blob_path,
            "needs_recluster": needs_recluster,
        }

    except HTTPException:
        raise
    except Exception as e:
        if job_id:
            push_event(job_id, {"type": "vac_ingest_error", "error": str(e)})
            set_error(job_id, str(e))
        raise

async def run_vacataire_pipeline_for_one_file(
    f: UploadFile,
    uploader_email: Optional[str] = None,
    do_cluster_after: bool = True,
    job_id: Optional[str] = None,
    source_label: str = "Vacataires",
    finalize_job: bool = False,
) -> Dict[str, Any]:
    """
    1) Ingestion d'un CV (parse + LLM + embedding + save)
    2) (optionnel) Reclustering global de TOUS les vacataires
    """
    if job_id:
        push_event(job_id, {"type": "vac_pipeline_start"})

    try:
        ingest_res = await ingest_vacataire_cv(
            f,
            uploader_email=uploader_email,
            job_id=job_id,
            source_label=source_label,
        )

        cluster_res: Dict[str, Any] = {"skipped": True}

        if do_cluster_after:
            if job_id:
                from utils.job_tracker import set_stage as _set_stage
                push_event(job_id, {"type": "vac_cluster_start"})
                _set_stage(job_id, "vac_cluster", 0, {})

            cluster_res = await asyncio.to_thread(recluster_vacataires, source_label)

            if job_id:
                _set_stage(job_id, "vac_cluster", 100, {"reclustered": True})
                push_event(job_id, {"type": "vac_cluster_done", "result": cluster_res})

        if job_id:
            push_event(job_id, {"type": "vac_pipeline_done", "ingest": ingest_res, "clustering": cluster_res})
            if finalize_job:
                push_event(job_id, {"type": "done"})
                set_done(job_id, delete_after=False)

        return {"ok": True, "ingest": ingest_res, "clustering": cluster_res}

    except HTTPException as he:
        if job_id:
            push_event(job_id, {"type": "vac_pipeline_error", "error": he.detail})
        raise
    except Exception as e:
        if job_id:
            push_event(job_id, {"type": "vac_pipeline_error", "error": str(e)})
        raise

def list_vacataires(compact: bool = True) -> Dict[str, Any]:
    items = list(VAC_COLL.find({}))
    out: List[Dict[str, Any]] = []
    for it in items:
        if compact:
            out.append(
                {
                    "collab_key": it.get("_id"),
                    "cluster": (it.get("cluster") or {}).get("semantic"),
                    "score": (it.get("cluster") or {}).get("score"),
                    "pdf": (it.get("cv") or {}).get("filename"),
                    "pdf_blob": (it.get("cv") or {}).get("pdf_blob"),
                    "needs_recluster": it.get("needs_recluster"),
                    "updated_at": it.get("updated_at"),
                }
            )
        else:
            out.append(it)
    return {"ok": True, "total": len(out), "items": out}

def get_all_vacataires(compact: bool = True) -> List[Dict[str, Any]]:
    try:
        items = list(VAC_COLL.find({}))
        out: List[Dict[str, Any]] = []

        for it in items:
            if compact:
                out.append(
                    {
                        "collab_key": it.get("_id"),
                        "cluster": (it.get("cluster") or {}).get("semantic"),
                        "score": (it.get("cluster") or {}).get("score"),
                        "pdf": (it.get("cv") or {}).get("filename"),
                        "parsed": (it.get("cv") or {}).get("parsed"),
                        "needs_recluster": it.get("needs_recluster"),
                        "updated_at": it.get("updated_at"),
                    }
                )
            else:
                out.append(it)

        return out

    except Exception as e:
        return []

def load_vacataire_vectors(
    ids: Optional[List[str]] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, Optional[np.ndarray], str]]:
    query = {}
    if ids:
        query = {"_id": {"$in": list(map(str, ids))}}
    elif exclude_ids:
        query = {"_id": {"$nin": list(map(str, exclude_ids))}}

    projection = {
        "_id": 1,
        "cv.filename": 1,
        "embedding.blob": 1,
        "embedding_text": 1,
    }

    docs = VAC_COLL.find(query, projection)
    out: List[Tuple[str, str, str, Optional[np.ndarray], str]] = []

    for d in docs:
        collab_key = str(d.get("_id"))
        filename = (d.get("cv") or {}).get("filename", "")
        blob_path = (d.get("embedding") or {}).get("blob")
        emb_text = d.get("embedding_text") or ""

        vec: Optional[np.ndarray] = None
        if blob_path:
            vec = _download_blob_to_ndarray(blob_path)
            if vec is not None:
                vec = vec.astype(np.float32)

        out.append(("vacataires", collab_key, filename, vec, emb_text))

    return out

async def delete_vacataire_by_collab_key(
    collab_key: str,
    do_recluster: bool = True,
) -> Dict[str, Any]:
    """
    Supprime un vacataire (par _id = collab_key), son CV dans le blob,
    son embedding, puis recalcule les clusters (va_clusters).
    Si la personne était seule dans un cluster, ce cluster disparaîtra
    naturellement lors du reclustering complet.
    """
    doc = VAC_COLL.find_one({"_id": collab_key})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Vacataire '{collab_key}' introuvable.")

    cv_info = doc.get("cv") or {}
    emb_info = doc.get("embedding") or {}

    pdf_blob_path = cv_info.get("pdf_blob")     
    emb_blob_path = emb_info.get("blob")        

    await _azure_delete_by_path(pdf_blob_path)
    await _azure_delete_by_path(emb_blob_path)

    VAC_COLL.delete_one({"_id": collab_key})

    recluster_result: Dict[str, Any] = {}
    if do_recluster:
        recluster_result = recluster_vacataires("Vacataires")

    return {
        "ok": True,
        "deleted_collab_key": collab_key,
        "deleted_pdf_blob": pdf_blob_path,
        "deleted_emb_blob": emb_blob_path,
        "recluster": recluster_result,
    }

def recluster_vacataires(source_label: str = "Vacataires") -> Dict[str, Any]:
    """
    Recalcule les clusters pour TOUS les vacataires à partir de embedding_text
    (on ne lit plus les anciens blobs qui ont des noms invalides).

    - lit tous les vacataires qui ont embedding_text
    - re-génère les vecteurs via embed_model
    - applique KMeans global (_auto_kmeans_vac)
    - met à jour :
        * cluster.semantic
        * embedding.blob (en les rangeant par cluster avec des noms safe)
        * needs_recluster = False
    """
    docs = list(VAC_COLL.find({"embedding_text": {"$exists": True}}))

    if not docs:
        return {"ok": False, "reason": "no_docs", "total": 0, "clusters": 0}

    embed_model = build_embed_model()

    collab_keys: List[str] = []
    vecs: List[np.ndarray] = []

    for d in docs:
        collab_key = str(d.get("_id"))
        emb_text = (d.get("embedding_text") or "").strip()
        if not emb_text:
            continue

        try:
            v = embed_vectors(embed_model, [emb_text])[0]
            v = v.astype(np.float32)
            if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-6:
                continue
        except Exception as e:
            continue

        collab_keys.append(collab_key)
        vecs.append(v)

    if not vecs:
        return {"ok": False, "reason": "no_vectors", "total": 0, "clusters": 0}

    X = np.stack(vecs, axis=0)
    labels = _auto_kmeans_vac(X)
    labels_list = labels.tolist()
    uniq = sorted(set(int(l) for l in labels_list)) if labels.size else []

    for ck, vec, lbl in zip(collab_keys, vecs, labels_list):
        cluster_id = int(lbl)

        emb_blob = upsert_single_embedding_and_label2(
            source=source_label,
            collab_key=ck,
            vec=vec,
            label=str(cluster_id),  
        )

        VAC_COLL.update_one(
            {"_id": ck},
            {
                "$set": {
                    "cluster.semantic": cluster_id,
                    "cluster.score": None,
                    "needs_recluster": False,
                    "embedding.blob": emb_blob,
                    "updated_at": _now_iso(),
                }
            },
            upsert=False,
        )

    return {
        "ok": True,
        "source": source_label,
        "total": len(collab_keys),
        "clusters": len(uniq),
    }
