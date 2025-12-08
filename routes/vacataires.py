from __future__ import annotations
import asyncio
from typing import List, Optional, Set, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from colorama import init as colorama_init, Fore
from utils.job_tracker import new_job, push_event, set_stage, set_done, set_error
from services.Vacataires.vacataires_services import _exists_pdf_filename, run_vacataire_pipeline_for_one_file, ingest_vacataire_cv, list_vacataires
from services.Vacataires.vacataires_services import recluster_vacataires as recluster_vacataires_service

router = APIRouter()
colorama_init(autoreset=True)
_BACKGROUND_TASKS: Set[asyncio.Task] = set()

class _MemoryUpload:
    """Rejouer un UploadFile depuis la RAM (évite I/O closed file)."""
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._data = data
    async def read(self) -> bytes:
        return self._data

@router.post("/vacataires/upload")
async def upload_vacataires(
    files: List[UploadFile] = File(...),
    uploader_email: Optional[str] = Query(None),
    recluster_after: bool = Query(True, description="Reclustering auto si seuil atteint"),
    source_label: str = Query("Vacataires", description="Nom logique de la source"),
    max_parallel: int = Query(4, ge=1, le=10, description="Nb max en // (défaut 4)"),
    drop_jobs: str = Query("none", description="none|collection_if_idle|force_collection"),
):
    """
    Batch → 1 job_id.
    - Bufferise tous les fichiers avant la réponse (évite 'I/O operation on closed file')
    - Lance le pipeline en tâche de fond
    - Réponse immédiate: 202 + { job_id, accepted[], rejected[], total }
    """
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu.")

    job_id = new_job()
    push_event(job_id, {"type": "vac_batch_start", "count": len(files)})
    set_stage(job_id, "vac_pipeline", 0, {"total_files": len(files)})
    set_stage(job_id, "vac_upload", 0, {})
    set_stage(job_id, "vac_summaries", 0, {})
    set_stage(job_id, "vac_cluster", 0, {})
    set_stage(job_id, "vac_saving", 0, {})

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    buffered: List[_MemoryUpload] = []

    async def _buffer_one(f: UploadFile):
        filename = f.filename or "cv.pdf"
        try:
            data = await f.read()
            if not data:
                return None, {"filename": filename, "reason": "empty_file"}, False
            mem = _MemoryUpload(filename, data)
            return mem, {"filename": filename, "size": len(data)}, True
        except Exception as e:
            return None, {"filename": filename, "reason": str(e)}, False

    results = await asyncio.gather(*[_buffer_one(f) for f in files])

    for mem, info, is_ok in results:
        if is_ok and mem is not None:
            buffered.append(mem)
            accepted.append(info)
        else:
            rejected.append(info)

    duplicates: List[Dict[str, Any]] = []
    unique_buffered: List[_MemoryUpload] = []
    unique_accepted: List[Dict[str, Any]] = []

    seen_filenames: set[str] = set()

    for mem, info in zip(buffered, accepted):
        filename = mem.filename

        existing = _exists_pdf_filename(filename)
        if existing:
            duplicates.append({
                "filename": filename,
                "reason": "exists_in_db",
                "collab_key": existing.get("_id"),
            })
            continue

        if filename in seen_filenames:
            duplicates.append({
                "filename": filename,
                "reason": "duplicate_in_batch",
            })
            continue

        seen_filenames.add(filename)
        unique_buffered.append(mem)
        unique_accepted.append(info)

    buffered = unique_buffered
    accepted = unique_accepted

    total = len(accepted) + len(rejected) + len(duplicates)

    push_event(job_id, {
        "type": "vac_batch_buffered",
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duplicates": len(duplicates),
    })

    if duplicates:
        push_event(job_id, {
            "type": "vac_duplicates_detected",
            "items": duplicates,
        })


    sem = asyncio.BoundedSemaphore(max_parallel)

    async def _process_all():
        try:
            total_files = len(buffered)
            had_errors = False
            first_error_msg = None
            done = 0
            success = 0 

            lock = asyncio.Lock()
            sem = asyncio.Semaphore(max_parallel)

            async def _one(mem: _MemoryUpload):
                nonlocal done, had_errors, first_error_msg, success
                async with sem:
                    push_event(job_id, {"type": "vac_file_start", "filename": mem.filename})
                    try:
                        await run_vacataire_pipeline_for_one_file(
                            f=mem,
                            uploader_email=uploader_email,
                            do_cluster_after=recluster_after,
                            job_id=job_id,
                            source_label=source_label,
                            finalize_job=False,
                        )
                        success += 1
                        push_event(job_id, {
                            "type": "vac_file_done",
                            "filename": mem.filename,
                        })
                    except HTTPException as he:
                        had_errors = True
                        msg = he.detail if hasattr(he, "detail") else str(he)
                        if not first_error_msg:
                            first_error_msg = msg
                        push_event(job_id, {
                            "type": "vac_file_error",
                            "filename": mem.filename,
                            "error": msg,
                        })
                    except Exception as e:
                        had_errors = True
                        msg = str(e)
                        if not first_error_msg:
                            first_error_msg = msg
                        push_event(job_id, {
                            "type": "vac_file_error",
                            "filename": mem.filename,
                            "error": msg,
                        })

                    async with lock:
                        done += 1
                        set_stage(
                            job_id,
                            "vac_upload",
                            int(done * 100 / max(1, total_files)),
                            {"file": mem.filename, "done": done, "total": total_files},
                        )

            await asyncio.gather(*[_one(mem) for mem in buffered])

            failed = total_files - success
            push_event(job_id, {
                "type": "vac_batch_done",
                "total": total_files,
                "success": success,
                "failed": failed,
            })

            if success == 0 and had_errors:
                set_error(job_id, first_error_msg or "all files failed")
            else:
                set_stage(job_id, "vac_pipeline", 100, {
                    "success": success,
                    "failed": failed,
                })
                push_event(job_id, {
                    "type": "done",
                    "success": success,
                    "failed": failed,
                })
                set_done(job_id, delete_after=False, drop_mode=drop_jobs)

        except Exception as e:
            push_event(job_id, {"type": "vac_batch_error", "error": str(e)})
            set_error(job_id, str(e))
        finally:
            _BACKGROUND_TASKS.discard(asyncio.current_task())

    t = asyncio.create_task(_process_all())
    _BACKGROUND_TASKS.add(t)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "accepted": accepted, "rejected": rejected, "duplicates": duplicates, "total": total},
    )

@router.post("/vacataires/ingest_one")
async def ingest_one_vacataire(
    file: UploadFile = File(...),
    uploader_email: Optional[str] = Query(None),
    source_label: str = Query("Vacataires"),
    recluster_after: bool = Query(False),
    drop_jobs: str = Query("none", description="none|collection_if_idle|force_collection"),
):
    job_id = new_job()
    try:
        res = await ingest_vacataire_cv(
            f=file,
            uploader_email=uploader_email,
            job_id=job_id,
            source_label=source_label,
        )
        push_event(job_id, {"type": "done"})
        set_done(job_id, delete_after=False, drop_mode=drop_jobs)
        return {"ok": True, "job_id": job_id, "ingest": res}
    except Exception as e:
        push_event(job_id, {"type": "vac_ingest_error", "error": str(e)})
        set_error(job_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vacataires")
def get_vacataires(compact: bool = True):
    return list_vacataires(compact=compact)

@router.post("/vacataires/recluster")
def recluster_vacataires_route(source_label: str = Query("Vacataires")):
    res = recluster_vacataires_service(source_label)
    return {"ok": True, "result": res}
