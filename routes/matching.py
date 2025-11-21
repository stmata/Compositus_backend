from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional, List
from colorama import init as colorama_init, Fore, Style
from services.matching_service import match_pipeline_async
from services.parsing_service import parse_job_file_with_service_async 
from utils.job_tracker import new_job, set_stage, push_event, set_error, set_done

router = APIRouter()
colorama_init(autoreset=True)

def _c(txt: str, color: str = Fore.WHITE, bright: bool = False) -> str:
    return f"{Style.BRIGHT if bright else ''}{color}{txt}{Style.RESET_ALL}"


router = APIRouter()

@router.post("/match_candidates")
async def match_candidates(
    background: BackgroundTasks,
    job_file: Optional[UploadFile] = File(None),
    job_text: Optional[str] = Form(None),
    email: str = Form(...),
    candidate_source: str = Form("internal"),
    department: str = Form("All"),
    save_results: bool = Form(True),
    cvs: Optional[List[str]] = Form(None),
    offer_mode: str = Form("new"),
    job_filename: Optional[str] = Form(None),
    stream_mode: bool = Form(False),
):
    selected_external_ids = cvs or []

    if offer_mode == "existing":
        if not job_filename:
            raise HTTPException(status_code=400, detail="job_filename requis en mode 'existing'.")

        if not stream_mode:
            return await match_pipeline_async(
                email=email, job_text=None, candidate_source=candidate_source, department=department,
                save_results=save_results, top_k=50, external_ids=selected_external_ids,
                job_filename=job_filename, file_bytes=None, offer_mode="existing",
            )

        job_id = new_job()

        def run_job():
            def on_progress(stage: str, percent: int, meta: Optional[dict]):
                set_stage(job_id, stage, percent, meta or {})
                push_event(job_id, {
                    "type": "stage",
                    "name": stage,
                    "percent": int(percent),
                    "meta": (meta or {})
                })
            async def runner():
                try:
                    res = await match_pipeline_async(
                        email=email, job_text=None, candidate_source=candidate_source, department=department,
                        save_results=save_results, top_k=50, external_ids=selected_external_ids,
                        job_filename=job_filename, file_bytes=None, offer_mode="existing",
                        on_progress=on_progress,
                    )
                    push_event(job_id, {"type": "result", "payload": res})
                    set_done(job_id)
                except Exception as e:
                    set_error(job_id, str(e))

            import asyncio
            asyncio.run(runner())

        background.add_task(run_job)
        return {"ok": True, "job_id": job_id}

    if not job_file and not job_text:
        raise HTTPException(status_code=400, detail="Fournir job_file OU job_text.")
    if job_file and job_text:
        raise HTTPException(status_code=400, detail="Un seul des deux: job_file OU job_text.")

    _job_text: Optional[str] = None
    _file_bytes: Optional[bytes] = None
    _job_filename: Optional[str] = job_filename

    if job_file:
        _job_filename = job_file.filename or "offer.txt"
        content: bytes = await job_file.read()
        _file_bytes = content
        _job_text = await parse_job_file_with_service_async(content, _job_filename)
        if not _job_text:
            raise HTTPException(status_code=502, detail="Service d'extraction indisponible.")
    else:
        _job_text = (job_text or "").strip()
        if not _job_text:
            raise HTTPException(status_code=400, detail="job_text vide.")
        _job_filename = _job_filename or "offer.txt"
        _file_bytes = _job_text.encode("utf-8")

    if not stream_mode:
        return await match_pipeline_async(
            email=email, job_text=_job_text, candidate_source=candidate_source, department=department,
            save_results=save_results, top_k=50, external_ids=selected_external_ids,
            job_filename=_job_filename, file_bytes=_file_bytes, offer_mode="new",
        )

    job_id = new_job()

    def on_progress(stage: str, percent: int, meta: Optional[dict]):
        set_stage(job_id, stage, percent, meta or {})
        push_event(job_id, {
            "type": "stage",
            "name": stage,
            "percent": int(percent),
            "meta": (meta or {})
        })
    async def runner():
        try:
            res = await match_pipeline_async(
                email=email, job_text=_job_text, candidate_source=candidate_source, department=department,
                save_results=save_results, top_k=50, external_ids=selected_external_ids,
                job_filename=_job_filename, file_bytes=_file_bytes, offer_mode="new",
                on_progress=on_progress,
            )
            push_event(job_id, {"type": "result", "payload": res})
            set_done(job_id)
        except Exception as e:
            set_error(job_id, str(e))

    background.add_task(lambda: __import__("asyncio").run(runner()))
    return {"ok": True, "job_id": job_id}

