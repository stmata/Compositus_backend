from fastapi import APIRouter, UploadFile, Query, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
from colorama import init as colorama_init, Fore, Style
from services.employees_service import (
    import_employee_csv,
    run_post_import_pipeline,      
    get_all_employees,
    delete_employee_by_collab_key
)
from services.vacataires_services import get_all_vacataires
from utils.job_tracker import new_job, push_event, set_stage, set_error
from services.job_info_service import get_all_offers
from services.vacataires_services import delete_vacataire_by_collab_key

router = APIRouter()
colorama_init(autoreset=True)

PALETTE = [Fore.MAGENTA, Fore.CYAN, Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.WHITE]

KIND_ALIAS = {
    "professeurs": "EAP_Professeurs",
    "administratifs": "EAP_Administratif",
    "entretiens": "EAP_EntretiensProfessionnels",
    "eap_professeurs": "EAP_Professeurs",
    "eap_administratifs": "EAP_Administratif",
    "eap_administratif": "EAP_Administratif",
    "eap_entretiensprofessionnels": "EAP_EntretiensProfessionnels",
}

CANONICAL_KINDS = {"EAP_Professeurs", "EAP_Administratif", "EAP_EntretiensProfessionnels"}

def canon_kind(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "")
    return KIND_ALIAS.get(key, raw)

def colorize(text: str, color: str, bright: bool = False) -> str:
    return f"{(Style.BRIGHT if bright else '')}{color}{text}{Style.RESET_ALL}"

@router.post("/import_employees")
async def import_employees(
    files: List[UploadFile] = File(...),
    kinds: List[str] = Form(...),
    background_tasks: BackgroundTasks = None,
):
    job_id = new_job()

    if not files:
        set_error(job_id, "no_files")
        push_event(job_id, {"type": "fatal", "error": "no_files"})
        raise HTTPException(400, "aucun fichier fourni.")
    if len(files) != len(kinds):
        set_error(job_id, "mismatch_files_kinds")
        push_event(job_id, {"type": "fatal", "error": "len(files) != len(kinds)"})
        raise HTTPException(400, "files et kinds doivent avoir la même longueur.")

    push_event(job_id, {"type": "start", "msg": "import started", "kinds": kinds})
    set_stage(job_id, "import", 0, {"total": len(files)})
    set_stage(job_id, "save",   0, {"total": len(files)})

    by_kind = {}
    for f, k in zip(files, kinds):
        k_canon = canon_kind(k)
        by_kind.setdefault(k_canon, []).append(f)

    total_files = len(files)
    done_files = 0

    async def _process_one_file(f: UploadFile, kind: str, idx_in_kind: int) -> Dict[str, Any]:
        nonlocal done_files
        filename = f.filename or "<unnamed>"
        push_event(job_id, {"type": "file_start", "kind": kind, "file": filename})

        try:
            content = await f.read()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, import_employee_csv, content, filename, kind, True)

            done_files += 1
            pct = int(done_files * 100 / total_files)

            push_event(job_id, {"type": "file_ok", "kind": kind, "file": filename})

            set_stage(job_id, "import", pct, {"done": done_files, "total": total_files})
            set_stage(job_id, "save",   pct, {"done": done_files, "total": total_files})
            push_event(job_id, {"type": "file_saved", "kind": kind, "file": filename, "pct": pct})

            return {"file": filename, "kind": kind, "success": True, "error": ""}

        except Exception as e:
            done_files += 1
            pct = int(done_files * 100 / total_files)

            push_event(job_id, {"type": "file_err", "kind": kind, "file": filename, "error": str(e)})

            set_stage(job_id, "import", pct, {"done": done_files, "total": total_files})
            set_stage(job_id, "save",   pct, {"done": done_files, "total": total_files})

            return {"file": filename, "kind": kind, "success": False, "error": str(e)}

    async def _process_kind(kind: str, fs: List[UploadFile]) -> Dict[str, Any]:
        tasks = [asyncio.create_task(_process_one_file(f, kind, i)) for i, f in enumerate(fs, start=1)]
        results = await asyncio.gather(*tasks)
        ok = sum(1 for r in results if r["success"])
        return {"kind": kind, "results": results, "ok": ok, "failed": len(fs)-ok}

    per_kind_results = await asyncio.gather(*[
        asyncio.create_task(_process_kind(kind, fs)) for kind, fs in by_kind.items()
    ])

    all_files = [r for kind_res in per_kind_results for r in kind_res["results"]]
    success = sum(1 for r in all_files if r["success"])
    failed = len(all_files) - success

    set_stage(job_id, "import", 100, {"ok": success, "failed": failed, "total": len(all_files)})
    set_stage(job_id, "save",   100, {"ok": success, "failed": failed, "total": len(all_files)})

    push_event(job_id, {"type": "import_end", "ok": success, "failed": failed})

    kinds_processed = [kr["kind"] for kr in per_kind_results if kr["ok"] > 0]

    if background_tasks and kinds_processed:
        push_event(job_id, {"type": "pipeline_scheduled", "kinds": kinds_processed})
        background_tasks.add_task(run_post_import_pipeline, kinds_processed, job_id)
    else:
        push_event(job_id, {"type": "pipeline_skipped", "reason": "no_kind_ok"})

    return JSONResponse(
        status_code=200,
        content={
            "success": failed == 0 and success > 0,
            "summary": {"total": len(all_files), "success": success, "failed": failed},
            "kinds": kinds_processed,
            "job_id": job_id,
        },
    )

@router.delete("/delete_employee")
async def delete_employee(
    category: Optional[str] = Query(
        None,
        description="Optionnel: 'employee' | 'vacataire'. Si None → essaie d'abord employee puis vacataire."
    ),
    full_name: Optional[str] = Query(
        None,
        description="Nom complet du collaborateur (optionnel, utilisé surtout pour les vacataires)."
    ),
    matricule_collaborateur: Optional[str] = Query(
        None,
        description="Matricule collaborateur (utilisé pour les employés internes)."
    ),
    position: Optional[str] = Query(
        None,
        description="Position: 'EAP_Professeurs' | 'EAP_Administratif' | 'EAP_EntretiensProfessionnels' | 'Vacataires'."
    ),
):

    if category in (None, "employee"):
        if not matricule_collaborateur:
            raise HTTPException(status_code=400, detail="matricule_collaborateur is required for employee deletion.")

        result_emp = delete_employee_by_collab_key(
            full_name=full_name,
            matricule_collaborateur=matricule_collaborateur,
            position=position,
            category="employee",
        )
        if result_emp and result_emp.get("deleted"):
            return {
                "deleted": True,
                "category": "employee",
                "details": result_emp,
                "full_name": full_name,
                "matricule_collaborateur": matricule_collaborateur,
                "position": position,
            }

    if category in (None, "vacataire"):
        if not full_name:
            raise HTTPException(status_code=400, detail="full_name is required for vacataire deletion.")

        res_vac = await delete_vacataire_by_collab_key(full_name)
        if res_vac and res_vac.get("ok"):
            return {
                "deleted": True,
                "category": "vacataire",
                "details": res_vac,
                "full_name": full_name,
            }

    raise HTTPException(
        status_code=404,
        detail=f"No employee/vacataire found matching matricule='{matricule_collaborateur}' and/or full_name='{full_name}'."
    )

@router.get("/people_overview")
def list_all_people_and_offers(email: Optional[str] = None):
    employees = get_all_employees()
    vacataires_data = get_all_vacataires(compact=True)
    offers_data = get_all_offers(email=email)

    employees = employees.get("items", employees) if isinstance(employees, dict) else employees
    vacataires = vacataires_data.get("items", vacataires_data) if isinstance(vacataires_data, dict) else vacataires_data
    offers = offers_data.get("items", offers_data) if isinstance(offers_data, dict) else offers_data

    if not isinstance(employees, list):
        employees = [employees]
    if not isinstance(vacataires, list):
        vacataires = [vacataires]
    if not isinstance(offers, list):
        offers = [offers]

    return JSONResponse(
        status_code=200,
        content={
            "internal": employees,
            "vacataires": vacataires,
            "offers": offers,
            "total_internal": len(employees),
            "total_vacataires": len(vacataires),
            "total_offers": len(offers),
            "total_all": len(employees) + len(vacataires) + len(offers),
        },
    )