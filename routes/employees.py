from fastapi import APIRouter, UploadFile, Query, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from colorama import init as colorama_init, Fore, Style
from services.employees_service import (
    get_all_employees,
    delete_employee_by_collab_key
)
from services.Vacataires.vacataires_services import get_all_vacataires, delete_vacataire_by_collab_key
from utils.job_tracker import new_job, push_event, set_stage, set_error
from services.Offers.job_info_service import get_all_offers
import pandas as pd
from io import BytesIO
from services.Professors.prof_processing import (
    build_professors_consolidated_json,
    get_all_professors
)
from services.Professors.prof_pipeline import (
    run_prof_post_import_pipeline
) 
import math
from bson import ObjectId

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

async def read_csv_to_df(upload: UploadFile) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    
    content = await upload.read()
    return pd.read_csv(
        BytesIO(content),
        sep=None,           
        engine="python",    
        dtype=str,
        keep_default_na=False
    )

def canon_kind(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "")
    return KIND_ALIAS.get(key, raw)

def colorize(text: str, color: str, bright: bool = False) -> str:
    return f"{(Style.BRIGHT if bright else '')}{color}{text}{Style.RESET_ALL}"

@router.post("/import_employees", response_model=None)
async def import_employees(
    files: List[UploadFile] = File(...),
    kinds: List[str] = Form(...),
    background_tasks: BackgroundTasks = None,
):
    if not files:
        raise HTTPException(400, "aucun fichier fourni.")

    if len(files) != len(kinds):
        raise HTTPException(400, "files et kinds doivent avoir la même longueur.")

    job_id = new_job()

    total_files = len(files)

    push_event(job_id, {"type": "start", "msg": "import started", "kinds": kinds})
    set_stage(job_id, "import", 0, {"total": total_files})
    set_stage(job_id, "summaries", 0, {"total": 1})
    set_stage(job_id, "clustering", 0, {"total": 1})
    set_stage(job_id, "save", 0, {"total": total_files})
    set_stage(job_id, "total", 0, {"step": "start"})
    prof_files: list[UploadFile] = []
    admin_files: list[UploadFile] = []
    ent_files: list[UploadFile] = []

    for f, k in zip(files, kinds):
        if k == "EAP_Professeurs":
            prof_files.append(f)
        elif k == "EAP_Administratif":
            admin_files.append(f)
        elif k == "EAP_EntretiensProfessionnels":
            ent_files.append(f)

    if not prof_files:
        set_error(job_id, "no_prof_files")
        push_event(job_id, {"type": "fatal", "error": "no_prof_files"})
        raise HTTPException(400, "aucun fichier professeur fourni (EAP_Professeurs).")

    main_prof: UploadFile | None = None
    pub_file: UploadFile | None = None
    research_file: UploadFile | None = None
    teaching_file: UploadFile | None = None

    for f in prof_files:
        name = (f.filename or "").lower()
        if "publication" in name:
            pub_file = f
        elif "research" in name or "iresearch" in name:
            research_file = f
        elif "teaching" in name:
            teaching_file = f
        else:
            main_prof = f

    if not (main_prof and pub_file and research_file and teaching_file):
        set_error(job_id, "missing_prof_csv")
        push_event(
            job_id,
            {
                "type": "fatal",
                "error": "Missing one or more professor CSV files (prof, publication, research, teaching).",
            },
        )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "detail": "Missing one or more professor CSV files (prof, publication, research, teaching).",
                "job_id": job_id,
            },
        )
    push_event(job_id, {"type": "file_start", "kind": "EAP_Professeurs", "file": main_prof.filename})

    df_prof = await read_csv_to_df(main_prof)
    df_pub = await read_csv_to_df(pub_file)
    df_research = await read_csv_to_df(research_file)
    df_teaching = await read_csv_to_df(teaching_file)
    set_stage(job_id, "import", 100, {"done": 4, "total": 4})
    push_event(job_id, {
        "type": "file_saved",
        "kind": "EAP_Professeurs",
        "file": main_prof.filename,
        "pct": 100,
    })

    result_save = build_professors_consolidated_json(
        df_prof,
        df_pub,
        df_research,
        df_teaching,
    )

    set_stage(
        job_id,
        "save",
        100,
        {"ok": result_save["inserted"], "failed": 0, "total": result_save["total"]},
    )
    push_event(job_id, {"type": "import_end", "ok": result_save["inserted"], "failed": 0})

    set_stage(job_id, "total", 33, {"step": "import_done"})

    push_event(job_id, {"type": "pipeline_scheduled", "kinds": ["EAP_Professeurs"]})
    background_tasks.add_task(run_prof_post_import_pipeline, job_id=job_id)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "summary": {
                "total": result_save["total"],
                "success": result_save["inserted"],
                "failed": 0,
            },
            "kinds": ["EAP_Professeurs"],
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

def sanitize_for_json(obj):
    """
    Nettoie récursivement un objet pour être JSON-safe :
    - remplace NaN / inf par None
    - convertit ObjectId en str
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            val = obj.item()
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
    except Exception:
        pass

    return obj

@router.get("/people_overview")
def list_all_people_and_offers(email: Optional[str] = None):
    professors_data = get_all_professors()   
    vacataires_data = get_all_vacataires(compact=True)
    offers_data = get_all_offers(email=email)

    professors = professors_data.get("items", professors_data) if isinstance(professors_data, dict) else professors_data
    vacataires = vacataires_data.get("items", vacataires_data) if isinstance(vacataires_data, dict) else vacataires_data
    offers = offers_data.get("items", offers_data) if isinstance(offers_data, dict) else offers_data

    if not isinstance(professors, list):
        professors = [professors]
    if not isinstance(vacataires, list):
        vacataires = [vacataires]
    if not isinstance(offers, list):
        offers = [offers]
    raw_content = {
            "internal": professors,
            "vacataires": vacataires,
            "offers": offers,
            "total_internal": len(professors),
            "total_vacataires": len(vacataires),
            "total_offers": len(offers),
            "total_all": len(professors) + len(vacataires) + len(offers),
    }

    safe_content = sanitize_for_json(raw_content)

    return JSONResponse(
        status_code=200,
        content=safe_content,
    )


