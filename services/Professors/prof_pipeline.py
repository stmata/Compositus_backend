from typing import Optional, Dict, Any
from utils.job_tracker import (
    push_event,
    set_stage,
    set_done,
    set_error,
)
from services.user_flags_service import (
    mark_all_users_update_done,
    mark_all_users_update_error,
)
from services.summary_service import _humanize_error_message, summarize_professors_in_batches
from services.Professors.prof_processing import run_prof_clustering_pipeline

from typing import Optional, Dict, Any
from utils.job_tracker import (
    push_event,
    set_stage,
    set_done,
    set_error,
)
from services.user_flags_service import (
    mark_all_users_update_done,
    mark_all_users_update_error,
)
from services.summary_service import (
    _humanize_error_message,
    summarize_professors_in_batches,
)
from services.Professors.prof_processing import run_prof_clustering_pipeline


def run_prof_post_import_pipeline(job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Pipeline après import des professeurs :
    1) Génération des résumés
    2) Clustering
    3) Mise à jour des flags
    """

    result: Dict[str, Any] = {
        "summaries": None,
        "clustering": None,
    }

    try:

        if job_id:
            push_event(job_id, {"type": "summaries_start"})
            set_stage(job_id, "summaries", 0, {})

        sum_res = summarize_professors_in_batches(
            max_workers=8,
            group_size = 10,
            job_id=job_id,
        )

        result["summaries"] = sum_res

        if job_id:
            push_event(job_id, {"type": "summaries_done", "result": sum_res})
            set_stage(job_id, "summaries", 100, sum_res)

        if job_id:
            push_event(job_id, {"type": "clustering_start"})
            set_stage(job_id, "clustering", 0, {})


        cluster_res = run_prof_clustering_pipeline(job_id=job_id)

        result["clustering"] = cluster_res
        if job_id:
            push_event(job_id, {"type": "clustering_done", "result": cluster_res})
            set_stage(job_id, "clustering", 100, cluster_res)
            set_done(job_id, result)

        mark_all_users_update_done()
        return result

    except Exception as exc:
        print("\n[PIPELINE] ✗ Erreur détectée dans le pipeline :", exc)

        if job_id:
            human_msg = _humanize_error_message(exc)
            push_event(job_id, {"type": "error", "message": human_msg})
            set_error(job_id, human_msg)

        mark_all_users_update_error()

        raise

