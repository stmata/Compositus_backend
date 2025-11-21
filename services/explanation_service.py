from __future__ import annotations
from typing import Dict, Any, List
from services.llm_service import _llm, _extract_json
from utils.prompts import build_match_expl_prompt

def explain_topN(job_info: Dict[str, Any], filtered_ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pour chaque candidat dans filtered_ranked, on appelle le LLM pour expliquer
    pourquoi il matche (ou non) avec l'offre.

    ATTENTION :
    - Le LLM NE DOIT PAS calculer de score. Il met "match_score": null.
    - C'EST ICI que l'on injecte le match_score final (en pourcentage),
      à partir de combined_score ou, à défaut, du score d'embedding.
    """
    out: List[Dict[str, Any]] = []


    for r in filtered_ranked:
        nm = r.get("collab_key") or r.get("name") or "Unknown"

        emb_score_norm01 = float(r.get("score", 0.0))

        profile_text = r.get("profile_text", "") or ""

        candidate_payload = {
            "name": nm,
            "profile_text": profile_text,
            "raw": r,  
        }

        prompt = build_match_expl_prompt(
            job_description=job_info,
            candidate=candidate_payload,
        )

        try:
            raw = _llm(prompt)
            js = _extract_json(raw)
        except Exception as e:
            js = None

        if not isinstance(js, dict):
            js = {}

        for lang in ("en", "fr"):
            lang_obj = js.get(lang)
            if not isinstance(lang_obj, dict):
                lang_obj = {
                    "name": nm,
                    "summary": {
                        "skills_matched": [],
                        "skills_missing": [],
                        "experience_matched": [],
                        "experience_missing": [],
                        "qualifications_matched": [],
                        "qualifications_missing": [],
                    },
                    "reasoning": "",
                    "suggestions": [],
                }
            lang_obj["name"] = nm
            js[lang] = lang_obj

        out.append({
            "name": nm,
            "embedding_score": emb_score_norm01,   
            "llm": js,                            
        })

    return out
