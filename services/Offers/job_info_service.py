from __future__ import annotations
import math, re, hashlib,  numpy as np, io
from colorama import Fore
from utils.console import _c
from services.llm_service import _llm, _extract_json
from services.chunking_service import _semantic_chunks
from utils.prompts import job_info_prompt
from utils.db_service import MongoDBManager
import os, re, io, hashlib, mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings
from fastapi import HTTPException
from services.embedding_service import adownload_blob_to_ndarray

mongo = MongoDBManager()
JOBS = mongo.get_collection("jobs_offers")
CTX_MAX_TOKENS = int(os.getenv("CTX_MAX_TOKENS")) 
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS"))
TARGET_CHARS = int(os.getenv("TARGET_CHARS"))
CHUNK_PREVIEW_MAX = int(os.getenv("CHUNK_PREVIEW_MAX"))

def collapse_whitespace(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n\s*\r?\n+", "\n\n", s)
    return s.strip()

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)  

def _job_info_chunk_prompt(section_text: str) -> str:
    return f"""
Condense the following job description section into a compact factual paragraph (<= 200 words).
Focus on duties, must-have skills, constraints, environment, seniority, and domain hints.
No lists, no JSON, no hallucinations.
Return ONLY the paragraph.

SECTION:
{section_text}
""".strip()

def extract_job_info(job_text: str) -> Dict[str, Any]:
    text = collapse_whitespace(job_text or "")
    if not text:
        return {}

    toks = estimate_tokens(text)

    if toks <= CTX_MAX_TOKENS:
        out = _extract_json(_llm(job_info_prompt(text)))
        return out

    chunks = _semantic_chunks(text, max_chunks=MAX_CHUNKS, target_chars=TARGET_CHARS)

    summaries: List[str] = []
    for i, ch in enumerate(chunks, 1):
        preview = ch[:CHUNK_PREVIEW_MAX] + ("…[truncated]" if len(ch) > CHUNK_PREVIEW_MAX else "")
        try:
            part = _llm(_job_info_chunk_prompt(ch)).strip()
            if part:
                summaries.append(part)
            else:
                print(_c(f"output {i}", Fore.YELLOW))
        except Exception as e:
            print(_c(f"chunk {i} failed: {e}", Fore.RED))

    merged = "\n\n".join(summaries) if summaries else text[:15_000]
    out = _extract_json(_llm(job_info_prompt(merged)))
    return out

def job_info_to_embedding_text(job_info: Dict[str, Any]) -> str:
    data = job_info.get("en") or job_info.get("fr") or {}

    def _str(x: Any) -> str:
        return x if isinstance(x, str) else ""

    def _list(x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _first_nonempty(*vals: Any) -> str:
        for v in vals:
            s = _str(v)
            if s:
                return s
        return ""

    parts: List[str] = []

    jt = data.get("job_title")
    if isinstance(jt, dict):
        title_bits = [
            _str(jt.get("normalized")) or _str(jt.get("raw")),
            _str(jt.get("seniority")),
        ]
        specs = ", ".join([_str(s) for s in _list(jt.get("specializations")) if _str(s)])
        aliases = ", ".join([_str(s) for s in _list(jt.get("aliases")) if _str(s)])
        if specs:
            title_bits.append(f"specializations: {specs}")
        if aliases:
            title_bits.append(f"aliases: {aliases}")
        title_line = " | ".join([b for b in title_bits if b])
    else:
        title_line = _str(jt)
    if title_line:
        parts.append(title_line)

    sd = data.get("short_description")
    if isinstance(sd, dict):
        sd_bits = [
            _str(sd.get("summary")),
            f"team: {_str(sd.get('team_context'))}" if _str(sd.get("team_context")) else "",
            f"impact: {_str(sd.get('business_impact'))}" if _str(sd.get("business_impact")) else "",
        ]
        domains = ", ".join([_str(d) for d in _list(sd.get("domain_keywords")) if _str(d)])
        if domains:
            sd_bits.append(f"domains: {domains}")
        sd_line = " | ".join([b for b in sd_bits if b])
    else:
        sd_line = _str(sd)
    if sd_line:
        parts.append(sd_line)

    resp_lines: List[str] = []
    for r in _list(data.get("responsibilities")):
        if isinstance(r, dict):
            stmt = _str(r.get("statement"))
            cats = _str(r.get("category"))
            kpis = ", ".join([_str(k) for k in _list(r.get("kpis")) if _str(k)])
            tools = ", ".join([_str(t) for t in _list(r.get("tools")) if _str(t)])
            freq = _str(r.get("frequency"))
            tag = _str(r.get("seniority_tag"))
            line = "; ".join(
                [x for x in [
                    stmt,
                    f"cat={cats}" if cats else "",
                    f"kpis={kpis}" if kpis else "",
                    f"tools={tools}" if tools else "",
                    f"freq={freq}" if freq else "",
                    f"role={tag}" if tag else "",
                ] if x]
            )
        else:
            line = _str(r)
        if line:
            resp_lines.append(line)
    if resp_lines:
        parts.append("\n".join(resp_lines))

    skill_lines: List[str] = []
    for s in _list(data.get("required_skills")):
        if isinstance(s, dict):
            name = _str(s.get("name")) or _str(s.get("normalized"))
            if not name:
                continue
            sub = ", ".join([_str(x) for x in _list(s.get("subskills")) if _str(x)])
            ver = ", ".join([_str(x) for x in _list(s.get("versions_or_flavors")) if _str(x)])
            cat = _str(s.get("category"))
            typ = _str(s.get("type"))
            prof = _str(s.get("proficiency"))
            yrs = s.get("years_min")
            rec = s.get("recency_months_max")
            must = bool(s.get("must_have"))
            w = s.get("weight")
            tag_bits = []
            if must:
                tag_bits.append("must-have")
            if prof:
                tag_bits.append(f"level={prof}")
            if yrs is not None:
                tag_bits.append(f"years_min={yrs}")
            if rec is not None:
                tag_bits.append(f"recency<= {rec}m")
            if w is not None:
                tag_bits.append(f"w={w:.2f}")
            if cat:
                tag_bits.append(f"cat={cat}")
            if typ:
                tag_bits.append(f"type={typ}")
            if ver:
                tag_bits.append(f"versions={ver}")
            if sub:
                tag_bits.append(f"subskills={sub}")

            rep = 1
            try:
                if isinstance(w, (int, float)):
                    rep = max(1, min(5, int(round(1 + 2 * float(w)))))
            except Exception:
                pass

            core = (name + " ") * rep
            line = (core.strip() + " :: " + ", ".join(tag_bits)).strip(" :")
        else:
            line = _str(s)
        if line:
            skill_lines.append(line)
    if skill_lines:
        parts.append("\n".join(skill_lines))

    de = data.get("desired_experience")
    if isinstance(de, dict):
        de_bits = []
        for key in ["years_min", "years_max", "work_model", "client_facing", "travel_percent_max"]:
            val = de.get(key)
            if val not in (None, ""):
                de_bits.append(f"{key}={val}")
        for key in ["industries", "domains", "methodologies", "environments", "security_clearance", "notable_project_examples"]:
            arr = [ _str(x) for x in _list(de.get(key)) if _str(x) ]
            if arr:
                de_bits.append(f"{key}=" + ", ".join(arr))
        loc = de.get("location")
        if isinstance(loc, dict):
            loc_bits = [ _str(loc.get("city")), _str(loc.get("region_or_state")), _str(loc.get("country")) ]
            loc_line = ", ".join([x for x in loc_bits if x])
            if loc_line:
                de_bits.append(f"location={loc_line}")
        lead = de.get("leadership")
        if isinstance(lead, dict):
            pmin = lead.get("people_managed_min")
            lmin = lead.get("projects_led_min")
            lb = []
            if pmin not in (None, ""):
                lb.append(f"people_managed_min={pmin}")
            if lmin not in (None, ""):
                lb.append(f"projects_led_min={lmin}")
            if lb:
                de_bits.append("leadership=" + ", ".join(lb))
        tsr = de.get("team_size_range")
        if isinstance(tsr, list) and len(tsr) == 2:
            de_bits.append(f"team_size_range=[{tsr[0]}, {tsr[1]}]")

        if de_bits:
            parts.append("desired_experience: " + " | ".join(de_bits))
    else:
        if _str(de):
            parts.append(_str(de))

    q = data.get("qualifications")
    if isinstance(q, dict):
        q_bits = []
        edu = q.get("education")
        if isinstance(edu, dict):
            lvl = edu.get("degree_level_min")
            fos = [ _str(x) for x in _list(edu.get("fields_of_study")) if _str(x) ]
            seg = []
            if lvl:
                seg.append(f"degree_min={lvl}")
            if fos:
                seg.append("fields=" + ", ".join(fos))
            if seg:
                q_bits.append("education: " + " | ".join(seg))

        certs = []
        for c in _list(q.get("certifications")):
            if isinstance(c, dict):
                nm = _str(c.get("name"))
                iss = _str(c.get("issuer"))
                req = c.get("required")
                tag = nm
                if iss:
                    tag += f" ({iss})"
                if req is True:
                    tag += " [required]"
                certs.append(tag)
            else:
                certs.append(_str(c))
        if certs:
            q_bits.append("certs: " + ", ".join([x for x in certs if x]))

        langs = []
        for l in _list(q.get("languages")):
            if isinstance(l, dict):
                nm = _str(l.get("name"))
                lv = _str(l.get("level"))
                langs.append(f"{nm}:{lv}" if nm else "")
            else:
                langs.append(_str(l))
        if [x for x in langs if x]:
            q_bits.append("languages: " + ", ".join([x for x in langs if x]))

        wa = []
        for w in _list(q.get("work_authorization")):
            if isinstance(w, dict):
                country = _str(w.get("country"))
                req = w.get("required")
                if country:
                    wa.append(f"{country}:{'required' if req else 'optional'}")
            else:
                wa.append(_str(w))
        if wa:
            q_bits.append("work_auth: " + ", ".join(wa))

        for key in ["background_checks", "physical_requirements", "other"]:
            arr = [ _str(x) for x in _list(q.get(key)) if _str(x) ]
            if arr:
                q_bits.append(f"{key}: " + ", ".join(arr))

        if q_bits:
            parts.append("\n".join(q_bits))
    else:
        if _str(q):
            parts.append(_str(q))

    icon = (
    job_info.get("react_icon_import")
    or job_info.get("en", {}).get("react_icon_import")
    or job_info.get("fr", {}).get("react_icon_import")
    or "import { AiOutlineInfoCircle } from 'react-icons/ai';"
    )

    if isinstance(icon, str) and icon:
        parts.append(f"icon={icon}")

    return "\n".join([p for p in parts if _str(p)]).strip()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _guess_content_type(filename: str) -> str:
    ctype, _ = mimetypes.guess_type(filename or "")
    return ctype or "application/octet-stream"

def _normalize_for_hash(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def _sha256_text(s: Optional[str]) -> str:
    return hashlib.sha256(_normalize_for_hash(s).encode("utf-8")).hexdigest()

def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", (s or "").strip()) or "x"

async def _ensure_container(bsc: BlobServiceClient, container_name: str) -> None:
    cc = bsc.get_container_client(container_name)
    try:
        await cc.create_container()
    except ResourceExistsError:
        pass

async def save_job_offer_if_new(
    *,
    file_bytes: bytes,
    filename: str,
    job_text: str,
    job_info: Dict[str, Any],
    emb_text: str,
    job_vec: np.ndarray,
    uploader: str,
    department: str,
    candidate_source: str,
) -> Dict[str, Any]:
    """
    - Upsert par `filename` (PAS de doublon) :
        * si `filename` existe -> on remplace blobs + on met à jour le doc Mongo
        * sinon -> on insère un nouveau doc
    - Blobs stockés sous un nom STABLE dérivé du filename (overwrite=True).
    """
    conn_str = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(status_code=500, detail="AZURE_BLOB_CONNECTION_STRING manquant.")

    OFFERS_CONTAINER = os.getenv("JOB_OFFERS_CONTAINER", "job-offers-pdfs")
    OFFERS_EMB_CONTAINER = os.getenv("JOB_OFFERS_EMB_CONTAINER", "job-offers-embeddings")

    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="file_bytes vide.")
    filename = (filename or "job-offer.txt").strip()
    if "." not in filename:
        filename += ".txt"

    if job_vec is None:
        raise HTTPException(status_code=400, detail="job_vec is None.")
    vec = np.asarray(job_vec, dtype=np.float32)
    if not np.isfinite(vec).all() or vec.ndim != 1 or vec.size == 0:
        raise HTTPException(status_code=400, detail="Embedding invalide.")

    content_hash = _sha256_text(emb_text)

    existing = JOBS.find_one({"filename": filename})
    already_exists = existing is not None

    safe_filename = _safe_key(filename)           
    file_blob_name = f"{safe_filename}"           
    emb_blob_name = f"{safe_filename}/full.npy"  

    async with BlobServiceClient.from_connection_string(conn_str) as bsc:
        await _ensure_container(bsc, OFFERS_CONTAINER)
        await _ensure_container(bsc, OFFERS_EMB_CONTAINER)

        offers_cc = bsc.get_container_client(OFFERS_CONTAINER)
        file_blob = offers_cc.get_blob_client(file_blob_name)
        await file_blob.upload_blob(
            data=file_bytes,
            overwrite=True, 
            content_settings=ContentSettings(content_type=_guess_content_type(filename)),
        )
        file_blob_path = f"{OFFERS_CONTAINER}/{file_blob_name}"

        emb_cc = bsc.get_container_client(OFFERS_EMB_CONTAINER)
        emb_blob = emb_cc.get_blob_client(emb_blob_name)
        vec16 = vec.astype(np.float16, copy=False)
        buf = io.BytesIO()
        np.save(buf, vec16, allow_pickle=False)
        buf.seek(0)
        await emb_blob.upload_blob(
            data=buf.getvalue(),
            overwrite=True,  
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )
        emb_blob_path = f"{OFFERS_EMB_CONTAINER}/{emb_blob_name}"

    summary = (
        (job_info.get("en") or {}).get("short_description", {}).get("summary")
        or (job_info.get("fr") or {}).get("short_description", {}).get("summary")
        or None
    )

    now = _now_iso()
    doc_set = {
        "uploader_email": uploader,
        "updated_at": now,
        "content_hash": content_hash,
        "job_text_preview": (job_text or "")[:2000],
        "job_info": job_info,
        "summary": summary,
        "department": department,
        "candidate_source": candidate_source,
        "file": {
            "container": OFFERS_CONTAINER,
            "blob": file_blob_path,           
            "size": len(file_bytes),
            "content_type": _guess_content_type(filename),
        },
        "embedding": {
            "container": OFFERS_EMB_CONTAINER,
            "blob": emb_blob_path,          
            "dim": int(vec.shape[0]),
            "dtype": "float16",
            "job_key": safe_filename,
        },
    }

    if already_exists:
        JOBS.update_one({"_id": existing["_id"]}, {"$set": doc_set})
        _id = str(existing["_id"])
        return {
            "already_exists": True,
            "replaced": True,
            "_id": _id,
            "content_hash": content_hash,
            "file_blob": file_blob_path,
            "emb_blob": emb_blob_path,
            "job_key": safe_filename,
        }
    else:
        doc = {
            "filename": filename,
            "created_at": now,
            **doc_set,
        }
        res = JOBS.insert_one(doc)
        _id = str(res.inserted_id)
        return {
            "already_exists": False,
            "replaced": False,
            "_id": _id,
            "content_hash": content_hash,
            "file_blob": file_blob_path,
            "emb_blob": emb_blob_path,
            "job_key": safe_filename,
        }

def get_all_offers(email: str | None = None) -> dict:
    """
    Récupère toutes les offres d’emploi depuis la collection 'jobs_offers'.

    - Si `email` est fourni → filtre par uploader_email.
    - Retourne: filename, job_info, summary, uploader_email
    """
    try:
        mongo = MongoDBManager()
        JOBS = mongo.get_collection("jobs_offers")

        query = {}
        if email:
            query = {"uploader_email": email}

        projection = {
            "filename": 1,
            "created_at":1,
            "job_info": 1,
            "summary": 1,
            "uploader_email": 1,
        }

        cursor = JOBS.find(query, projection)
        offers = []
        for doc in cursor:
            offers.append({
                "_id": str(doc.get("_id")),
                "filename": doc.get("filename"),
                "created_at": doc.get("created_at"),
                "job_info": doc.get("job_info"),
                "summary": doc.get("summary"),
                "uploader_email": doc.get("uploader_email"),
            })

        return {"ok": True, "total": len(offers), "items": offers}

    except Exception as e:
        return {"ok": False, "error": str(e), "items": []}
    
async def load_job_offer_assets_by_filename(filename: str) -> Tuple[Dict[str, Any], np.ndarray, str]:
    """
    Retourne (job_info, job_vec_float32, job_text_preview) pour une offre déjà enregistrée.
    Lève HTTPException si introuvable / incomplet.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="filename manquant pour l’offre existante.")

    doc = JOBS.find_one({"filename": filename})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Offre introuvable pour filename={filename!r}.")

    emb_meta = (doc.get("embedding") or {})
    emb_blob_path = emb_meta.get("blob")
    if not emb_blob_path:
        raise HTTPException(status_code=500, detail="Chemin d'embedding manquant pour cette offre.")

    vec = await adownload_blob_to_ndarray(emb_blob_path, dtype=np.float32)
    if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.size == 0:
        raise HTTPException(status_code=500, detail="Embedding corrompu ou invalide.")

    job_vec: np.ndarray = vec  

    job_info: Dict[str, Any] = doc.get("job_info") or {}
    
    job_text_preview: str = doc.get("job_text_preview") or ""
    return job_info, job_vec, job_text_preview

