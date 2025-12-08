import json
import pandas as pd
from utils.competency_taxonomy import COMPETENCY_TAXONOMY, COMPETENCY_TAXONOMY_FR
from utils.db_service import MongoDBManager
from dotenv import load_dotenv
from services.embedding_service import _ensure_env_or_die, _build_embed_model, _embed_texts
from services.Clustering.cluster_storage_service import _save_embedding_blob
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from utils.job_tracker import set_stage
from services.candidate_service import _download_blob_to_ndarray, _is_all, _normalize_dept

load_dotenv()
mongo = MongoDBManager()
users = mongo.get_collection("Professeurs")

def standardize_nip(df):
    """Convert NIP to standardized string format"""
    df = df.copy()
    df["NIP"] = df["NIP"].astype(str).str.strip()
    return df

def safe_list_unique(series):
    """Safely create a list of unique non-null values"""
    return list(series.dropna().unique())

def concat_abstracts(series):
    """Concatenate all non-null abstracts with separator"""
    abstracts = series.dropna().tolist()
    return " ||| ".join(abstracts) if abstracts else None

def extract_competencies(text, taxonomy):
    """
    Extract competencies from text based on keyword matching.
    Returns a list of matched competencies.
    """
    if pd.isna(text) or not text:
        return []

    text_lower = text.lower()
    found_competencies = []

    for competency, keywords in taxonomy.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_competencies.append(competency)
                break  

    return found_competencies

def extract_all_competencies(abstract_en, abstract_fr):
    """
    Extract competencies from both English and French abstracts.
    Combines and deduplicates results.
    """
    competencies_en = extract_competencies(abstract_en, COMPETENCY_TAXONOMY)
    competencies_fr = extract_competencies(abstract_fr, COMPETENCY_TAXONOMY_FR)

    all_competencies = list(set(competencies_en + competencies_fr))
    return all_competencies if all_competencies else None

def create_unified_competency_profile(row):
    """
    Create a unified competency profile from multiple sources.
    Sources: Research interests, Teaching interests, Abstract-derived competencies
    """
    all_competencies = []

    if isinstance(row.get("Research_Interests_List"), list):
        all_competencies.extend(row["Research_Interests_List"])

    if isinstance(row.get("Teaching_Interests_List"), list):
        all_competencies.extend(row["Teaching_Interests_List"])

    if isinstance(row.get("Competencies_From_Abstracts"), list):
        all_competencies.extend(row["Competencies_From_Abstracts"])

    seen = set()
    unique_competencies = []
    for comp in all_competencies:
        comp_lower = comp.lower() if isinstance(comp, str) else comp
        if comp_lower not in seen:
            seen.add(comp_lower)
            unique_competencies.append(comp)

    return unique_competencies if unique_competencies else None

def prepare_for_csv(df):
    """Convert list columns to JSON strings for CSV compatibility"""
    df_csv = df.copy()
    list_columns = [
        "Research_Interests",
        "Teaching_Interests",
        "Unified_Competencies",
        "Publication_Types",
        "Publication_Years",
        "Publication_Titles",
        "Journals",
        "Keywords_EN",
        "Keywords_FR",
        "Abstract_Derived_Competencies",
    ]

    for col in list_columns:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )
    return df_csv

def build_professors_consolidated_json(
    df_professors,
    df_publications,
    df_research,
    df_teaching,
):
    df_professors = standardize_nip(df_professors)
    df_publications = standardize_nip(df_publications)
    df_research = standardize_nip(df_research)
    df_teaching = standardize_nip(df_teaching)

    df_research_agg = df_research.groupby("NIP").agg(
        {
            "Name": "first",
            "Firstname": "first",
            "Research Interests": lambda x: list(x.dropna().unique()),
        }
    ).reset_index()
    df_research_agg.rename(
        columns={"Research Interests": "Research_Interests_List"}, inplace=True
    )

    df_teaching_agg = df_teaching.groupby("NIP").agg(
        {
            "Name": "first",
            "Firstname": "first",
            "Teaching Interests": lambda x: list(x.dropna().unique()),
        }
    ).reset_index()
    df_teaching_agg.rename(
        columns={"Teaching Interests": "Teaching_Interests_List"}, inplace=True
    )

    df_publications_agg = df_publications.groupby("NIP").agg(
        {
            "ID publication": "count", 
            "Type of publication": safe_list_unique,
            "Intellectual contributions.Year": lambda x: list(x.dropna().unique()),
            "Title": safe_list_unique,
            "Journal": safe_list_unique,
            "Abstract FR": concat_abstracts,
            "Abstract EN": concat_abstracts,
            "Keywords FR": safe_list_unique,
            "Keyword EN": safe_list_unique,
        }
    ).reset_index()

    df_publications_agg.columns = [
        "NIP",
        "Publication_Count",
        "Publication_Types",
        "Publication_Years",
        "Publication_Titles",
        "Journals",
        "All_Abstracts_FR",
        "All_Abstracts_EN",
        "All_Keywords_FR",
        "All_Keywords_EN",
    ]


    df_publications_agg["Competencies_From_Abstracts"] = df_publications_agg.apply(
        lambda row: extract_all_competencies(
            row["All_Abstracts_EN"], row["All_Abstracts_FR"]
        ),
        axis=1,
    )

    df_merged = df_professors.copy()

    df_merged = df_merged.merge(
        df_research_agg[["NIP", "Research_Interests_List"]],
        on="NIP",
        how="left",
    )

    df_merged = df_merged.merge(
        df_teaching_agg[["NIP", "Teaching_Interests_List"]],
        on="NIP",
        how="left",
    )

    df_merged = df_merged.merge(df_publications_agg, on="NIP", how="left")
    
    df_merged["Unified_Competency_Profile"] = df_merged.apply(
        create_unified_competency_profile, axis=1
    )
   
    column_mapping = {
        "NIP": "NIP",
        "Type of position": "Position_Type",
        "Gender": "Gender",
        "Title": "Academic_Title",
        "Start date in the institution": "Start_Date",
        "FTE": "FTE",
        "Qualification_AACSB": "AACSB_Qualification",
        "Campus": "Campus",
        "Discipline": "Discipline",
        "Faculty.Other category 2_Desc. language 2": "Faculty_Category",
        "Qualification CEFDG": "CEFDG_Qualification",
        "Academy": "Academy",
        "Research Center": "Research_Center",
        "Plan de charge théorique 24-25": "Workload_2024_25",
        "Plan de charge théorique 25-26": "Workload_2025_26",
        "Cours en FR": "Courses_FR",
        "Cours en EN": "Courses_EN",
        "Research_Interests_List": "Research_Interests",
        "Teaching_Interests_List": "Teaching_Interests",
        "Publication_Count": "Publication_Count",
        "Publication_Types": "Publication_Types",
        "Publication_Years": "Publication_Years",
        "Publication_Titles": "Publication_Titles",
        "Journals": "Journals",
        "All_Abstracts_FR": "Abstracts_FR",
        "All_Abstracts_EN": "Abstracts_EN",
        "All_Keywords_FR": "Keywords_FR",
        "All_Keywords_EN": "Keywords_EN",
        "Competencies_From_Abstracts": "Abstract_Derived_Competencies",
        "Unified_Competency_Profile": "Unified_Competencies",
    }

    df_final = df_merged.rename(columns=column_mapping)

    final_columns = [
        "NIP",
        "Academic_Title",
        "Position_Type",
        "Gender",
        "Campus",
        "Discipline",
        "Academy",
        "Research_Center",
        "Start_Date",
        "FTE",
        "AACSB_Qualification",
        "CEFDG_Qualification",
        "Faculty_Category",
        "Workload_2024_25",
        "Workload_2025_26",
        "Courses_FR",
        "Courses_EN",
        "Research_Interests",
        "Teaching_Interests",
        "Unified_Competencies",
        "Publication_Count",
        "Publication_Types",
        "Publication_Years",
        "Publication_Titles",
        "Journals",
        "Abstracts_EN",
        "Abstracts_FR",
        "Keywords_EN",
        "Keywords_FR",
        "Abstract_Derived_Competencies",
    ]

    df_final = df_final[final_columns]
    df_final = df_final.where(pd.notnull(df_final), None)
    records = df_final.to_dict(orient="records")

    users.delete_many({}) 
    if records:
        result = users.insert_many(records)
        inserted = len(result.inserted_ids)
    else:
        inserted = 0

    return {
        "success": True,
        "inserted": inserted,
        "total": len(records),
    }

def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()

def build_prof_embedding_text(doc: Dict[str, Any]) -> str:
    """
    Construit un texte compact pour l’embedding à partir des champs profs.
    Clé principale = NIP, mais ici on ne retourne que le texte.
    """
    parts: List[str] = []

    nip = _norm_str(doc.get("NIP"))
    if nip:
        parts.append(f"NIP: {nip}")

    campus = _norm_str(doc.get("Campus"))
    if campus:
        parts.append(f"Campus: {campus}")

    title = _norm_str(doc.get("Academic_Title"))
    if title:
        parts.append(f"Title: {title}")

    discipline = _norm_str(doc.get("Discipline"))
    if discipline:
        parts.append(f"Discipline: {discipline}")

    academy = _norm_str(doc.get("Academy"))
    if academy:
        parts.append(f"Academy: {academy}")

    position_type = _norm_str(doc.get("Position_Type"))
    if position_type:
        parts.append(f"Position type: {position_type}")

    aacsb = _norm_str(doc.get("AACSB_Qualification"))
    if aacsb:
        parts.append(f"AACSB: {aacsb}")

    cefdq = _norm_str(doc.get("CEFDG_Qualification"))
    if cefdq:
        parts.append(f"CEFDG: {cefdq}")

    def list_to_str(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x)
        if isinstance(v, str):
            return v
        return ""

    teaching = list_to_str(doc.get("Teaching_Interests"))
    if teaching:
        parts.append(f"Teaching interests: {teaching}")

    unified = list_to_str(doc.get("Unified_Competencies"))
    if unified:
        parts.append(f"Competencies: {unified}")

    abstract_comp = list_to_str(doc.get("Abstract_Derived_Competencies"))
    if abstract_comp:
        parts.append(f"Abstract competencies: {abstract_comp}")

    return " | ".join(parts).strip()

def _auto_kmeans_for_profs(vectors: np.ndarray, k_min: int = 2, k_max: int = 20) -> np.ndarray:
    """
    KMeans simple avec choix de k basé sur le silhouette score.
    Très proche de ce que tu faisais 'comme avant'.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    k_max_eff = min(k_max, n - 1)
    best_k = 2
    best_sil = -1.0

    for k in range(k_min, k_max_eff + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(vectors)

        if len(set(labels)) < 2:
            continue

        try:
            sil = silhouette_score(vectors, labels)
        except Exception:
            continue

        if sil > best_sil:
            best_sil = sil
            best_k = k

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    return km.fit_predict(vectors)

def run_prof_clustering_pipeline(job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    1) lit tous les profs dans Mongo (collection 'Professeurs')
    2) construit un texte d'embedding par NIP
    3) calcule les embeddings (Azure OpenAI)
    4) fait du KMeans pour clusteriser
    5) sauvegarde les embeddings dans Azure Blob + cluster_id dans Mongo

    Met à jour la progression 'clustering' dans le job_tracker si job_id est fourni.
    """
    _ensure_env_or_die()

    cursor = users.find(
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
    entries: List[tuple[str, str]] = []  

    for doc in docs:
        nip = _norm_str(doc.get("NIP"))
        if not nip:
            continue
        text = build_prof_embedding_text(doc)
        if not text:
            continue
        entries.append((nip, text))

    if not entries:
        if job_id:
            set_stage(job_id, "clustering", 100, {
                "step": "no_entries",
                "updated": 0,
                "total": 0,
            })
        return {"ok": False, "reason": "no_entries"}

    n = len(entries)

    if job_id:
        set_stage(job_id, "clustering", 20, {
            "step": "embeddings_pre",
            "total": n,
        })

    embed_model = _build_embed_model()
    texts = [t for (_, t) in entries]
    vectors = _embed_texts(embed_model, texts, batch_size=128)

    if vectors.shape[0] != n:
        raise RuntimeError("Embedding size mismatch")

    if job_id:
        set_stage(job_id, "clustering", 40, {
            "step": "embeddings_done",
            "total": n,
        })

    labels = _auto_kmeans_for_profs(vectors, k_min=2, k_max=20)
    n_clusters = len(set(labels)) if labels.size else 0

    if job_id:
        set_stage(job_id, "clustering", 60, {
            "step": "kmeans_done",
            "total": n,
            "clusters": int(n_clusters),
        })

    updated = 0
    log_every = max(5, n // 10) 

    for (nip, _), lbl, vec in zip(entries, labels.tolist(), vectors.tolist()):
        blob_path = _save_embedding_blob(
            "Professeurs",
            int(lbl),
            nip,
            np.array(vec, dtype=np.float32),
        )

        users.update_one(
            {"NIP": nip},
            {
                "$set": {
                    "cluster.semantic_id": int(lbl),
                    "cluster.embedding_blob": blob_path,
                }
            },
            upsert=False,
        )
        updated += 1

        if job_id and (updated % log_every == 0 or updated == n):
            pct = 60 + int(40 * updated / n)
            set_stage(job_id, "clustering", pct, {
                "step": "saving",
                "updated": updated,
                "total": n,
                "clusters": int(n_clusters),
            })

    if job_id:
        set_stage(job_id, "clustering", 100, {
            "step": "clustering_done",
            "updated": updated,
            "total": n,
            "clusters": int(n_clusters),
        })

    return {
        "ok": True,
        "total_professors": n,
        "total_updated": updated,
        "clusters": int(n_clusters),
    }

def load_professor_vectors(department: str) -> List[Tuple[str, str, str, Optional[np.ndarray], str]]:
    """
    Charge les candidats issus de la collection 'Professeurs'.

    Retourne une liste de tuples :
      (source_key, collab_key, display_name, embedding_vector, profile_text)
    compatible avec le reste de la pipeline (rank, explain_topN, etc.).
    """
    if not _is_all(department) and _normalize_dept("EAP_Professeurs") != _normalize_dept(department):
        return []

    cursor = users.find(
        {},
        {
            "_id": 0,
            "NIP": 1,
            "Campus": 1,
            "Academic_Title": 1,
            "Discipline": 1,
            "Academy": 1,
            "Position_Type": 1,
            "AACSB_Qualification": 1,
            "CEFDG_Qualification": 1,
            "cluster.embedding_blob": 1,
            "summary.llm": 1,   
        },
    )

    out: List[Tuple[str, str, str, Optional[np.ndarray], str]] = []
    src_key = "EAP_Professeurs"  

    for doc in cursor:
        nip = str(doc.get("NIP") or "").strip()
        if not nip:
            continue

        cluster_info = doc.get("cluster") or {}
        blob_path = cluster_info.get("embedding_blob")
        vec: Optional[np.ndarray] = None
        if blob_path:
            vec = _download_blob_to_ndarray(blob_path)
            if vec is not None:
                vec = vec.astype(np.float32)
        profile_text = build_prof_embedding_text(doc)

        out.append((src_key, nip, nip, vec, profile_text))
    return out

def build_prof_profile_text(prof_doc: Dict[str, Any]) -> str:
    """
    Construit un texte de profil lisible pour un professeur.
    On réutilise ton build_prof_embedding_text pour avoir un bon résumé.
    """
    if not prof_doc:
        return "(no profile data)"

    try:
        return build_prof_embedding_text(prof_doc)
    except Exception:
        return str(prof_doc)
    




from typing import List, Dict, Any

def get_all_professors(department: str = "All") -> List[Dict[str, Any]]:
    """
    Retourne la liste de tous les professeurs depuis la collection 'Professeurs'.

    - Si department = "All" → on retourne tout.
    - Si un 'department' spécifique est fourni et ne correspond pas à EAP_Professeurs,
      on retourne une liste vide (même logique que load_professor_vectors).
    - On enlève le champ _id (ObjectId) pour que ce soit JSON-serializable.
    - On ajoute un champ 'profile_text' basé sur build_prof_embedding_text.
    """
    if not _is_all(department) and _normalize_dept("EAP_Professeurs") != _normalize_dept(department):
        return []

    cursor = users.find(
        {},
        {
            "_id": 0,  
            "NIP": 1,
            "Academic_Title": 1,
            "Position_Type": 1,
            "Gender": 1,
            "Campus": 1,
            "Discipline": 1,
            "Academy": 1,
            "Research_Center": 1,
            "Start_Date": 1,
            "FTE": 1,
            "AACSB_Qualification": 1,
            "CEFDG_Qualification": 1,
            "Faculty_Category": 1,
            "Workload_2024_25": 1,
            "Workload_2025_26": 1,
            "Courses_FR": 1,
            "Courses_EN": 1,
            "Research_Interests": 1,
            "Teaching_Interests": 1,
            "Unified_Competencies": 1,
            "Publication_Count": 1,
            "Publication_Types": 1,
            "Publication_Years": 1,
            "Publication_Titles": 1,
            "Journals": 1,
            "Abstracts_EN": 1,
            "Abstracts_FR": 1,
            "Keywords_EN": 1,
            "Keywords_FR": 1,
            "Abstract_Derived_Competencies": 1,
            "cluster.semantic_id": 1,
            "cluster.embedding_blob": 1,
        },
    )

    profs: List[Dict[str, Any]] = []

    for doc in cursor:
        doc["NIP"] = _norm_str(doc.get("NIP"))
        doc["profile_text"] = build_prof_embedding_text(doc)
        cluster = doc.get("cluster") or {}
        if "cluster.semantic_id" in doc:
            doc["cluster_id"] = doc.pop("cluster.semantic_id")
        elif isinstance(cluster, dict) and "semantic_id" in cluster:
            doc["cluster_id"] = cluster.get("semantic_id")

        profs.append(doc)

    return profs
