from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from utils.db_service import MongoDBManager
from dotenv import load_dotenv

load_dotenv()
mongo = MongoDBManager()
users = mongo.get_collection("Admins")
vacataires = mongo.get_collection("vacataires")

def save_to_user_upload_history(email: str,
                                candidate_source: str,
                                department: str,
                                job_description: Dict[str, Any],
                                matches: List[Dict[str, Any]],
                                *,
                                offer_mode: str,                
                                is_new_offer: bool,             
                               ) -> Optional[str]:
    date_iso = datetime.utcnow().isoformat()+"+00:00"
    entry = {
        "date": date_iso,
        "doc_type": "file",
        "candidate_source": candidate_source,
        "department": department if department else "All",
        "job_description": job_description,
        "matches": matches,
        "offer_mode": offer_mode,           
        "is_new_offer": bool(is_new_offer),  
       
    }
    res = users.update_one({"email": email}, {"$push": {"upload_history": entry}})
    if res.matched_count == 0:
        return None
    return email

def get_user_upload_history(email: str) -> Optional[List[Dict]]:
    user = users.find_one({"email": email}, {"_id": 0, "upload_history": 1})
    return user.get("upload_history") if user else []

def save_user_upload_result(
    email: str,
    job_data: Dict,
    matches: List[Dict],
    doc_type: str,
    candidate_source: str,
    department: Optional[str] = None,
):
    now = datetime.now(timezone.utc).isoformat()
    users.update_one(
        {"email": email},
        {
            "$push": {
                "upload_history": {
                    "date": now,
                    "doc_type": doc_type,
                    "candidate_source": candidate_source.lower(),
                    "department": department or "All",
                    "job_description": job_data,
                    "matches": matches,
                }
            },
            "$set": {"first_use": False},
        },
    )

def delete_user_upload_history(
    email: str,
    date_iso: Optional[str] = None,
    index: Optional[int] = None,
    delete_all: bool = False,
):
    user = users.find_one({"email": email})
    if not user:
        return {"user_found": False, "deleted_count": 0, "remaining_count": 0}

    history = user.get("upload_history", [])
    original_len = len(history)

    if delete_all:
        history = []
        deleted_by = "all"
    elif date_iso:
        history = [h for h in history if h.get("date") != date_iso]
        deleted_by = "date"
    elif index is not None:
        if index < 0 or index >= len(history):
            raise ValueError("index out of range")
        history.pop(index)
        deleted_by = "index"
    else:
        raise ValueError("No deletion criteria provided.")

    deleted_count = original_len - len(history)

    users.update_one({"_id": user["_id"]}, {"$set": {"upload_history": history}})

    return {
        "user_found": True,
        "deleted_count": deleted_count,
        "remaining_count": len(history),
        "deleted_by": deleted_by,
    }

def list_vacataires_min() -> List[Dict]:
    cursor = vacataires.find({}, {"_id": 1, "cv.filename": 1})
    return [
        {"id": str(doc["_id"]), "filename": doc.get("cv", {}).get("filename")}
        for doc in cursor
    ]
