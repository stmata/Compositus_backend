from fastapi import HTTPException
from utils.db_service import MongoDBManager 
from dotenv import load_dotenv

load_dotenv()

mongo = MongoDBManager()
users = mongo.get_collection("Admins")

def _normalize_email(email: str) -> str:
    """Nettoie et normalise l'email, lève une 400 si vide."""
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email manquant.")
    return email

def _get_user_by_email(email: str) -> dict | None:
    """Renvoie le document user brut depuis Mongo (ou None)."""
    return users.find_one({"email": email})

def verify_user_exists(email: str) -> dict:
    email_norm = _normalize_email(email)
    user = _get_user_by_email(email_norm)

    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")

    user.pop("_id", None)
    user.setdefault("access_scope", "none")

    return user