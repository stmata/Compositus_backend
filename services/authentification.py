from fastapi import HTTPException
from utils.db_service import MongoDBManager 
import os
from dotenv import load_dotenv

load_dotenv()

mongo = MongoDBManager()
users = mongo.get_collection("users")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))

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
    """
    Vérifie si l'utilisateur existe dans la DB.
    - Normalise l'email
    - Cherche dans Mongo
    - Lève 404 si absent
    - Retourne l'objet user (sans _id) si trouvé
    """
    email_norm = _normalize_email(email)
    user = _get_user_by_email(email_norm)

    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")

    user.pop("_id", None)
    return user
