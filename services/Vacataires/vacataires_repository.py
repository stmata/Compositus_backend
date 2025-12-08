from typing import Optional
from utils.db_service import MongoDBManager
from dotenv import load_dotenv
import os

load_dotenv()

mongo = MongoDBManager()
VAC_COLL = mongo.get_collection("vacataires")
PDFS_CONTAINER = os.getenv("CVS_CONTAINER", "cvs-pdfs")

def _exists_pdf_filename(filename: str) -> Optional[dict]:
    if not filename:
        return None

    doc = VAC_COLL.find_one({"cv.filename": filename})
    if doc:
        return doc

    pdf_blob_path = f"{PDFS_CONTAINER}/{filename}"
    doc = VAC_COLL.find_one({"cv.pdf_blob": pdf_blob_path})
    return doc

def _exists_collab_key(collab_key: str) -> Optional[dict]:
    if not collab_key:
        return None
    return VAC_COLL.find_one({"_id": collab_key})
