from fastapi import APIRouter, Query, HTTPException
from services.history_service import (
    get_user_upload_history,
    list_vacataires_min,
    delete_user_upload_history
)

router = APIRouter()

@router.get("/history")
def get_user_history(email: str = Query(..., description="Email of the user")):
    history = get_user_upload_history(email)
    vacataires = list_vacataires_min()

    return {
        "upload_history": history,
        "vacataires": vacataires,
    }

@router.delete("/history/entry")
def delete_history(
    email: str = Query(...),
    date: str = Query(...),
):
    result = delete_user_upload_history(
        email=email,
        date_iso=date,
        delete_all=False,
        index=None,
    )

    if not result["user_found"]:
        raise HTTPException(status_code=404, detail="User not found.")

    if result["deleted_count"] == 0:
        raise HTTPException(status_code=404, detail="No history entry found for this date.")

    return {
        "deleted": True,
        "deleted_count": result["deleted_count"],
        "remaining_count": result["remaining_count"],
        "deleted_date": date,
    }
