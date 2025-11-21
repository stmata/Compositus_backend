from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from models.models import LoginRequest
from services.authentification import verify_user_exists
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest):
    try:
        user = verify_user_exists(data.email)
        return JSONResponse(content={"user": user}, status_code=200)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
  