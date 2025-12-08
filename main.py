from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.authentification import router as authentification_router
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from routes.history import router as history_router
from routes.employees import router as employees_router
from routes.matching import router as matching_router 
from routes.employees import router as employees_read_router 
from routes.jobs_stream import router as jobs_stream_router 
from routes.vacataires import router as vacataires_router 

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(authentification_router)
app.include_router(history_router)
app.include_router(employees_router)
app.include_router(matching_router)
app.include_router(employees_read_router)
app.include_router(jobs_stream_router)
app.include_router(vacataires_router)
