from __future__ import annotations
import asyncio, json, time
from typing import AsyncIterator
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from utils.job_tracker import subscribe

router = APIRouter()

def _sse(event: str, data: dict | str | None = None) -> str:
    if data is None:
        return f"event: {event}\n\n"
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str, request: Request):
    async def event_source() -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue()

        async def pump_tracker():
            async for chunk in subscribe(job_id):  
                await q.put(chunk)

        async def pump_keepalive():
            while True:
                await asyncio.sleep(12)
                await q.put(_sse("keepalive", {"ts": int(time.time())}))

        t1 = asyncio.create_task(pump_tracker())
        t2 = asyncio.create_task(pump_keepalive())
        try:
            while True:
                if await request.is_disconnected():
                    break
                item = await q.get()
                yield item
        finally:
            for t in (t1, t2):
                try: t.cancel()
                except: pass
    origin = request.headers.get("origin")
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=headers)
