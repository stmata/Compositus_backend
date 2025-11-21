from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio, json
from utils.db_service import MongoDBManager

router = APIRouter()
mongo = MongoDBManager()
JOBS = mongo.get_collection("jobs")

@router.get("/jobs/{job_id}/stream")
async def stream(job_id: str):
    async def gen():
        last_count = -1
        last_state = None
        try:
            doc = JOBS.find_one({"_id": job_id})
            if not doc:
                yield {"event": "final", "data": json.dumps({"state": "gone", "reason": "not_found"})}
                return

            yield {"event": "update", "data": json.dumps({
                "state": doc.get("state"),
                "stages": doc.get("stages", {}),
                "last_event": (doc.get("events") or [None])[-1],
            })}
            last_state = (doc.get("state") or "").lower()
            last_count = len(doc.get("events") or [])

            while True:
                doc = JOBS.find_one({"_id": job_id})
                if not doc:
                    yield {"event": "final", "data": json.dumps({"state": "gone", "reason": "not_found"})}
                    return

                state = (doc.get("state") or "").lower()
                evts = doc.get("events") or []
                if state != last_state or len(evts) != last_count:
                    last_state = state
                    last_count = len(evts)
                    yield {"event": "update", "data": json.dumps({
                        "state": doc.get("state"),
                        "stages": doc.get("stages", {}),
                        "last_event": (evts[-1] if evts else None),
                    })}

                if state in ("done", "error"):
                    yield {"event": "final", "data": json.dumps({"state": state})}
                    return

                yield {"event": "keepalive", "data": "ping"}
                await asyncio.sleep(0.7)
        except asyncio.CancelledError:
            raise

    return EventSourceResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
