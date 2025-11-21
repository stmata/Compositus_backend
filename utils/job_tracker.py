from __future__ import annotations
import asyncio, json
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Set, Optional
from uuid import uuid4
from utils.db_service import MongoDBManager

mongo = MongoDBManager()
JOBS = mongo.get_collection("jobs")

_SUBS: Dict[str, Set[asyncio.Queue[str]]] = {}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _snapshot(job_id: str) -> dict:
    doc = JOBS.find_one({"_id": job_id}) or {}
    events = doc.get("events") or []
    last_event = events[-1] if events else {}
    return {
        "state": doc.get("state", "running"),
        "stages": doc.get("stages", {}),
        "last_event": last_event,
        "error": doc.get("error"),  
        "seq": int(doc.get("seq", 0)),
    }

async def _broadcast(job_id: str) -> None:
    subs = _SUBS.get(job_id)
    if not subs:
        return
    payload = _snapshot(job_id)
    chunk = f"event: update\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    for q in list(subs):
        try:
            q.put_nowait(chunk)
        except Exception:
            pass

def _fire_and_forget(coro):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)

def subscribe(job_id: str) -> AsyncIterator[str]:
    q: asyncio.Queue[str] = asyncio.Queue()
    _SUBS.setdefault(job_id, set()).add(q)

    async def _gen():
        try:
            await _broadcast(job_id)  
            while True:
                chunk = await q.get()
                yield chunk
        finally:
            _SUBS.get(job_id, set()).discard(q)
            if not _SUBS.get(job_id):
                _SUBS.pop(job_id, None)

    return _gen()

def new_job() -> str:
    job_id = uuid4().hex
    JOBS.insert_one({
        "_id": job_id,
        "state": "running",
        "seq": 0,                      
        "stages": {},
        "events": [{"ts": _now_iso(), "type": "created"}],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    })
    _fire_and_forget(_broadcast(job_id))
    return job_id

def _inc_seq_and_push(job_id: str, update: Optional[dict] = None, event: Optional[dict] = None) -> None:
    """Fait un $inc seq + $set updated_at (+$set custom) et $push events (si fourni), atomiquement."""
    now = _now_iso()
    set_fields = {"updated_at": now}
    if update:
        set_fields.update(update)

    ops = {
        "$set": set_fields,
        "$inc": {"seq": 1},
    }
    if event is not None:
        ops["$push"] = {"events": event}

    JOBS.update_one({"_id": job_id}, ops, upsert=True)
    _fire_and_forget(_broadcast(job_id))

def push_event(job_id: str, event: dict):
    ev = {"ts": _now_iso(), **(event or {})}
    _inc_seq_and_push(job_id, update={}, event=ev)

def set_stage(job_id: str, name: str, percent: int, meta: dict | None = None):
    st = {"percent": int(percent), "meta": (meta or {})}
    _inc_seq_and_push(job_id, update={f"stages.{name}": st}, event=None)

def set_done(job_id: str, delete_after: bool = False, drop_mode: str = "none"):
    _inc_seq_and_push(job_id, update={"state": "done"}, event=None)

def set_error(job_id: str, msg: str):
    _inc_seq_and_push(job_id, update={"state": "error", "error": str(msg)}, event=None)

def history_for_job(job_id: str) -> dict:
    return _snapshot(job_id)
