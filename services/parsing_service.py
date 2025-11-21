from __future__ import annotations
import os, mimetypes, asyncio, random, tempfile
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from colorama import Fore
from utils.console import _c
from llama_parse import LlamaParse

load_dotenv()

LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
PARSE_MAX_PARALLEL = int(os.getenv("PARSE_MAX_PARALLEL", "5"))
PARSE_MAX_RETRIES  = int(os.getenv("PARSE_MAX_RETRIES", "2"))
PARSE_BASE_DELAY   = float(os.getenv("PARSE_BASE_DELAY", "1.5"))

_PARSE_SEM = asyncio.BoundedSemaphore(PARSE_MAX_PARALLEL)

def _build_llama_client() -> LlamaParse:
    if not LLAMA_PARSE_API_KEY:
        raise RuntimeError("LLAMA_PARSE_API_KEY manquant dans l'environnement.")
    return LlamaParse(
        api_key=LLAMA_PARSE_API_KEY,
        result_type="markdown",
        verbose=False,
        num_workers=2,
    )

def _detect_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename or "")
    return mime or "application/octet-stream"

def _extract_text_from_docs(documents: List) -> str:
    parts = []
    for d in documents or []:
        t = getattr(d, "text", None)
        if t:
            parts.append(t.strip())
    return ("\n\n".join(parts)).strip()

def parse_one_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Version synchrone: écrit sur disque puis passe un chemin (str) à LlamaParse.
    """
    if "." not in (filename or ""):
        filename = (filename or "document") + ".pdf"

    mime = _detect_mime(filename)
    suffix = os.path.splitext(filename)[1] or ".pdf"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)

        client = _build_llama_client()
        documents = client.load_data(tmp_path)

        text = _extract_text_from_docs(documents)
        ok = bool(text)
        return {
            "filename": filename,
            "mime": mime,
            "ok": ok,
            "text": text if ok else "",
            "error": "" if ok else "llamaparse_empty_text",
        }
    except Exception as e:
        return {
            "filename": filename,
            "mime": mime,
            "ok": False,
            "text": "",
            "error": f"llamaparse_exception: {e}",
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

class _parse_slot:
    def __init__(self): self.sem = _PARSE_SEM
    async def __aenter__(self): await self.sem.acquire()
    async def __aexit__(self, exc_type, exc, tb): self.sem.release()

def _log_retry(filename: str, attempt: int, delay: float, err: str):
    print(_c(f"retry {attempt} in {delay:.2f}s :: {filename} :: {err}", Fore.YELLOW))

async def parse_one_file_async(
    file_bytes: bytes,
    filename: str,
    *,
    retries: int = PARSE_MAX_RETRIES,
    base_delay: float = PARSE_BASE_DELAY,
) -> Dict[str, Any]:
    attempt = 0
    last_err: Optional[str] = None
    while attempt <= retries:
        try:
            async with _parse_slot():
                return await asyncio.to_thread(parse_one_file, file_bytes, filename)
        except Exception as e:
            attempt += 1
            last_err = str(e)
            if attempt > retries:
                break
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.6)
            _log_retry(filename, attempt, delay, last_err)
            await asyncio.sleep(delay)

    return {
        "filename": filename,
        "mime": _detect_mime(filename),
        "ok": False,
        "text": "",
        "error": f"llamaparse_exception_after_retries: {last_err}",
    }

async def parse_many_parallel(
    files: List[Tuple[bytes, str]],
    *,
    max_parallel: int = PARSE_MAX_PARALLEL,
    retries: int = PARSE_MAX_RETRIES,
) -> Dict[str, Any]:
    local_sem = asyncio.BoundedSemaphore(max_parallel)
    results: List[Dict[str, Any]] = [None] * len(files)

    async def _one(idx: int, content: bytes, name: str):
        async with local_sem:
            out = await parse_one_file_async(content, name, retries=retries)
            results[idx] = out

    tasks = [asyncio.create_task(_one(i, content, name or "unnamed")) for i, (content, name) in enumerate(files)]
    await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r and r.get("ok"))
    err = len(results) - ok
    return {"ok": ok, "err": err, "results": results}

async def parse_job_file_with_service_async(file_bytes: bytes, filename: str) -> str:
    """
    Wrapper qui retourne uniquement le texte.
    """
    out: Dict[str, Any] = await parse_one_file_async(file_bytes, filename)
    return out.get("text", "") if out.get("ok") else ""

