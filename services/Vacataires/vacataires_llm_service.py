from openai import AzureOpenAI
import re, os, asyncio, uuid, json
from typing import Optional
from dotenv import load_dotenv
from utils.prompts import build_cv_prompt

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_VER = os.getenv("OPENAI_API_VERSION")
API_BASE = (os.getenv("API_BASE") or "").rstrip("/")
AZURE_DEPLOYMENT_SUMMARY = os.getenv("AZURE_DEPLOYMENT_SUMMARY", "gpt-4o-mini")

try:
    llm_client = AzureOpenAI(api_key=API_KEY, api_version=API_VER, azure_endpoint=API_BASE)
except Exception as e:
    llm_client = None

def _extract_json_loose(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _llm_cv_extract_sync(cv_text: str) -> Optional[dict]:
    if llm_client is None:
        return None
    prompt = build_cv_prompt(cv_text)
    resp = llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_SUMMARY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    out = (resp.choices[0].message.content or "").strip()
    return _extract_json_loose(out)

async def _llm_cv_extract(cv_text: str) -> Optional[dict]:
    return await asyncio.to_thread(_llm_cv_extract_sync, cv_text)

def _pick_collab_key(parsed: Optional[dict], filename: str) -> str:
    def safe_person_key(name: str) -> str:
        return (name or "inconnu").replace(".", "·").replace("$", "￩").strip()
    full = (parsed or {}).get("identity", {}).get("full_name") if isinstance(parsed, dict) else None
    if full and str(full).strip():
        return safe_person_key(str(full))
    root = os.path.splitext(os.path.basename(filename or "cv"))[0]
    return safe_person_key(root or f"vac_{uuid.uuid4().hex[:6]}")
