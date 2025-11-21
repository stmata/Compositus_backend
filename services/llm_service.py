from __future__ import annotations
import os, json, re
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI
from colorama import init as colorama_init
from utils.console import _c

load_dotenv()
colorama_init(autoreset=True)

API_KEY  = os.getenv("API_KEY")
API_VER  = os.getenv("OPENAI_API_VERSION")
API_BASE = (os.getenv("API_BASE") or "").rstrip("/")
AZURE_DEPLOYMENT_SUMMARY = os.getenv("AZURE_DEPLOYMENT_SUMMARY", "gpt-4o-mini")

try:
    client_llm = AzureOpenAI(api_key=API_KEY, api_version=API_VER, azure_endpoint=API_BASE)
except Exception as e:
    client_llm = None

def _llm(prompt: str) -> str:
    if client_llm is None:
        raise RuntimeError("AzureOpenAI client not initialized")
    resp = client_llm.chat.completions.create(
        model=AZURE_DEPLOYMENT_SUMMARY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception as e:
            return {}
