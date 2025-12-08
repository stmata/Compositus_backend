from __future__ import annotations
from services.llm_service import API_KEY, API_VER, API_BASE
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS", "text-embedding-3-large")
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS"))
TARGET_CHARS = int(os.getenv("TARGET_CHARS"))

def _semantic_chunks(text: str,
                     max_chunks: int = MAX_CHUNKS,
                     target_chars: int = TARGET_CHARS) -> list[str]:
    embed_model = AzureOpenAIEmbedding(
        model=AZURE_EMBED_DEPLOYMENT,
        api_key=API_KEY,
        azure_endpoint=API_BASE,
        api_version=API_VER,
    )
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95.0,
        buffer_size=1,
    )
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    node_texts = [getattr(n, "text", "") for n in nodes if getattr(n, "text", "").strip()]

    chunks: list[str] = []
    cur = ""
    for t in node_texts:
        cand = f"{cur}\n\n{t}".strip() if cur else t
        if len(cand) <= target_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = t
    if cur:
        chunks.append(cur)

    while len(chunks) > max_chunks and len(chunks) >= 2:
        sizes = [len(c) for c in chunks]
        i = sizes.index(min(sizes))
        j = i + 1 if i + 1 < len(chunks) else i - 1
        if j < 0:
            break
        a, b = min(i, j), max(i, j)
        merged = f"{chunks[a]}\n\n{chunks[b]}".strip()
        keep = [c for k, c in enumerate(chunks) if k not in (a, b)]
        keep.insert(a, merged)
        chunks = keep

    return chunks
