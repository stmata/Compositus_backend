from __future__ import annotations
from typing import List
import numpy as np
from services.llm_service import API_KEY, API_VER, API_BASE
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import io
from azure.storage.blob.aio import BlobServiceClient as AioBlobServiceClient
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS", "text-embedding-3-large")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
API_BASE = (os.getenv("API_BASE") or "").rstrip("/")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS", "text-embedding-3-large")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE"))
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
EMB_CONTAINER = os.getenv("EMB_CONTAINER", "employee-embeddings")

def _embed_texts2(texts: List[str]) -> np.ndarray:
    all_vecs: List[List[float]] = []
    client = AzureOpenAI(api_key=API_KEY, api_version=API_VER, azure_endpoint=API_BASE)
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        for t in batch:
            r = client.embeddings.create(model=AZURE_EMBED_DEPLOYMENT, input=t)
            all_vecs.append(r.data[0].embedding)
    arr = np.array(all_vecs, dtype=np.float32)
    return arr

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _download_blob_to_ndarray(blob_path: str) -> np.ndarray | None:
    """
    Télécharge un blob .npy depuis Azure et le convertit en np.ndarray.
    blob_path peut être de la forme :
        "employee-embeddings/Vacataires/C8/Leïla Mansouri.npy"
        ou "embeddings/xyz.npy"
    """
    try:
        if not blob_path:
            return None

        if not AZURE_BLOB_CONNECTION_STRING:
            return None

        parts = blob_path.split("/", 1)
        if len(parts) != 2:
            return None

        container_name, blob_name = parts

        bsc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
        container = bsc.get_container_client(container_name)
        blob_client = container.get_blob_client(blob_name)

        stream = io.BytesIO()
        download = blob_client.download_blob()
        download.readinto(stream)
        stream.seek(0)

        arr = np.load(stream)
        arr = arr.astype(np.float32)
        return arr

    except Exception as e:
        return None
    
async def adownload_blob_to_ndarray(
    blob_path: str,
    dtype: "np.dtype | str | None" = np.float32,
    allow_pickle: bool = False,
) -> "np.ndarray | None":
    if not blob_path or not AZURE_BLOB_CONNECTION_STRING:
        return None

    parts = blob_path.split("/", 1)
    if len(parts) != 2:
        return None
    container_name, blob_name = parts

    try:
        async with AioBlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING) as bsc:
            container = bsc.get_container_client(container_name)
            blob = container.get_blob_client(blob_name)
            data = await blob.download_blob()
            content = await data.readall()

        stream = io.BytesIO(content)
        try:
            arr = np.load(stream, allow_pickle=allow_pickle)
        except ValueError as e:
            if not allow_pickle and "Object arrays cannot be loaded when allow_pickle=False" in str(e):
                stream.seek(0)
                arr = np.load(stream, allow_pickle=True)
            else:
                raise

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr
    except Exception as e:
        return None

def _ensure_env_or_die():
    miss = []
    for k, v in {
        "API_KEY": API_KEY, "API_BASE": API_BASE,
        "OPENAI_API_VERSION": API_VERSION, "AZURE_DEPLOYMENT_EMBEDDINGS": AZURE_EMBED_DEPLOYMENT,
        "AZURE_BLOB_CONNECTION_STRING": AZURE_BLOB_CONNECTION_STRING, "EMB_CONTAINER": EMB_CONTAINER,
    }.items():
        if not v:
            miss.append(k)
    if miss:
        raise RuntimeError(f"Env manquantes: {miss}")

def _build_embed_model() -> AzureOpenAIEmbedding:
    return AzureOpenAIEmbedding(
        model=AZURE_EMBED_DEPLOYMENT,
        api_key=API_KEY,
        azure_endpoint=API_BASE,
        api_version=API_VERSION,
    )

def _embed_texts(embed: AzureOpenAIEmbedding, texts: List[str], batch_size: int = 128) -> np.ndarray:
    clean_texts = [t if isinstance(t, str) else "" for t in texts]
    if not any(t.strip() for t in clean_texts):
        return np.zeros((len(clean_texts), 1), dtype=np.float32)
    vecs: List[List[float]] = []
    for i in range(0, len(clean_texts), batch_size):
        chunk = clean_texts[i : i + batch_size]
        vecs.extend(embed.get_text_embedding_batch(chunk))
    return np.array(vecs, dtype=np.float32)
