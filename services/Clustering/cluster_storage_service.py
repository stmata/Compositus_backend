import os, io, time, unicodedata, re, uuid
import numpy as np
from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from colorama import Fore
from utils.console import _c
import os
from dotenv import load_dotenv


load_dotenv()

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
EMB_CONTAINER = os.getenv("EMB_CONTAINER", "employee-embeddings")

_bsc: Optional[BlobServiceClient] = None


def _get_blob_service() -> BlobServiceClient:
    global _bsc
    if _bsc is None:
        _bsc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    return _bsc

def _get_container():
    bsc = _get_blob_service()
    cc = bsc.get_container_client(EMB_CONTAINER)
    try:
        cc.create_container()
    except ResourceExistsError:
        pass
    return cc

def _blob_upload(container_client, blob_path: str, data: bytes, content_type: Optional[str] = None):
    for attempt in range(5):
        try:
            container_client.upload_blob(name=blob_path, data=data, overwrite=True, content_type=content_type)
            return
        except Exception as e:
            if attempt == 4: raise
            time.sleep(0.3 * (2 ** attempt))

def _save_embedding_blob(source: str, cluster_id: int, collab_key: str, vec: np.ndarray) -> str:
    """
    Sauvegarde l'embedding en float16 .npy sous:
      <container>/<source>/C<cluster_id>/<collab_key>.npy
    Retourne le chemin logique (container/path) que tu pourras stocker dans Mongo.
    """
    arr = np.asarray(vec, dtype=np.float16)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)

    blob_path = f"{source}/C{int(cluster_id)}/{collab_key}.npy"
    cc = _get_container()
    _blob_upload(cc, blob_path, buf.getvalue(), content_type="application/octet-stream")
    return f"{EMB_CONTAINER}/{blob_path}"

def _cleanup_source_clusters(source: str):
    """
    Supprime tous les blobs du préfixe <source>/C* pour repartir clean.
    """
    cc = _get_container()
    prefix = f"{source}/C"
    blobs = cc.list_blobs(name_starts_with=prefix)
    to_delete = [b.name for b in blobs]
    if not to_delete:
        return 0
    deleted = 0
    for name in to_delete:
        try:
            cc.delete_blob(name)
            deleted += 1
        except Exception:
            pass
    return deleted

def _safe_blob_id(name: str) -> str:
    """
    Transforme 'Michée Losomba' -> 'Michee_Losomba'
    pour l'utiliser dans un nom de fichier blob.
    """
    if not name:
        return f"vac_{uuid.uuid4().hex[:8]}"

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    name = re.sub(r"[^\w\-]+", "_", name)

    name = name.strip("_")

    if not name:
        name = f"vac_{uuid.uuid4().hex[:8]}"

    return name

def upsert_single_embedding_and_label(
    source: str,
    collab_key: str,
    vec: np.ndarray,
    label: Optional[int],
) -> str:
    """
    Sauvegarde le .npy pour un seul collaborateur sous:
      <EMB_CONTAINER>/<source>/C<label|m1>/<collab_key>.npy
    Retourne le chemin logique 'container/path'. (Pas de write Mongo ici -> laisse l'appelant décider)
    """
    cid = (label if isinstance(label, int) and label is not None else -1)
    return _save_embedding_blob(source, int(cid), collab_key, np.array(vec, dtype=np.float32))

def upsert_single_embedding_and_label2(
    source: str,
    collab_key: str,
    vec: np.ndarray,
    label: str | int | None,
) -> str:
    """
    Sauvegarde un embedding dans le container d'employés :

        <source>/Cluster<label>/<safe_collab_key>.npy

    - Si label est None → Cluster-1 (non assigné)
    - Écrase l’ancien embedding
    - Retourne blob_path (ex: 'Vacataires/Cluster0/Maroua_Bouain.npy')
    """


    if label is None:
        cluster_dir = "Cluster-1"
    else:
        try:
            cid = int(label)
            cluster_dir = f"Cluster{cid}"
        except (ValueError, TypeError):
            cluster_dir = f"Cluster{str(label)}"


    safe_name = _safe_blob_id(collab_key)
    filename = f"{safe_name}.npy"


    blob_path = f"{source}/{cluster_dir}/{filename}"


    bsc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    container = bsc.get_container_client(EMB_CONTAINER)

    try:
        container.create_container()
    except ResourceExistsError:
        pass

    existing_blobs = container.list_blobs(name_starts_with=f"{source}/")

    for b in existing_blobs:
        if b.name.endswith(f"/{filename}") and b.name != blob_path:
            try:
                container.delete_blob(b.name)
            except Exception as e:
                print(_c(f" Erreur suppression ancien blob {b.name}: {e}", Fore.YELLOW))

    buf = io.BytesIO()
    np.save(buf, vec.astype(np.float32))
    buf.seek(0)
    blob = container.get_blob_client(blob_path)
    blob.upload_blob(buf.getvalue(), overwrite=True)
    return f"{EMB_CONTAINER}/{blob_path}"

def delete_employee_embedding(collab_key: str, source: Optional[str] = None) -> int:
    """
    Supprime les embeddings d'un collaborateur dans le conteneur Azure Blob.

    - collab_key : identifiant du collaborateur (même valeur utilisée pour nommer le .npy)
    - source     : section/source optionnelle (ex: "EAP_Professeurs").
                   Si None → on cherche dans toutes les sources.

    Retourne le nombre de blobs supprimés.
    """
    if not collab_key:
        return 0

    cc = _get_container()
    deleted = 0

    if source:
        prefix = f"{source}/"
        blobs_iter = cc.list_blobs(name_starts_with=prefix)
    else:
        blobs_iter = cc.list_blobs()

    for blob in blobs_iter:
        if blob.name.endswith(f"/{collab_key}.npy"):
            try:
                cc.delete_blob(blob.name)
                deleted += 1
            except Exception as e:
                print(_c(f"{blob.name}: {e}", Fore.RED))

    return deleted

def _move_embedding_blob(source: str, from_label: Optional[int], to_label: int, collab_key: str) -> str:
    cc = _get_container()
    src = f"{source}/C{int(from_label) if isinstance(from_label, int) else 'm1'}/{collab_key}.npy"
    dst = f"{source}/C{int(to_label)}/{collab_key}.npy"
    data = None
    try:
        data = cc.get_blob_client(src).download_blob().readall()
    except Exception:
        pass
    if data is None:
        return f"{EMB_CONTAINER}/{dst}"
    _blob_upload(cc, dst, data, "application/octet-stream")
    try:
        cc.delete_blob(src)
    except Exception:
        pass
    return f"{EMB_CONTAINER}/{dst}"
