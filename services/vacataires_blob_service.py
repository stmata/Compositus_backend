from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient
import os
from utils.db_service import MongoDBManager
from dotenv import load_dotenv
from typing import Optional
from colorama import Fore, Style

load_dotenv()
mongo = MongoDBManager()

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")

PDFS_CONTAINER = os.getenv("CVS_CONTAINER", "cvs-pdfs")

def _guess_content_type(filename: str) -> str:
    import mimetypes
    ctype, _ = mimetypes.guess_type(filename or "")
    return ctype or "application/octet-stream"

async def _ensure_container(bsc: BlobServiceClient, container_name: str) -> None:
    cc = bsc.get_container_client(container_name)
    try:
        await cc.create_container()
    except ResourceExistsError:
        pass

async def _azure_upload_pdf_bytes(file_bytes: bytes, filename: str) -> str:
    safe_name = os.path.basename(filename or "cv.pdf")

    async with BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING) as bsc:
        await _ensure_container(bsc, PDFS_CONTAINER)
        container = bsc.get_container_client(PDFS_CONTAINER)

        blob_name = safe_name
        blob = container.get_blob_client(blob_name)

        await blob.upload_blob(
            data=file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=_guess_content_type(filename)),
        )
        return blob_name

async def _azure_delete_pdf(blob_name: str) -> None:
    async with BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING) as bsc:
        cc = bsc.get_container_client(PDFS_CONTAINER)
        try:
            await cc.delete_blob(blob_name, delete_snapshots="include")
        except Exception:
            pass

async def _azure_delete_by_path(path: Optional[str]) -> None:
    """Supprime un blob à partir d'un chemin 'container/blob'."""
    if not path:
        return

    parts = path.split("/", 1)
    if len(parts) != 2:
        return

    container_name, blob_name = parts

    try:
        async with BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING) as bsc:
            cc = bsc.get_container_client(container_name)
            await cc.delete_blob(blob_name, delete_snapshots="include")
    except Exception as e:
        print(f"{Fore.YELLOW}{Style.RESET_ALL} _azure_delete_by_path({path}) failed: {e}")