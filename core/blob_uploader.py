"""
blob_uploader.py — Handles uploading local files and raw text to Azure Blob Storage.
"""

import os
import logging
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME

logger = logging.getLogger("blob")

def get_blob_service_client():
    if not AZURE_STORAGE_CONNECTION_STRING:
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception as e:
        logger.error("Failed to create BlobServiceClient: %s", e)
        return None

def ensure_container_exists(blob_service_client):
    try:
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        if not container_client.exists():
            logger.info("Container '%s' does not exist. Creating it...", AZURE_CONTAINER_NAME)
            container_client.create_container()
        return container_client
    except Exception as e:
        logger.error("Failed to verify/create container: %s", e)
        return None

def upload_file_to_blob(local_file_path, blob_folder="raw_files"):
    """
    Uploads a local file to Azure Blob Storage under the specified virtual folder.
    """
    client = get_blob_service_client()
    if not client:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set. Skipping upload of %s.", local_file_path)
        return False
        
    container_client = ensure_container_exists(client)
    if not container_client:
        return False

    filename = os.path.basename(local_file_path)
    blob_name = f"{blob_folder}/{filename}"

    try:
        blob_client = container_client.get_blob_client(blob_name)
        logger.info("Uploading %s to Azure Blob '%s'...", filename, blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info("Successfully uploaded '%s'.", blob_name)
        return True
    except Exception as e:
        logger.error("ERROR uploading %s: %s", filename, e)
        return False

def upload_text_to_blob(text_content, original_filename, blob_folder="raw_txt_files"):
    """
    Uploads a raw text string to Azure Blob Storage under the specified virtual folder.
    The filename is derived from the original PDF filename, replacing the extension with .txt.
    """
    client = get_blob_service_client()
    if not client:
        return False
        
    container_client = ensure_container_exists(client)
    if not container_client:
        return False

    # Change .pdf to .txt
    base_name, _ = os.path.splitext(original_filename)
    filename = f"{base_name}.txt"
    blob_name = f"{blob_folder}/{filename}"

    try:
        blob_client = container_client.get_blob_client(blob_name)
        logger.info("Uploading extracted text to Azure Blob '%s'...", blob_name)
        blob_client.upload_blob(text_content, overwrite=True)
        logger.info("Successfully uploaded '%s'.", blob_name)
        return True
    except Exception as e:
        logger.error("ERROR uploading text for %s: %s", original_filename, e)
        return False
