import logging
import httpx
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)

logger = logging.getLogger("llm")

def get_client():
    """Create and return an Azure OpenAI client with a 60-second timeout."""
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "[llm] Azure OpenAI not configured. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env"
        )

    # 180s timeout to allow long generation
    timeout = httpx.Timeout(180.0, connect=15.0)

    logger.info("Connecting to: %s", AZURE_OPENAI_ENDPOINT)
    logger.info("API version: %s", AZURE_OPENAI_API_VERSION)
    logger.info("Deployment: %s", AZURE_OPENAI_DEPLOYMENT_NAME)

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        timeout=timeout,
    )
