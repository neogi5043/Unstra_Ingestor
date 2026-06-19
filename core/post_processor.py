import json
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from config import AZURE_OPENAI_DEPLOYMENT_NAME
from llm.llm_client import get_client
from llm.prompt import CLEANUP_SYSTEM_PROMPT, CLEANUP_USER_PROMPT

logger = logging.getLogger("post_processor")

def clean_extracted_data(data_map: dict, full_text: str) -> dict:
    """
    Sends extracted data to the LLM for OCR cleanup and sentence stitching.
    Args:
        data_map: dict of { field_id: raw_text_value }
        full_text: The full aggregated text of the document for context.
    Returns:
        dict: { field_id: cleaned_text_value }
    """
    if not data_map:
        return {}

    raw_json = json.dumps(data_map, indent=2)
    user_prompt = CLEANUP_USER_PROMPT.format(raw_json=raw_json, full_text=full_text)

    logger.info("Sending %d fields to LLM for cleanup...", len(data_map))
    
    try:
        client = get_client()
        import openai
        messages = [
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _call_api():
            try:
                return client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
            except openai.BadRequestError:
                return client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )

        start_time = time.time()
        response = _call_api()
        logger.info("Cleanup response received in %.1fs", time.time() - start_time)
        
        raw_content = response.choices[0].message.content.strip()
        
        # Parse JSON
        match = __import__("re").search(r"\{.*\}", raw_content, __import__("re").DOTALL)
        if match:
            cleaned_data = json.loads(match.group(0))
            logger.info("Successfully parsed cleaned data with %d keys", len(cleaned_data))
            return cleaned_data
        else:
            cleaned_data = json.loads(raw_content)
            logger.info("Successfully parsed cleaned data with %d keys", len(cleaned_data))
            return cleaned_data

    except Exception as e:
        logger.error("LLM Cleanup failed: %s", e)
        return data_map # Fallback to original data if cleanup fails
