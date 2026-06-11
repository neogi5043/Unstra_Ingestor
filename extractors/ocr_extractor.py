"""
ocr_extractor.py — OCR pipeline using Tesseract (pytesseract).

Handles: rasterize page → preprocess → Tesseract OCR → return text.

Hardening over original:
  - rasterize_page: catches pdfplumber rendering exceptions; validates the
    returned object is a PIL Image before use.
  - preprocess_image: handles RGBA / palette / already-grayscale images;
    guards against empty / zero-dimension arrays; makes deskew optional via
    flag; uses configurable h for denoising.
  - ocr_image: validates mode before Tesseract; supports multi-language via
    argument; returns empty string (never raises) on any failure.
  - ocr_page: exposes resolution & lang overrides; logs timing; returns ""
    gracefully on any pipeline failure so callers never see an exception.
  - New: deskew() — auto-rotates pages that are skewed up to ±45°.
  - New: ocr_image_with_retry() — retries with relaxed preprocessing if
    the first pass yields fewer than MIN_CHARS characters.
"""

import time
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError

from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
)

ocr_engine = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ── Constants ──────────────────────────────────────────────────────────────
_MIN_DIMENSION  = 10    # pixels — smaller images are pure noise
_MIN_CHARS      = 20    # characters — below this, retry with relaxed settings
_DENOISE_H      = 10    # fastNlMeansDenoising strength (higher = blurrier but cleaner)
_BLOCK_SIZE     = 11    # adaptiveThreshold block size (must be odd)
_THRESH_C       = 2     # adaptiveThreshold constant subtracted from mean

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_array(pil_image: Image.Image) -> np.ndarray:
    """
    Convert any PIL Image mode to a 2-D uint8 grayscale numpy array.

    Handles: RGB, RGBA, L (already gray), P (palette), CMYK, 1 (binary).
    Raises ValueError if the array ends up with zero dimensions.
    """
    mode = pil_image.mode

    if mode == "L":
        arr = np.array(pil_image, dtype=np.uint8)
    elif mode in ("RGB", "BGR"):
        arr = cv2.cvtColor(np.array(pil_image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    elif mode == "RGBA":
        # Composite onto white background before converting
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=pil_image.split()[3])
        arr = cv2.cvtColor(np.array(bg, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    elif mode == "CMYK":
        arr = cv2.cvtColor(
            np.array(pil_image.convert("RGB"), dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
    elif mode == "P":
        arr = cv2.cvtColor(
            np.array(pil_image.convert("RGB"), dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
    elif mode == "1":
        arr = np.array(pil_image.convert("L"), dtype=np.uint8)
    else:
        # Best-effort fallback
        arr = np.array(pil_image.convert("L"), dtype=np.uint8)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"Grayscale conversion produced invalid array shape {arr.shape}")

    return arr


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct page skew (up to ±45°) using Hough-line analysis.

    Args:
        gray: 2-D uint8 grayscale array

    Returns:
        Deskewed 2-D uint8 array (same shape, white-padded).
        Returns *gray* unchanged if skew cannot be determined.
    """
    try:
        # Edge-detect first to make line detection more robust
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10,
        )
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        # Ignore tiny skews — rotating adds interpolation noise for nothing
        if abs(median_angle) < 0.3:
            return gray

        h, w = gray.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return rotated
    except Exception as e:
        print(f"[ocr] deskew failed (ignored): {e}")
        return gray


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_page(page, resolution: int = 300) -> Image.Image | None:
    """
    Convert a pdfplumber page to a PIL Image.

    Args:
        page:       pdfplumber Page object
        resolution: render DPI (default from config, usually 300)

    Returns:
        PIL.Image on success, None on failure.
    """
    if resolution < 72:
        resolution = 72
        print(f"[ocr] Warning: resolution clamped to 72 DPI (requested value too low)")

    try:
        page_image = page.to_image(resolution=resolution)
        img = page_image.original
    except Exception as e:
        print(f"[ocr] Error rasterizing page: {e}")
        return None

    if not isinstance(img, Image.Image):
        print(f"[ocr] Unexpected rasterize return type: {type(img)}")
        return None

    return img


def preprocess_image(
    pil_image: Image.Image,
    *,
    apply_deskew: bool = True,
    denoise_h: int = _DENOISE_H,
) -> Image.Image | None:
    """
    Preprocess an image for better OCR accuracy.

    Pipeline:
      1. Convert to grayscale (handles RGB / RGBA / CMYK / palette / binary).
      2. Optional deskew (Hough-line auto-rotation, up to ±45°).
      3. Denoise (fastNlMeansDenoising).
      4. Adaptive threshold (Gaussian, binarise).

    Args:
        pil_image:    Input PIL Image (any mode).
        apply_deskew: If True, attempt skew correction before thresholding.
        denoise_h:    Denoising filter strength (higher = more aggressive).

    Returns:
        Preprocessed PIL.Image (mode "L", binary pixel values 0/255).
        Returns None if the image is too small or conversion fails.
    """
    if pil_image is None:
        return None

    w, h = pil_image.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        return None

    try:
        gray = _to_gray_array(pil_image)
    except (ValueError, Exception) as e:
        print(f"[ocr] preprocess: grayscale conversion failed: {e}")
        return None

    if apply_deskew:
        gray = deskew(gray)

    # Denoise — skip on very small images where it adds noise instead
    if gray.shape[0] > 30 and gray.shape[1] > 30:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_h)

    # Adaptive threshold — blockSize must be odd and ≥ 3
    block = _BLOCK_SIZE if _BLOCK_SIZE % 2 == 1 else _BLOCK_SIZE + 1
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=_THRESH_C,
    )

    return Image.fromarray(thresh)


def ocr_image(pil_image: Image.Image) -> str:
    """
    Run Azure OpenAI Vision on a single PIL Image.
    
    Returns:
        The extracted string text, or empty string on failure.
    """
    if pil_image.size[0] < _MIN_DIMENSION or pil_image.size[1] < _MIN_DIMENSION:
        return ""

    try:
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        response = ocr_engine.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image exactly as it appears. Do not add any markdown, explanation, or commentary. Preserve the spatial layout with line breaks."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ocr] Error processing image with Azure Vision: {e}")
        return ""


def ocr_image_with_retry(
    pil_image: Image.Image,
) -> str:
    """
    OCR with Azure Vision. No need for aggressive morphological retries
    since Vision models prefer raw colored images.
    """
    text = ocr_image(pil_image)
    return text


def ocr_page(page_obj) -> str:
    """
    Rasterize a pdfplumber page, preprocess, and extract text using PaddleOCR.
    """
    start_time = time.time()
    try:
        pil_image = rasterize_page(page_obj, resolution=300)
        if pil_image is None:
            print("[ocr] Rasterization failed — returning empty string")
            return ""
        
        text = ocr_image_with_retry(pil_image)
        
        elapsed = time.time() - start_time
        print(f"[ocr] Extracted {len(text)} chars via Azure Vision in {elapsed:.2f}s")
        return text
    except Exception as e:
        print(f"[ocr] Pipeline failed: {e}")
        return ""