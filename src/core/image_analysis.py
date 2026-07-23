"""Image quality analysis and OCR text extraction."""
import logging
from typing import Dict, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR functionality will be limited.")

try:
    from langdetect import detect as langdetect_detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Cannot detect language.")

DEFAULT_OCR_CONFIG = "--psm 6"


class ImageAnalyzer:
    """OCR extraction and image quality metrics for preprocessed page images."""

    def extract_text_with_ocr(self, image: Image.Image, lang: str = "eng") -> str:
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            return pytesseract.image_to_string(image, lang=lang, config=DEFAULT_OCR_CONFIG)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_text_boxes(self, image: Image.Image, lang: str = "eng") -> List[Dict]:
        if not TESSERACT_AVAILABLE:
            return []
        try:
            data = pytesseract.image_to_data(
                image, lang=lang, config=DEFAULT_OCR_CONFIG, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"OCR text-box extraction failed: {e}")
            return []

        boxes = []
        for i, text in enumerate(data.get("text", [])):
            if not text.strip():
                continue
            boxes.append({
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "confidence": float(data["conf"][i]),
            })
        return boxes

    def analyze_image_quality(self, image: Image.Image) -> Dict:
        arr = np.array(image.convert("L"))
        if OPENCV_AVAILABLE:
            blur_variance = cv2.Laplacian(arr, cv2.CV_64F).var()
        else:
            blur_variance = float(np.var(arr))
        brightness = float(arr.mean())
        contrast = float(arr.std())
        return {
            "blur_variance": float(blur_variance),
            "brightness": brightness,
            "contrast": contrast,
            "is_blurry": blur_variance < 100,
            "is_dark": brightness < 80,
            "is_low_contrast": contrast < 30,
        }

    def detect_language(self, text: str) -> str:
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "unknown"
        try:
            return langdetect_detect(text)
        except Exception:
            return "unknown"

    def is_image_based_pdf_page(self, image: Image.Image, quality_metrics: Dict) -> bool:
        width, height = image.size
        aspect_ratio = width / height if height else 0
        return 0.5 < aspect_ratio < 2.0 and not quality_metrics.get("is_blurry", False)

    def extract_structured_content(self, image: Image.Image, lang: str = "eng") -> Dict:
        text = self.extract_text_with_ocr(image, lang=lang)
        text_boxes = self.extract_text_boxes(image, lang=lang)
        quality = self.analyze_image_quality(image)
        language = self.detect_language(text)
        return {
            "text": text,
            "text_boxes": text_boxes,
            "quality_metrics": quality,
            "language": language,
        }
