"""Image preprocessing utilities for OCR."""
import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV (cv2) not installed. Some image processing features will be limited.")


class ImageHandler:
    """Loads and preprocesses images for OCR, including watermark removal."""

    supported_formats = ['.bmp', '.gif', '.png', '.tiff', '.webp', '.jpg', '.jpeg']

    def load_image(self, file_path: Union[str, Path]) -> Image.Image:
        return Image.open(file_path)

    def load_image_from_bytes(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def get_image_dimensions(self, image: Image.Image) -> tuple:
        return image.size

    def save_image(self, image: Image.Image, file_path: Union[str, Path]) -> None:
        image.save(file_path)

    def auto_rotate_image(self, image: Image.Image) -> Image.Image:
        """Rotate image based on EXIF orientation metadata."""
        try:
            return ImageOps.exif_transpose(image)
        except Exception as e:
            logger.warning(f"Auto-rotate failed: {e}")
            return image

    def remove_noise(self, image: Image.Image) -> Image.Image:
        """Denoise image. Uses OpenCV bilateral filter if available, else PIL median filter."""
        if OPENCV_AVAILABLE:
            arr = np.array(image.convert("RGB"))
            denoised = cv2.bilateralFilter(arr, 9, 75, 75)
            return Image.fromarray(denoised)
        return image.filter(ImageFilter.MedianFilter(size=3))

    def enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(factor)

    def enhance_sharpness(self, image: Image.Image, factor: float = 1.3) -> Image.Image:
        return ImageEnhance.Sharpness(image).enhance(factor)

    def enhance_brightness(self, image: Image.Image, factor: float = 1.3) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(factor)

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        return image.convert("L")

    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        return image.resize((width, height))

    def apply_gaussian_blur(self, image: Image.Image, radius: float = 2.0) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def remove_watermark(self, image: Image.Image) -> Image.Image:
        """Strip light-gray tiled/ghosted watermarks via Otsu binarization.

        Watermarks in scanned knitting-pattern PDFs are consistently much
        lighter (grayscale ~190-240) than actual ink (near-black), so a
        global Otsu threshold cleanly separates them while preserving text
        strokes. No-ops if OpenCV is unavailable.
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Skipping watermark removal.")
            return image
        gray = np.array(image.convert("L"))
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binarized)

    def deskew_image(self, image: Image.Image) -> Image.Image:
        """Correct page skew using Hough line detection. Opt-in; requires OpenCV."""
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Skipping deskew operation.")
            return image
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None or len(lines) == 0:
            return image
        rho, theta = lines[0][0]
        angle = (theta * 180 / np.pi) - 90
        (h, w) = arr.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(arr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(rotated)

    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Full preprocessing pipeline before handing an image to OCR."""
        image = self.auto_rotate_image(image)
        image = self.remove_noise(image)
        image = self.convert_to_grayscale(image)
        image = self.remove_watermark(image)
        return image
