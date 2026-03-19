"""OCR module — extract text from cropped label images.

Uses Tesseract OCR on CPU. The Metis AIPU handles YOLO detection,
then crops are sent here for text extraction.

Install: sudo apt-get install tesseract-ocr tesseract-ocr-eng
         pip install pytesseract
"""

import logging

logger = logging.getLogger(__name__)

# Try to import pytesseract, fall back gracefully
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning(
        "pytesseract not installed. OCR disabled. "
        "Install with: sudo apt-get install tesseract-ocr && pip install pytesseract"
    )


def extract_text(image, confidence_threshold=0.0):
    """Extract text from a cropped label image using Tesseract.

    Args:
        image: numpy array (BGR or RGB) of the cropped label region.
        confidence_threshold: minimum word confidence to include (0-100).

    Returns:
        tuple of (text: str, confidence: float 0.0-1.0)
    """
    if not HAS_TESSERACT:
        return "", 0.0

    try:
        import cv2

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Try multiple strategies — pick the one that returns the most text
        configs = [
            ("--psm 6", gray),   # grayscale, uniform block
            ("--psm 3", gray),   # grayscale, auto
            ("--psm 11", gray),  # grayscale, sparse text
        ]

        best_text = ""
        best_conf = 0.0

        for psm_config, img in configs:
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT,
                config=psm_config,
            )

            words = []
            confidences = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > confidence_threshold and word.strip():
                    words.append(word.strip())
                    confidences.append(conf)

            text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            if len(text) > len(best_text):
                best_text = text
                best_conf = avg_conf

        return best_text, best_conf

    except Exception as e:
        logger.error("OCR failed: %s", e)
        return "", 0.0
