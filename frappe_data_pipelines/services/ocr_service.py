"""
OCR Service for text extraction from images and scanned documents.

Supports PaddleOCR for multilingual OCR with excellent Arabic support.
Falls back to simpler methods if PaddleOCR is not available.
"""
import frappe
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    detected_languages: List[str]
    bounding_boxes: Optional[List[dict]] = None


class OCRService:
    """
    OCR service using PaddleOCR for text extraction.

    Supports 109 languages including Arabic and English with
    excellent accuracy on mixed-language documents.
    """

    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'ar': 'Arabic',
        'ch': 'Chinese',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
    }

    def __init__(self, languages: List[str] = None):
        """
        Initialize OCR service.

        Args:
            languages: List of language codes to use (e.g., ['en', 'ar'])
                      If None, will auto-detect or use English + Arabic
        """
        self.languages = languages or ['en', 'ar']
        self._ocr = None

    def _get_ocr(self):
        """Lazy load PaddleOCR to avoid import overhead."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                # Use multilingual model for Arabic/English
                # PaddleOCR will download models on first use
                self._ocr = PaddleOCR(
                    use_angle_cls=True,  # Detect text rotation
                    lang='en',  # Base language, multi-lang handled by model
                    show_log=False,
                    use_gpu=False  # Use CPU for compatibility
                )
            except ImportError:
                frappe.log_error(
                    title="PaddleOCR not installed",
                    message="Install paddleocr and paddlepaddle for OCR support"
                )
                raise ImportError(
                    "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
                )

        return self._ocr

    def extract_text(self, image_path: str) -> OCRResult:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult with extracted text and metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            return self._extract_with_paddleocr(image_path)
        except ImportError:
            # Fall back to basic extraction if PaddleOCR not available
            return self._extract_fallback(image_path)
        except Exception as e:
            frappe.log_error(
                title=f"OCR failed for {image_path}",
                message=str(e)
            )
            return OCRResult(
                text="",
                confidence=0.0,
                detected_languages=[]
            )

    def _extract_with_paddleocr(self, image_path: str) -> OCRResult:
        """Extract text using PaddleOCR."""
        ocr = self._get_ocr()

        # Run OCR
        result = ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return OCRResult(
                text="",
                confidence=0.0,
                detected_languages=[]
            )

        # Extract text and confidence from results
        texts = []
        confidences = []
        bounding_boxes = []

        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_conf = line[1]  # (text, confidence)

                if text_conf and len(text_conf) >= 2:
                    text = text_conf[0]
                    conf = text_conf[1]

                    texts.append(text)
                    confidences.append(conf)
                    bounding_boxes.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': conf
                    })

        # Combine texts with newlines, preserving reading order
        combined_text = '\n'.join(texts)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Detect languages from the text
        detected_langs = self._detect_languages(combined_text)

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            detected_languages=detected_langs,
            bounding_boxes=bounding_boxes
        )

    def _extract_fallback(self, image_path: str) -> OCRResult:
        """
        Fallback OCR using pytesseract if available.
        """
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng+ara')

            return OCRResult(
                text=text.strip(),
                confidence=0.5,  # Unknown confidence for tesseract
                detected_languages=self._detect_languages(text)
            )
        except ImportError:
            frappe.log_error(
                title="No OCR engine available",
                message="Neither PaddleOCR nor pytesseract is installed"
            )
            return OCRResult(
                text="",
                confidence=0.0,
                detected_languages=[]
            )

    def _detect_languages(self, text: str) -> List[str]:
        """
        Detect languages in the extracted text.

        Returns list of ISO language codes.
        """
        if not text:
            return []

        detected = []

        # Simple heuristic detection
        # Check for Arabic characters
        if any('\u0600' <= char <= '\u06FF' for char in text):
            detected.append('ar')

        # Check for English/Latin characters
        if any(char.isascii() and char.isalpha() for char in text):
            detected.append('en')

        # Check for Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            detected.append('ch')

        # Use langdetect for more accurate detection if available
        try:
            from langdetect import detect_langs

            langs = detect_langs(text)
            for lang in langs:
                if lang.prob > 0.3:  # Only include if confidence > 30%
                    code = lang.lang
                    if code not in detected:
                        detected.append(code)
        except Exception:
            pass

        return detected if detected else ['unknown']

    def extract_from_pdf_page(self, pdf_path: str, page_number: int) -> OCRResult:
        """
        Extract text from a specific PDF page using OCR.

        Useful for scanned PDFs that don't have embedded text.

        Args:
            pdf_path: Path to the PDF file
            page_number: 0-indexed page number

        Returns:
            OCRResult with extracted text
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import io

            # Open PDF and get the page
            doc = fitz.open(pdf_path)
            if page_number >= len(doc):
                return OCRResult(text="", confidence=0.0, detected_languages=[])

            page = doc[page_number]

            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Save to temp file for OCR
            temp_path = frappe.get_site_path("private", "temp", f"ocr_page_{page_number}.png")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            img.save(temp_path)

            # Run OCR
            result = self.extract_text(temp_path)

            # Clean up
            try:
                os.remove(temp_path)
            except Exception:
                pass

            doc.close()
            return result

        except ImportError:
            frappe.log_error(
                title="PyMuPDF not installed",
                message="Install pymupdf for PDF OCR support"
            )
            return OCRResult(text="", confidence=0.0, detected_languages=[])
        except Exception as e:
            frappe.log_error(
                title=f"PDF OCR failed for page {page_number}",
                message=str(e)
            )
            return OCRResult(text="", confidence=0.0, detected_languages=[])

    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Detect if a PDF is scanned (image-based) vs text-based.

        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample

        Returns:
            True if PDF appears to be scanned
        """
        try:
            import fitz

            doc = fitz.open(pdf_path)
            total_pages = min(len(doc), sample_pages)

            text_lengths = []
            for i in range(total_pages):
                page = doc[i]
                text = page.get_text()
                text_lengths.append(len(text.strip()))

            doc.close()

            # If average text length is very low, likely scanned
            avg_text = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            return avg_text < 50  # Less than 50 chars average = likely scanned

        except Exception:
            return False

    def test_connection(self) -> dict:
        """Test OCR setup."""
        try:
            ocr = self._get_ocr()
            return {
                "success": True,
                "message": "PaddleOCR is ready for text extraction"
            }
        except ImportError as e:
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"OCR setup failed: {str(e)}"
            }


def get_ocr_service(languages: List[str] = None) -> OCRService:
    """
    Factory function to get OCR service.

    Args:
        languages: Optional list of language codes

    Returns:
        OCRService instance
    """
    return OCRService(languages=languages)
