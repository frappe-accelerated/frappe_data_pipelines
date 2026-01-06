"""
Decision Agent for intelligent document processing.

Analyzes files to determine the optimal processing strategy
(text, visual, structured, hybrid) based on content type.
"""
import frappe
import os
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingPlan:
    """Plan for how to process a document."""
    strategy: str  # "text", "visual", "structured", "hybrid"
    requires_ocr: bool
    requires_vision: bool
    detected_languages: List[str]
    document_type: str  # "article", "presentation", "spreadsheet", "form", "general"
    has_images: bool
    has_tables: bool
    suggested_chunk_size: int
    visual_elements: List[dict]  # [{"page": 1, "type": "chart", "bbox": [...]}]
    notes: str


class DecisionAgent:
    """
    Intelligent agent for document analysis and processing decisions.

    Analyzes file metadata and content to determine the optimal
    processing path for embedding generation.
    """

    # MIME type to document type mapping
    MIME_TYPE_MAP = {
        # Text documents
        'text/plain': ('text', 'article'),
        'text/markdown': ('text', 'article'),
        'text/html': ('text', 'article'),

        # Office documents
        'application/pdf': ('hybrid', 'general'),  # PDFs need analysis
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ('text', 'article'),
        'application/msword': ('text', 'article'),
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': ('structured', 'presentation'),
        'application/vnd.ms-powerpoint': ('structured', 'presentation'),
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ('structured', 'spreadsheet'),
        'application/vnd.ms-excel': ('structured', 'spreadsheet'),

        # Images
        'image/png': ('visual', 'image'),
        'image/jpeg': ('visual', 'image'),
        'image/gif': ('visual', 'image'),
        'image/webp': ('visual', 'image'),
        'image/tiff': ('visual', 'image'),
        'image/bmp': ('visual', 'image'),
    }

    # Extension fallbacks
    EXTENSION_MAP = {
        '.txt': ('text', 'article'),
        '.md': ('text', 'article'),
        '.html': ('text', 'article'),
        '.htm': ('text', 'article'),
        '.pdf': ('hybrid', 'general'),
        '.docx': ('text', 'article'),
        '.doc': ('text', 'article'),
        '.pptx': ('structured', 'presentation'),
        '.ppt': ('structured', 'presentation'),
        '.xlsx': ('structured', 'spreadsheet'),
        '.xls': ('structured', 'spreadsheet'),
        '.png': ('visual', 'image'),
        '.jpg': ('visual', 'image'),
        '.jpeg': ('visual', 'image'),
        '.gif': ('visual', 'image'),
        '.webp': ('visual', 'image'),
        '.tiff': ('visual', 'image'),
        '.tif': ('visual', 'image'),
        '.bmp': ('visual', 'image'),
    }

    def __init__(self):
        try:
            settings = frappe.get_single("Data Pipeline Settings")
            self.default_chunk_size = settings.chunk_size or 1000
        except Exception:
            self.default_chunk_size = 1000

    def analyze(
        self,
        file_path: str,
        mime_type: str = None,
        file_size: int = None
    ) -> ProcessingPlan:
        """
        Analyze a file and create a processing plan.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type (will be detected if not provided)
            file_size: Optional file size in bytes

        Returns:
            ProcessingPlan with recommended processing strategy
        """
        # Detect MIME type if not provided
        if not mime_type:
            mime_type = self._detect_mime_type(file_path)

        # Get initial strategy from MIME type
        strategy, doc_type = self._get_initial_strategy(file_path, mime_type)

        # Initialize plan
        plan = ProcessingPlan(
            strategy=strategy,
            requires_ocr=False,
            requires_vision=False,
            detected_languages=[],
            document_type=doc_type,
            has_images=False,
            has_tables=False,
            suggested_chunk_size=self.default_chunk_size,
            visual_elements=[],
            notes=""
        )

        # Refine based on content analysis
        if strategy == 'visual':
            plan.requires_vision = True
            plan.requires_ocr = True
            plan.has_images = True
        elif strategy == 'hybrid':
            # Need deeper analysis for PDFs
            plan = self._analyze_pdf(file_path, plan)
        elif strategy == 'structured':
            plan = self._analyze_structured(file_path, doc_type, plan)

        # Detect languages from file name and any extracted text
        plan.detected_languages = self._detect_languages_from_path(file_path)

        # Adjust chunk size based on document type
        plan.suggested_chunk_size = self._suggest_chunk_size(doc_type, plan)

        return plan

    def _get_initial_strategy(self, file_path: str, mime_type: str) -> tuple:
        """Get initial strategy from MIME type or extension."""
        # Try MIME type first
        if mime_type and mime_type in self.MIME_TYPE_MAP:
            return self.MIME_TYPE_MAP[mime_type]

        # Fall back to extension
        ext = Path(file_path).suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]

        # Default to text processing
        return ('text', 'general')

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type of a file."""
        try:
            import magic
            return magic.from_file(file_path, mime=True)
        except ImportError:
            # Fall back to extension-based detection
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type or 'application/octet-stream'
        except Exception:
            return 'application/octet-stream'

    def _analyze_pdf(self, file_path: str, plan: ProcessingPlan) -> ProcessingPlan:
        """
        Analyze PDF content to determine processing needs.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            total_pages = len(doc)

            # Sample pages for analysis
            sample_size = min(5, total_pages)
            text_lengths = []
            image_counts = []
            has_any_images = False

            for i in range(sample_size):
                page = doc[i]

                # Check text content
                text = page.get_text()
                text_lengths.append(len(text.strip()))

                # Check for images
                images = page.get_images()
                image_counts.append(len(images))
                if images:
                    has_any_images = True

            doc.close()

            avg_text = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            avg_images = sum(image_counts) / len(image_counts) if image_counts else 0

            # Determine if scanned (low text, high images)
            if avg_text < 100 and has_any_images:
                plan.strategy = 'visual'
                plan.requires_ocr = True
                plan.requires_vision = True
                plan.notes = "Appears to be a scanned document"
            elif has_any_images and avg_images > 1:
                plan.strategy = 'hybrid'
                plan.has_images = True
                plan.requires_vision = True
                plan.notes = "Mixed text and images"
            else:
                plan.strategy = 'text'
                plan.notes = "Text-based PDF"

            # Check for tables (basic heuristic - lots of structured content)
            if avg_text > 500:
                # Could have tables
                plan.has_tables = True

            plan.has_images = has_any_images

        except ImportError:
            plan.notes = "PyMuPDF not installed, assuming text PDF"
        except Exception as e:
            plan.notes = f"PDF analysis failed: {str(e)}"

        return plan

    def _analyze_structured(
        self,
        file_path: str,
        doc_type: str,
        plan: ProcessingPlan
    ) -> ProcessingPlan:
        """
        Analyze structured documents (presentations, spreadsheets).
        """
        if doc_type == 'presentation':
            plan.notes = "Presentation - will process slide by slide"
            plan.suggested_chunk_size = 500  # Smaller chunks for slides
            plan.requires_vision = True  # Often have charts/diagrams
            plan.has_images = True
        elif doc_type == 'spreadsheet':
            plan.notes = "Spreadsheet - will process row groups with headers"
            plan.has_tables = True
            plan.suggested_chunk_size = 800

        return plan

    def _detect_languages_from_path(self, file_path: str) -> List[str]:
        """
        Detect possible languages from file path/name.

        This is a heuristic - actual language detection happens during processing.
        """
        name = Path(file_path).stem.lower()

        # Check for Arabic characters in filename
        if any('\u0600' <= char <= '\u06FF' for char in name):
            return ['ar', 'en']

        # Default to English
        return ['en']

    def _suggest_chunk_size(self, doc_type: str, plan: ProcessingPlan) -> int:
        """Suggest optimal chunk size based on document type."""
        if doc_type == 'presentation':
            return 500  # Smaller for slides
        elif doc_type == 'spreadsheet':
            return 800  # Row groups
        elif plan.has_images and plan.requires_vision:
            return 1200  # Larger for visual descriptions
        elif doc_type == 'article':
            return 1000  # Standard for text
        else:
            return self.default_chunk_size

    def should_use_smart_pipeline(self, plan: ProcessingPlan) -> bool:
        """
        Determine if smart pipeline features should be used.
        """
        return (
            plan.requires_vision or
            plan.requires_ocr or
            plan.strategy in ('visual', 'hybrid', 'structured') or
            plan.has_images or
            plan.has_tables or
            'ar' in plan.detected_languages  # Arabic benefits from smart processing
        )


def get_decision_agent() -> DecisionAgent:
    """Factory function to get decision agent."""
    return DecisionAgent()


def analyze_file(file_path: str, mime_type: str = None) -> ProcessingPlan:
    """
    Convenience function to analyze a file.

    Args:
        file_path: Path to the file
        mime_type: Optional MIME type

    Returns:
        ProcessingPlan
    """
    agent = DecisionAgent()
    return agent.analyze(file_path, mime_type)
