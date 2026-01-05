"""
Text extraction from various file formats.
Supports PDF, DOCX, TXT, and Markdown files.
"""
import frappe
from pathlib import Path
from typing import Optional


class TextExtractor:
    """Extract text content from various file formats."""

    SUPPORTED_EXTENSIONS = {
        "pdf": "_extract_pdf",
        "txt": "_extract_txt",
        "docx": "_extract_docx",
        "md": "_extract_markdown",
    }

    @classmethod
    def extract(cls, file_path: str) -> str:
        """Extract text from file based on extension."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")

        if extension not in cls.SUPPORTED_EXTENSIONS:
            frappe.throw(f"Unsupported file type: {extension}")

        method_name = cls.SUPPORTED_EXTENSIONS[extension]
        method = getattr(cls, method_name)
        return method(file_path)

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file type is supported based on settings."""
        if not file_path:
            return False

        extension = Path(file_path).suffix.lower().lstrip(".")

        # Check against enabled file types in settings
        try:
            settings = frappe.get_single("Data Pipeline Settings")
            enabled_types = [
                t.strip().lower()
                for t in (settings.enabled_file_types or "").split("\n")
                if t.strip()
            ]
            return extension in enabled_types and extension in cls.SUPPORTED_EXTENSIONS
        except Exception:
            # If settings don't exist yet, use defaults
            return extension in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def _extract_pdf(cls, file_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        import pdfplumber

        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    @classmethod
    def _extract_txt(cls, file_path: str) -> str:
        """Extract text from plain text file with encoding detection."""
        # Try UTF-8 first, then fall back to other encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort: ignore errors
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @classmethod
    def _extract_docx(cls, file_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    @classmethod
    def _extract_markdown(cls, file_path: str) -> str:
        """Extract text from Markdown file (keep as-is for semantic meaning)."""
        return cls._extract_txt(file_path)
