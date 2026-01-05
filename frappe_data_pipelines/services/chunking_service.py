"""
Text chunking service using LangChain text splitters.
"""
import frappe
from typing import List


class ChunkingService:
    """Split text into chunks for embedding."""

    @classmethod
    def get_splitter(cls):
        """Get configured text splitter."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        try:
            settings = frappe.get_single("Data Pipeline Settings")
            chunk_size = settings.chunk_size or 1000
            chunk_overlap = settings.chunk_overlap or 200
        except Exception:
            # Use defaults if settings don't exist
            chunk_size = 1000
            chunk_overlap = 200

        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    @classmethod
    def chunk_text(cls, text: str) -> List[str]:
        """Split text into chunks."""
        if not text or not text.strip():
            return []

        splitter = cls.get_splitter()
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
