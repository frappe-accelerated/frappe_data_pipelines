"""
Document Chunk - Stores chunked document content with vector metadata
"""
import frappe
from frappe.model.document import Document


class DocumentChunk(Document):
    def before_save(self):
        if self.chunk_text:
            self.character_count = len(self.chunk_text)
