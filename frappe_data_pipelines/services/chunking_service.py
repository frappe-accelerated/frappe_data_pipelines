"""
Text chunking service with semantic chunking support.

Supports both legacy character-based chunking and smart semantic chunking
that preserves document structure.
"""
import frappe
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SemanticChunk:
    """A chunk with semantic metadata."""
    text: str
    section_path: str  # e.g., "Chapter 1 > Introduction"
    chunk_index: int
    start_char: int
    end_char: int


class ChunkingService:
    """Split text into chunks for embedding."""

    @classmethod
    def get_splitter(cls):
        """Get configured text splitter for legacy mode."""
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
        """
        Split text into chunks (legacy mode).

        For backward compatibility, returns simple list of strings.
        """
        if not text or not text.strip():
            return []

        splitter = cls.get_splitter()
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class SemanticChunker:
    """
    Smart semantic chunker that preserves document structure.

    Detects headers, sections, and paragraphs to create semantically
    meaningful chunks instead of arbitrary character-based splits.
    """

    # Patterns for detecting document structure
    HEADER_PATTERNS = [
        # Markdown headers
        (r'^#{1,6}\s+(.+)$', 'markdown'),
        # Numbered sections (1., 1.1., etc.)
        (r'^(\d+(?:\.\d+)*\.?)\s+(.+)$', 'numbered'),
        # ALL CAPS headers
        (r'^([A-Z][A-Z\s]{5,})$', 'caps'),
        # Headers with colons
        (r'^([A-Za-z][^:]{3,50}):\s*$', 'colon'),
    ]

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_type: str = "general") -> List[SemanticChunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Document text to chunk
            doc_type: Type of document (general, markdown, code, etc.)

        Returns:
            List of SemanticChunk objects with section paths
        """
        if not text or not text.strip():
            return []

        # Detect and extract sections
        sections = self._detect_sections(text, doc_type)

        # Process sections into chunks
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _detect_sections(self, text: str, doc_type: str) -> List[dict]:
        """
        Detect document sections based on headers and structure.

        Returns list of dicts with 'path', 'text', 'start', 'end'
        """
        lines = text.split('\n')
        sections = []
        current_section = {
            'path': 'Document',
            'lines': [],
            'start': 0,
            'headers': []
        }
        current_pos = 0

        for line in lines:
            line_start = current_pos
            current_pos += len(line) + 1  # +1 for newline

            # Check if this line is a header
            header_match = self._match_header(line, doc_type)

            if header_match:
                # Save current section if it has content
                if current_section['lines']:
                    section_text = '\n'.join(current_section['lines'])
                    if section_text.strip():
                        sections.append({
                            'path': ' > '.join(current_section['headers']) or 'Document',
                            'text': section_text.strip(),
                            'start': current_section['start'],
                            'end': line_start
                        })

                # Update header stack
                level, title = header_match
                # Trim headers to appropriate level
                current_section['headers'] = current_section['headers'][:level - 1]
                current_section['headers'].append(title)
                current_section['lines'] = []
                current_section['start'] = current_pos
            else:
                current_section['lines'].append(line)

        # Don't forget the last section
        if current_section['lines']:
            section_text = '\n'.join(current_section['lines'])
            if section_text.strip():
                sections.append({
                    'path': ' > '.join(current_section['headers']) or 'Document',
                    'text': section_text.strip(),
                    'start': current_section['start'],
                    'end': current_pos
                })

        # If no sections detected, treat entire doc as one section
        if not sections:
            sections = [{
                'path': 'Document',
                'text': text.strip(),
                'start': 0,
                'end': len(text)
            }]

        return sections

    def _match_header(self, line: str, doc_type: str) -> Optional[Tuple[int, str]]:
        """
        Check if a line is a header.

        Returns (level, title) tuple if header, None otherwise.
        """
        line = line.strip()
        if not line:
            return None

        # Markdown headers
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            return (level, title)

        # Numbered sections (1., 1.1., 1.1.1., etc.)
        num_match = re.match(r'^(\d+(?:\.\d+)*\.?)\s+(.+)$', line)
        if num_match:
            num = num_match.group(1)
            level = len(num.split('.'))
            title = f"{num} {num_match.group(2).strip()}"
            return (level, title)

        # ALL CAPS headers (must be reasonable length)
        if line.isupper() and 5 <= len(line) <= 60:
            return (1, line.title())

        return None

    def _chunk_section(self, section: dict, start_index: int) -> List[SemanticChunk]:
        """
        Break a section into appropriately sized chunks.
        """
        text = section['text']
        path = section['path']

        # If section fits in one chunk, return as-is
        if len(text) <= self.max_chunk_size:
            return [SemanticChunk(
                text=text,
                section_path=path,
                chunk_index=start_index,
                start_char=section['start'],
                end_char=section['end']
            )]

        # Split at paragraph boundaries first
        chunks = []
        paragraphs = self._split_paragraphs(text)
        current_chunk_text = ""
        current_start = section['start']
        chunk_idx = start_index

        for para in paragraphs:
            # Would adding this paragraph exceed the limit?
            test_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + para

            if len(test_text) > self.max_chunk_size and current_chunk_text:
                # Save current chunk
                chunks.append(SemanticChunk(
                    text=current_chunk_text.strip(),
                    section_path=path,
                    chunk_index=chunk_idx,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk_text)
                ))
                chunk_idx += 1
                current_start = current_start + len(current_chunk_text)
                current_chunk_text = para
            else:
                current_chunk_text = test_text

            # If a single paragraph is too long, split it at sentences
            if len(current_chunk_text) > self.max_chunk_size:
                sentence_chunks = self._split_long_text(current_chunk_text, path, chunk_idx, current_start)
                chunks.extend(sentence_chunks)
                chunk_idx += len(sentence_chunks)
                current_start += len(current_chunk_text)
                current_chunk_text = ""

        # Don't forget remaining text
        if current_chunk_text.strip():
            chunks.append(SemanticChunk(
                text=current_chunk_text.strip(),
                section_path=path,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=section['end']
            ))

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_text(
        self,
        text: str,
        path: str,
        start_index: int,
        start_char: int
    ) -> List[SemanticChunk]:
        """
        Split long text at sentence boundaries.
        Falls back to character-based splitting if needed.
        """
        # Try splitting at sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_text = ""
        chunk_idx = start_index
        current_start = start_char

        for sentence in sentences:
            test_text = current_text + (" " if current_text else "") + sentence

            if len(test_text) > self.max_chunk_size and current_text:
                chunks.append(SemanticChunk(
                    text=current_text.strip(),
                    section_path=path,
                    chunk_index=chunk_idx,
                    start_char=current_start,
                    end_char=current_start + len(current_text)
                ))
                chunk_idx += 1
                current_start += len(current_text) + 1
                current_text = sentence
            else:
                current_text = test_text

        if current_text.strip():
            chunks.append(SemanticChunk(
                text=current_text.strip(),
                section_path=path,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=start_char + len(text)
            ))

        return chunks

    def get_section_paths(self, chunks: List[SemanticChunk]) -> List[str]:
        """Extract section paths from chunks."""
        return [chunk.section_path for chunk in chunks]

    def get_texts(self, chunks: List[SemanticChunk]) -> List[str]:
        """Extract text from chunks."""
        return [chunk.text for chunk in chunks]


def get_chunker(use_semantic: bool = True) -> object:
    """
    Factory function to get the appropriate chunker.

    Args:
        use_semantic: Whether to use semantic chunking

    Returns:
        ChunkingService or SemanticChunker instance
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")

        if use_semantic and settings.enable_smart_pipeline and settings.enable_semantic_chunking:
            return SemanticChunker(
                max_chunk_size=settings.chunk_size or 1000,
                chunk_overlap=settings.chunk_overlap or 200
            )
    except Exception:
        pass

    # Fall back to legacy chunker
    return ChunkingService()
