"""
Text chunking service with Docling-based smart chunking.

Uses Docling's HybridChunker for context-aware, structure-preserving chunking.
Falls back to legacy character-based chunking if Docling is unavailable.
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


class DoclingChunker:
    """
    Smart chunker using docling-core's HybridChunker.

    Features:
    - Structure-aware chunking that respects document hierarchy
    - Automatic merging of small peer chunks
    - Preserves headings and context
    - Uses lightweight docling-core (no heavy numpy/torch dependencies)
    """

    def __init__(self):
        """Initialize Docling chunker."""
        pass

    def chunk_text(self, text: str, doc_type: str = "text") -> List[SemanticChunk]:
        """
        Chunk plain text using docling-core's HybridChunker.

        Args:
            text: Plain text to chunk
            doc_type: Type hint for the document

        Returns:
            List of SemanticChunk objects
        """
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.labels import DocItemLabel
            from docling_core.transforms.chunker import HybridChunker

            # Create a DoclingDocument from plain text
            # Split into paragraphs and add each as a paragraph item
            doc = DoclingDocument(name="text_document")

            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [text.strip()] if text.strip() else []

            for para in paragraphs:
                doc.add_text(label=DocItemLabel.PARAGRAPH, text=para)

            # Create hybrid chunker with peer merging
            chunker = HybridChunker(merge_peers=True)

            # Chunk the document
            chunks = []
            for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
                section_path = self._build_section_path(chunk)

                chunks.append(SemanticChunk(
                    text=chunk.text,
                    section_path=section_path or "Document",
                    chunk_index=i,
                    start_char=0,
                    end_char=len(chunk.text)
                ))

            return chunks if chunks else self._fallback_chunk(text)

        except Exception as e:
            frappe.log_error(
                title="Docling chunking failed",
                message=f"Falling back to legacy chunker: {str(e)}"
            )
            return self._fallback_chunk(text)

    def _build_section_path(self, chunk) -> str:
        """Build section path from chunk metadata."""
        try:
            # Docling chunks have metadata with headings
            if hasattr(chunk, 'meta') and chunk.meta:
                headings = getattr(chunk.meta, 'headings', None)
                if headings:
                    return " > ".join(headings)

                # Try doc_items for section info
                doc_items = getattr(chunk.meta, 'doc_items', None)
                if doc_items and len(doc_items) > 0:
                    first_item = doc_items[0]
                    if hasattr(first_item, 'label'):
                        return str(first_item.label)

            return "Document"
        except Exception:
            return "Document"

    def _fallback_chunk(self, text: str) -> List[SemanticChunk]:
        """Fallback to simple chunking if Docling fails."""
        legacy = ChunkingService()
        simple_chunks = legacy.chunk_text(text)

        return [
            SemanticChunk(
                text=chunk,
                section_path="Document",
                chunk_index=i,
                start_char=0,
                end_char=len(chunk)
            )
            for i, chunk in enumerate(simple_chunks)
        ]


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

    # Target ~500 tokens = ~2000 chars - this is a soft target, not enforced
    DEFAULT_TARGET_SIZE = 2000  # ~500 tokens

    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 400):
        # Soft target - chunks can be larger or smaller to preserve semantic boundaries
        self.target_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap  # ~100 tokens

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

        # Merge small sections to avoid tiny chunks
        sections = self._merge_small_sections(sections)

        # Process sections into chunks
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _merge_small_sections(self, sections: List[dict]) -> List[dict]:
        """
        Adaptively merge sections aiming for target chunk size (~500 tokens).

        Strategy:
        - Accumulate sections until we're near target size
        - Target is soft - chunks can be larger/smaller to preserve semantic boundaries
        - Keep all content regardless of size
        """
        if not sections or len(sections) <= 1:
            return sections

        merged = []
        accumulator = None

        for section in sections:
            if accumulator is None:
                # Start new accumulator
                accumulator = section.copy()
                continue

            acc_len = len(accumulator['text'])

            # If accumulator is below target, merge to build it up
            if acc_len < self.target_chunk_size:
                # Merge sections to build up toward target size
                accumulator = {
                    'path': section['path'],  # Use latest section's path
                    'text': accumulator['text'] + "\n\n" + section['text'],
                    'start': accumulator['start'],
                    'end': section['end']
                }
            else:
                # Accumulator is at/above target size, start new chunk
                merged.append(accumulator)
                accumulator = section.copy()

        # Don't forget the last accumulator
        if accumulator:
            merged.append(accumulator)

        return merged

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
        # If section fits in target size, return as-is
        if len(text) <= self.target_chunk_size:
            return [SemanticChunk(
                text=text,
                section_path=path,
                chunk_index=start_index,
                start_char=section['start'],
                end_char=section['end']
            )]

        # Split at paragraph boundaries first, aiming for target size
        chunks = []
        paragraphs = self._split_paragraphs(text)
        current_chunk_text = ""
        current_start = section['start']
        chunk_idx = start_index

        for para in paragraphs:
            # Would adding this paragraph exceed target?
            test_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + para

            # If we're at/above target and have content, start new chunk
            if len(current_chunk_text) >= self.target_chunk_size and current_chunk_text:
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


def get_chunker(use_semantic: bool = True, use_docling: bool = True) -> object:
    """
    Factory function to get the appropriate chunker.

    Args:
        use_semantic: Whether to use semantic chunking
        use_docling: Whether to try Docling first (recommended)

    Returns:
        DoclingChunker, SemanticChunker, or ChunkingService instance
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")

        if use_semantic and settings.enable_smart_pipeline and settings.enable_semantic_chunking:
            # Try Docling first (best quality) - uses lightweight docling-core
            if use_docling:
                try:
                    # Test if docling-core is available
                    from docling_core.transforms.chunker import HybridChunker
                    return DoclingChunker()
                except ImportError:
                    frappe.log_error(
                        title="docling-core not available",
                        message="Falling back to SemanticChunker. Install with: pip install docling-core[chunking]"
                    )

            # Fall back to our custom semantic chunker
            chunk_size = settings.chunk_size or 2000  # ~500 tokens
            chunk_overlap = settings.chunk_overlap or 400  # ~100 tokens
            return SemanticChunker(
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    except Exception:
        pass

    # Fall back to legacy chunker
    return ChunkingService()
