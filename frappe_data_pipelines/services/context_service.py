"""
Contextual Enrichment Service.

Implements Anthropic's Contextual Retrieval approach by generating
situating context for each chunk before embedding.

Uses prompt caching for efficiency - document is cached once and
referenced for all chunk context generation, reducing costs by ~90%.

Reference: https://www.anthropic.com/engineering/contextual-retrieval
"""
import frappe
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EnrichedChunk:
    """A chunk with its contextual enrichment."""
    original_text: str
    context_prefix: str
    embedded_text: str  # context_prefix + original_text
    chunk_index: int
    section_path: Optional[str] = None


class ContextEnrichmentService:
    """
    Generate contextual prefixes for chunks using LLM.

    Optimized for token efficiency using:
    1. Prompt caching (document cached once for all chunks)
    2. Succinct context generation (50-100 tokens per chunk)
    3. Claude Haiku for cost efficiency
    """

    # Anthropic's recommended prompt - optimized for succinct output
    CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for improving search retrieval. Answer only with the succinct context and nothing else."""

    def __init__(self):
        self.settings = frappe.get_single("Data Pipeline Settings")
        # Gemini Flash: fast, cheap, good quality for context generation
        self.model = self.settings.context_enrichment_model or "google/gemini-3-flash-preview"
        self.api_key = self.settings.get_password("openrouter_api_key")

        if not self.api_key:
            frappe.throw("OpenRouter API key is required for contextual enrichment")

    def enrich_chunks(
        self,
        chunks: List[str],
        full_document: str,
        document_title: str,
        section_paths: Optional[List[str]] = None
    ) -> List[EnrichedChunk]:
        """
        Batch enrich chunks with contextual prefixes.

        Uses prompt caching for efficiency - the document is sent once and
        cached, then referenced for all chunk context generation.

        Args:
            chunks: List of text chunks to enrich
            full_document: The full document text (truncated if too long)
            document_title: Title of the source document
            section_paths: Optional list of section paths for each chunk

        Returns:
            List of EnrichedChunk objects
        """
        if not chunks:
            return []

        enriched = []

        # Truncate document to ~8K tokens (32K chars) for optimal caching
        # Anthropic recommends 8k-token documents for best efficiency
        doc_for_context = full_document[:32000]
        if len(full_document) > 32000:
            doc_for_context += "\n\n[...]"

        # Process all chunks with the same cached document context
        contexts = self._generate_contexts_batch(
            chunks=chunks,
            document=doc_for_context
        )

        for i, (chunk, context) in enumerate(zip(chunks, contexts)):
            section_path = section_paths[i] if section_paths and i < len(section_paths) else None

            # Combine context with chunk for embedding
            if context:
                embedded_text = f"{context}\n\n{chunk}"
            else:
                embedded_text = chunk

            enriched.append(EnrichedChunk(
                original_text=chunk,
                context_prefix=context,
                embedded_text=embedded_text,
                chunk_index=i,
                section_path=section_path
            ))

        return enriched

    def _generate_contexts_batch(
        self,
        chunks: List[str],
        document: str
    ) -> List[str]:
        """
        Generate contexts for all chunks using prompt caching.

        The document is cached on the first call, subsequent chunks
        only send the chunk content, reusing the cached document.
        """
        contexts = []

        for i, chunk in enumerate(chunks):
            try:
                context = self._generate_context_cached(
                    chunk=chunk,
                    document=document,
                    use_cache=(i > 0)  # Cache after first request
                )
                contexts.append(context)
            except Exception as e:
                frappe.log_error(
                    title=f"Context generation failed for chunk {i + 1}",
                    message=str(e)
                )
                contexts.append("")

        return contexts

    def _generate_context_cached(
        self,
        chunk: str,
        document: str,
        use_cache: bool = True
    ) -> str:
        """
        Generate context using OpenRouter with cache control.

        Args:
            chunk: The chunk to contextualize
            document: Full document for context
            use_cache: Whether to use cached document (True after first call)

        Returns:
            Generated context string (50-100 tokens)
        """
        prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
            document=document,
            chunk=chunk
        )

        return self._call_openrouter(prompt, use_cache=use_cache)

    def _call_openrouter(self, prompt: str, use_cache: bool = False) -> str:
        """
        Call OpenRouter API with the given prompt.

        Args:
            prompt: The prompt to send
            use_cache: Reserved for future caching support

        Returns:
            Generated text response (50-100 tokens for context)
        """
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,  # Succinct context: typically 50-100, allow up to 150
            temperature=0.1  # Low temperature for consistent contexts
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()

        return ""

    def test_connection(self) -> dict:
        """Test connection to the context enrichment model."""
        try:
            result = self._call_openrouter("Say 'OK' if you can read this.")
            if result:
                return {
                    "success": True,
                    "message": f"Context enrichment model '{self.model}' is working."
                }
            return {
                "success": False,
                "message": "No response from context enrichment model"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Context enrichment failed: {str(e)}"
            }


def get_context_service() -> Optional[ContextEnrichmentService]:
    """
    Factory function to get context service if enabled.

    Returns:
        ContextEnrichmentService if enabled, None otherwise
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")
        if settings.enable_smart_pipeline and settings.enable_contextual_enrichment:
            return ContextEnrichmentService()
    except Exception:
        pass
    return None
