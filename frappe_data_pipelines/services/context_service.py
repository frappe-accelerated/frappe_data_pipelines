"""
Contextual Enrichment Service.

Implements Anthropic's Contextual Retrieval approach by generating
situating context for each chunk before embedding.

Reference: https://www.anthropic.com/engineering/contextual-retrieval
"""
import frappe
import requests
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
    """Generate contextual prefixes for chunks using LLM."""

    CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>

Here is chunk {position} from "{title}":
<chunk>
{chunk}
</chunk>

Provide a brief context (2-3 sentences) that situates this chunk within the overall document. Include:
- What section/topic this belongs to
- Key entities or concepts referenced
- How it relates to the document's main theme

Answer only with the context, nothing else."""

    def __init__(self):
        self.settings = frappe.get_single("Data Pipeline Settings")
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
        total_chunks = len(chunks)

        # Truncate document to avoid context limits (50K chars ~= 12K tokens)
        doc_for_context = full_document[:50000]
        if len(full_document) > 50000:
            doc_for_context += "\n\n[Document truncated for context generation...]"

        for i, chunk in enumerate(chunks):
            try:
                context = self._generate_context(
                    chunk=chunk,
                    document=doc_for_context,
                    title=document_title,
                    position=f"{i + 1}/{total_chunks}"
                )
            except Exception as e:
                frappe.log_error(
                    title=f"Context generation failed for chunk {i + 1}",
                    message=str(e)
                )
                # Fall back to empty context if generation fails
                context = ""

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

    def _generate_context(
        self,
        chunk: str,
        document: str,
        title: str,
        position: str
    ) -> str:
        """
        Generate contextual prefix for a single chunk using OpenRouter.

        Args:
            chunk: The chunk text to contextualize
            document: The full document for context
            title: Document title
            position: Position string like "3/10"

        Returns:
            Generated context string
        """
        prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
            document=document,
            chunk=chunk,
            title=title,
            position=position
        )

        return self._call_openrouter(prompt)

    def _call_openrouter(self, prompt: str) -> str:
        """
        Call OpenRouter API with the given prompt.

        Args:
            prompt: The prompt to send

        Returns:
            Generated text response
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
            max_tokens=300,  # Context should be brief
            temperature=0.3  # Lower temperature for more consistent output
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
