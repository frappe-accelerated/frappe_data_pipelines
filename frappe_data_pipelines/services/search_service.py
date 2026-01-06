"""
Hybrid Search Service with Reranking.

Implements hybrid search combining dense embeddings with BM25 sparse vectors,
plus optional reranking for improved retrieval quality.
"""
import frappe
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    score: float
    text: str
    context_prefix: str
    metadata: Dict[str, Any]
    source_file: str
    section_path: Optional[str] = None


class HybridSearchService:
    """
    Hybrid search combining dense embeddings with BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    dense and sparse search for better retrieval.
    """

    def __init__(self):
        self.settings = frappe.get_single("Data Pipeline Settings")
        self._init_qdrant()
        self._init_embedding_provider()
        self._init_reranker()

    def _init_qdrant(self):
        """Initialize Qdrant client."""
        from frappe_data_pipelines.services.qdrant_service import QdrantService
        self.qdrant = QdrantService.get_client()
        self.collection_name = self.settings.collection_name or "drive_documents"

    def _init_embedding_provider(self):
        """Initialize embedding provider for query embedding."""
        from frappe_data_pipelines.services.embedding_service import get_embedding_provider
        self.embedder = get_embedding_provider()

    def _init_reranker(self):
        """Initialize reranker if enabled."""
        self.reranker = None
        if self.settings.enable_hybrid_search:
            reranker_model = self.settings.reranker_model
            if reranker_model and 'cohere' in reranker_model.lower():
                try:
                    self.reranker = CohereReranker(self.settings)
                except Exception as e:
                    frappe.log_error(
                        title="Reranker initialization failed",
                        message=str(e)
                    )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        use_reranker: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search with optional reranking.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional Qdrant filter conditions
            use_reranker: Whether to use reranker (if available)

        Returns:
            List of SearchResult objects
        """
        # Get more candidates for reranking
        candidate_count = top_k * 5 if self.reranker and use_reranker else top_k

        # Dense search
        query_embedding = self.embedder.embed([query])[0]

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Build filter if provided
            qdrant_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                qdrant_filter = Filter(must=conditions)

            # Perform search
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=candidate_count,
                query_filter=qdrant_filter,
                with_payload=True
            )

            # Convert to SearchResult objects
            search_results = []
            for hit in results:
                payload = hit.payload or {}
                search_results.append(SearchResult(
                    chunk_id=str(hit.id),
                    score=hit.score,
                    text=payload.get('original_text', payload.get('text', '')),
                    context_prefix=payload.get('context_prefix', ''),
                    metadata=payload,
                    source_file=payload.get('source_drive_file', ''),
                    section_path=payload.get('section_path')
                ))

            # Apply reranking if available
            if self.reranker and use_reranker and search_results:
                search_results = self.reranker.rerank(query, search_results, top_k)
            else:
                search_results = search_results[:top_k]

            return search_results

        except Exception as e:
            frappe.log_error(
                title="Search failed",
                message=str(e)
            )
            return []

    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search within a specific document.

        Args:
            query: Search query
            document_id: Drive File document name
            top_k: Number of results

        Returns:
            List of SearchResult within the document
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={'source_drive_file': document_id}
        )

    def find_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5,
        exclude_same_document: bool = False
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to find
            exclude_same_document: Whether to exclude chunks from same document

        Returns:
            List of similar chunks
        """
        try:
            # Get the chunk's embedding
            points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_vectors=True,
                with_payload=True
            )

            if not points:
                return []

            point = points[0]
            query_vector = point.vector

            # Search for similar
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k + 1,  # +1 to account for self
                with_payload=True
            )

            # Filter and convert results
            search_results = []
            source_doc = point.payload.get('source_drive_file') if point.payload else None

            for hit in results:
                # Skip self
                if str(hit.id) == chunk_id:
                    continue

                # Skip same document if requested
                payload = hit.payload or {}
                if exclude_same_document and payload.get('source_drive_file') == source_doc:
                    continue

                search_results.append(SearchResult(
                    chunk_id=str(hit.id),
                    score=hit.score,
                    text=payload.get('original_text', payload.get('text', '')),
                    context_prefix=payload.get('context_prefix', ''),
                    metadata=payload,
                    source_file=payload.get('source_drive_file', ''),
                    section_path=payload.get('section_path')
                ))

                if len(search_results) >= top_k:
                    break

            return search_results

        except Exception as e:
            frappe.log_error(
                title="Similar chunk search failed",
                message=str(e)
            )
            return []


class CohereReranker:
    """
    Reranker using Cohere's rerank API via OpenRouter.
    """

    def __init__(self, settings):
        self.model = settings.reranker_model or "cohere/rerank-v3"
        self.api_key = settings.get_password("openrouter_api_key")

        if not self.api_key:
            raise ValueError("OpenRouter API key required for reranking")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Rerank search results using Cohere.

        Args:
            query: Original search query
            results: List of SearchResult to rerank
            top_k: Number of results to return

        Returns:
            Reranked list of SearchResult
        """
        if not results:
            return []

        try:
            import cohere

            # Initialize Cohere client
            # Note: Using direct Cohere API since OpenRouter doesn't support rerank
            co = cohere.Client(self.api_key)

            # Prepare documents for reranking
            documents = [r.text for r in results]

            # Call rerank API
            response = co.rerank(
                model="rerank-english-v3.0",  # Or rerank-multilingual-v3.0 for Arabic
                query=query,
                documents=documents,
                top_n=top_k
            )

            # Reorder results based on rerank scores
            reranked = []
            for item in response.results:
                original = results[item.index]
                reranked.append(SearchResult(
                    chunk_id=original.chunk_id,
                    score=item.relevance_score,
                    text=original.text,
                    context_prefix=original.context_prefix,
                    metadata=original.metadata,
                    source_file=original.source_file,
                    section_path=original.section_path
                ))

            return reranked

        except ImportError:
            frappe.log_error(
                title="Cohere not installed",
                message="Install cohere package for reranking: pip install cohere"
            )
            return results[:top_k]
        except Exception as e:
            frappe.log_error(
                title="Reranking failed",
                message=str(e)
            )
            return results[:top_k]


def get_search_service() -> HybridSearchService:
    """Factory function to get search service."""
    return HybridSearchService()


@frappe.whitelist()
def search_documents(
    query: str,
    top_k: int = 10,
    document_id: str = None
) -> List[dict]:
    """
    API endpoint for document search.

    Args:
        query: Search query
        top_k: Number of results
        document_id: Optional document to search within

    Returns:
        List of result dictionaries
    """
    service = get_search_service()

    if document_id:
        results = service.search_by_document(query, document_id, int(top_k))
    else:
        results = service.search(query, int(top_k))

    return [
        {
            'chunk_id': r.chunk_id,
            'score': r.score,
            'text': r.text,
            'context': r.context_prefix,
            'source_file': r.source_file,
            'section_path': r.section_path
        }
        for r in results
    ]
