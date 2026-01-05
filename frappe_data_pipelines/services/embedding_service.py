"""
Embedding provider abstraction layer.
Supports both OpenRouter API and local Sentence Transformers models.
"""
import frappe
from typing import List, Optional
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers library."""

    _model_cache = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy load model with caching."""
        if self._model is None:
            if self.model_name not in self._model_cache:
                from sentence_transformers import SentenceTransformer
                self._model_cache[self.model_name] = SentenceTransformer(self.model_name)
            self._model = self._model_cache[self.model_name]
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not texts:
            return []

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension from model."""
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension


class OpenRouterProvider(EmbeddingProvider):
    """Remote embedding using OpenRouter API (OpenAI-compatible)."""

    DIMENSION_MAP = {
        "openai/text-embedding-ada-002": 1536,
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
    }

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = self.DIMENSION_MAP.get(model, 1536)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenRouter API."""
        if not texts:
            return []

        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        response = client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )

        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def get_dimension(self) -> int:
        """Return the embedding dimension for the model."""
        return self._dimension


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get the configured embedding provider."""
    settings = frappe.get_single("Data Pipeline Settings")

    if settings.embedding_provider == "OpenRouter":
        api_key = settings.get_password("openrouter_api_key")
        if not api_key:
            frappe.throw("OpenRouter API key is required when using OpenRouter provider")
        return OpenRouterProvider(
            api_key=api_key,
            model=settings.openrouter_model or "openai/text-embedding-3-small"
        )
    else:
        # Default to local Sentence Transformers
        return SentenceTransformersProvider(
            model_name=settings.local_model_name or "all-MiniLM-L6-v2"
        )
