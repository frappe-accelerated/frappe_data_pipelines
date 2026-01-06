"""
Embedding provider abstraction layer.

Supports:
- Ollama for local embeddings
- OpenRouter API for cloud embeddings
- Smart Pipeline v2 with user-configurable models
"""
import frappe
import requests
from typing import List, Optional
from abc import ABC, abstractmethod


# Model dimension mappings
OLLAMA_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}

OPENROUTER_DIMENSIONS = {
    # OpenAI models
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
    # Cohere models
    "cohere/embed-english-v3.0": 1024,
    "cohere/embed-multilingual-v3.0": 1024,
    "cohere/embed-english-light-v3.0": 384,
    # Voyage models
    "voyageai/voyage-3-large": 1024,
    "voyageai/voyage-3.5-lite": 1024,
    "voyageai/voyage-3": 1024,
    # Qwen models
    "qwen/qwen3-embedding-8b": 4096,
    "qwen/qwen3-embedding-4b": 2048,
    # Alibaba models
    "alibaba/gte-qwen2-7b-instruct": 3584,
}

# Default dimension for unknown models
DEFAULT_DIMENSION = 1536


def get_model_dimension(provider: str, model: str) -> int:
    """
    Get the embedding dimension for a given provider and model.

    Args:
        provider: "Local (Ollama)" or "OpenRouter"
        model: Model identifier

    Returns:
        Embedding dimension (vector size)
    """
    if provider == "Local (Ollama)":
        return OLLAMA_DIMENSIONS.get(model, 768)
    elif provider == "OpenRouter":
        return OPENROUTER_DIMENSIONS.get(model, DEFAULT_DIMENSION)

    # For smart pipeline, check all known models
    if model in OPENROUTER_DIMENSIONS:
        return OPENROUTER_DIMENSIONS[model]
    if model in OLLAMA_DIMENSIONS:
        return OLLAMA_DIMENSIONS[model]

    return DEFAULT_DIMENSION


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


class OllamaProvider(EmbeddingProvider):
    """Local embedding using Ollama API."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = OLLAMA_DIMENSIONS.get(model, 768)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama API."""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except requests.exceptions.RequestException as e:
                frappe.log_error(f"Ollama embedding error: {str(e)}", "Embedding Service")
                raise frappe.ValidationError(f"Failed to generate embedding: {str(e)}")

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        return self._dimension

    def test_connection(self) -> dict:
        """Test connection to Ollama server."""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if self.model not in model_names:
                return {
                    "success": False,
                    "message": f"Model '{self.model}' not found. Available models: {', '.join(model_names) or 'None'}. Run 'ollama pull {self.model}' to download it."
                }

            return {"success": True, "message": f"Connected to Ollama. Model '{self.model}' is available."}
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"}
        except Exception as e:
            return {"success": False, "message": str(e)}


class OpenRouterProvider(EmbeddingProvider):
    """Remote embedding using OpenRouter API (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = OPENROUTER_DIMENSIONS.get(model, DEFAULT_DIMENSION)

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

    def test_connection(self) -> dict:
        """Test connection to OpenRouter API."""
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

            # Try to get a simple embedding
            response = client.embeddings.create(
                model=self.model,
                input=["test"],
                encoding_format="float"
            )

            if response.data:
                dim = len(response.data[0].embedding)
                return {
                    "success": True,
                    "message": f"Connected to OpenRouter. Model '{self.model}' is working. Dimension: {dim}"
                }
            return {"success": False, "message": "No response from OpenRouter"}
        except Exception as e:
            return {"success": False, "message": str(e)}


class SmartPipelineProvider(EmbeddingProvider):
    """
    Embedding provider for Smart Pipeline v2.

    Uses user-configured model from Data Pipeline Settings.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._dimension = OPENROUTER_DIMENSIONS.get(model, DEFAULT_DIMENSION)
        self._detected_dimension = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured model."""
        if not texts:
            return []

        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        try:
            response = client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]

            # Detect dimension from first response
            if embeddings and not self._detected_dimension:
                self._detected_dimension = len(embeddings[0])

            return embeddings

        except Exception as e:
            frappe.log_error(
                title=f"Smart Pipeline embedding failed: {self.model}",
                message=str(e)
            )
            raise

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        if self._detected_dimension:
            return self._detected_dimension
        return self._dimension

    def test_connection(self) -> dict:
        """Test the embedding model."""
        try:
            embeddings = self.embed(["test"])
            if embeddings:
                dim = len(embeddings[0])
                return {
                    "success": True,
                    "message": f"Smart Pipeline embedding model '{self.model}' is working. Dimension: {dim}"
                }
            return {"success": False, "message": "No embeddings returned"}
        except Exception as e:
            return {"success": False, "message": str(e)}


def get_embedding_provider() -> EmbeddingProvider:
    """
    Factory function to get the configured embedding provider.

    Checks for Smart Pipeline first, then falls back to legacy provider selection.

    Returns:
        EmbeddingProvider instance
    """
    settings = frappe.get_single("Data Pipeline Settings")

    # Check if Smart Pipeline is enabled with a v2 embedding model
    if getattr(settings, 'enable_smart_pipeline', False):
        embedding_model_v2 = getattr(settings, 'embedding_model_v2', None)
        if embedding_model_v2:
            api_key = settings.get_password("openrouter_api_key")
            if api_key:
                return SmartPipelineProvider(
                    api_key=api_key,
                    model=embedding_model_v2
                )

    # Fall back to legacy provider selection
    if settings.embedding_provider == "OpenRouter":
        api_key = settings.get_password("openrouter_api_key")
        if not api_key:
            frappe.throw("OpenRouter API key is required when using OpenRouter provider")
        return OpenRouterProvider(
            api_key=api_key,
            model=settings.openrouter_model or "openai/text-embedding-3-small"
        )
    else:
        # Default to Ollama
        return OllamaProvider(
            model=settings.ollama_model or "nomic-embed-text",
            base_url=settings.ollama_url or "http://localhost:11434"
        )


def get_smart_embedding_dimension() -> int:
    """
    Get the embedding dimension for the smart pipeline model.

    Returns:
        Dimension of the configured embedding model
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")
        if settings.enable_smart_pipeline and settings.embedding_model_v2:
            return get_model_dimension("OpenRouter", settings.embedding_model_v2)
    except Exception:
        pass
    return DEFAULT_DIMENSION
