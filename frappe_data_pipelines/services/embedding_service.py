"""
Embedding provider abstraction layer.
Supports Ollama for local embeddings and OpenRouter API for cloud embeddings.
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
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
    "cohere/embed-english-v3.0": 1024,
    "cohere/embed-multilingual-v3.0": 1024,
    "cohere/embed-english-light-v3.0": 384,
}


def get_model_dimension(provider: str, model: str) -> int:
    """Get the embedding dimension for a given provider and model."""
    if provider == "Local (Ollama)":
        return OLLAMA_DIMENSIONS.get(model, 768)
    elif provider == "OpenRouter":
        return OPENROUTER_DIMENSIONS.get(model, 1536)
    return 768  # Default fallback


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
        self._dimension = OPENROUTER_DIMENSIONS.get(model, 1536)

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
                return {"success": True, "message": f"Connected to OpenRouter. Model '{self.model}' is working."}
            return {"success": False, "message": "No response from OpenRouter"}
        except Exception as e:
            return {"success": False, "message": str(e)}


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
        # Default to Ollama
        return OllamaProvider(
            model=settings.ollama_model or "nomic-embed-text",
            base_url=settings.ollama_url or "http://localhost:11434"
        )
