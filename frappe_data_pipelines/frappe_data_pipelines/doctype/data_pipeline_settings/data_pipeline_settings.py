"""
Data Pipeline Settings - Configuration for document embedding pipelines
"""
import frappe
from frappe.model.document import Document
from frappe_data_pipelines.services.embedding_service import (
    get_model_dimension,
    OllamaProvider,
    OpenRouterProvider,
)


class DataPipelineSettings(Document):
    def before_save(self):
        """Auto-set embedding dimension based on provider and model selection."""
        self.update_embedding_dimension()

    def update_embedding_dimension(self):
        """Update the embedding dimension based on current provider/model."""
        if self.embedding_provider == "Local (Ollama)":
            model = self.ollama_model or "nomic-embed-text"
        else:
            model = self.openrouter_model or "openai/text-embedding-3-small"

        self.embedding_dimension = get_model_dimension(self.embedding_provider, model)


@frappe.whitelist()
def test_connections():
    """Test both embedding provider and Qdrant connections."""
    results = []
    overall_success = True

    settings = frappe.get_single("Data Pipeline Settings")

    # Test embedding provider
    try:
        if settings.embedding_provider == "Local (Ollama)":
            provider = OllamaProvider(
                model=settings.ollama_model or "nomic-embed-text",
                base_url=settings.ollama_url or "http://localhost:11434"
            )
        else:
            api_key = settings.get_password("openrouter_api_key")
            if not api_key:
                results.append({
                    "name": "Embedding Provider",
                    "success": False,
                    "message": "OpenRouter API key is required"
                })
                overall_success = False
            else:
                provider = OpenRouterProvider(
                    api_key=api_key,
                    model=settings.openrouter_model or "openai/text-embedding-3-small"
                )

        if "provider" in dir():
            embed_result = provider.test_connection()
            results.append({
                "name": "Embedding Provider",
                "success": embed_result["success"],
                "message": embed_result["message"]
            })
            if not embed_result["success"]:
                overall_success = False
    except Exception as e:
        results.append({
            "name": "Embedding Provider",
            "success": False,
            "message": str(e)
        })
        overall_success = False

    # Test Qdrant connection
    try:
        from frappe_data_pipelines.services.qdrant_service import QdrantService
        qdrant_result = QdrantService.test_connection()
        results.append({
            "name": "Qdrant Database",
            "success": qdrant_result["success"],
            "message": qdrant_result.get("message", f"{qdrant_result.get('collections_count', 0)} collections")
        })
        if not qdrant_result["success"]:
            overall_success = False
    except Exception as e:
        results.append({
            "name": "Qdrant Database",
            "success": False,
            "message": str(e)
        })
        overall_success = False

    # Update connection status
    status_parts = []
    for r in results:
        icon = "✓" if r["success"] else "✗"
        status_parts.append(f"{icon} {r['name']}")
    settings.db_set("connection_status", " | ".join(status_parts))

    # Build HTML response
    html_parts = ["<ul style='list-style: none; padding-left: 0;'>"]
    for r in results:
        icon = "✅" if r["success"] else "❌"
        html_parts.append(f"<li>{icon} <strong>{r['name']}</strong>: {r['message']}</li>")
    html_parts.append("</ul>")

    return {
        "success": overall_success,
        "html": "".join(html_parts),
        "results": results
    }


@frappe.whitelist()
def test_qdrant_connection():
    """Test the Qdrant connection based on current settings."""
    try:
        from frappe_data_pipelines.services.qdrant_service import QdrantService

        result = QdrantService.test_connection()

        # Update connection status
        settings = frappe.get_single("Data Pipeline Settings")
        if result["success"]:
            settings.db_set("connection_status", f"Connected - {result.get('collections_count', 0)} collections")
        else:
            settings.db_set("connection_status", f"Failed: {result.get('message', 'Unknown error')}")

        return result

    except Exception as e:
        frappe.log_error(title="Qdrant Connection Test Failed", message=str(e))
        return {
            "success": False,
            "message": str(e)
        }
