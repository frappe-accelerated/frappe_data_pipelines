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
    provider = None
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

        if provider is not None:
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

    # Update connection status without triggering modified timestamp
    status_parts = []
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        status_parts.append(f"{r['name']}: {status}")
    settings.db_set("connection_status", " | ".join(status_parts), update_modified=False)

    # Build HTML response
    html_parts = ["<ul style='list-style: none; padding-left: 0;'>"]
    for r in results:
        color = "green" if r["success"] else "red"
        status = "OK" if r["success"] else "FAILED"
        html_parts.append(f"<li><strong style='color: {color};'>[{status}]</strong> {r['name']}: {r['message']}</li>")
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

        # Update connection status without modifying timestamp
        settings = frappe.get_single("Data Pipeline Settings")
        if result["success"]:
            settings.db_set("connection_status", f"Connected - {result.get('collections_count', 0)} collections", update_modified=False)
        else:
            settings.db_set("connection_status", f"Failed: {result.get('message', 'Unknown error')}", update_modified=False)

        return result

    except Exception as e:
        frappe.log_error(title="Qdrant Connection Test Failed", message=str(e))
        return {
            "success": False,
            "message": str(e)
        }


@frappe.whitelist()
def process_existing_files():
    """
    Queue all existing Drive files for embedding processing.
    Useful for processing files uploaded before the app was installed.
    """
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    settings = frappe.get_single("Data Pipeline Settings")

    if not settings.enable_auto_processing:
        return {
            "success": False,
            "message": "Auto processing is not enabled in settings"
        }

    # Get all Drive files that are not folders
    drive_files = frappe.get_all(
        "Drive File",
        filters={"is_group": 0},
        fields=["name", "title", "path", "mime_type", "file_size"]
    )

    # Get already processed files
    existing_jobs = frappe.get_all(
        "Embedding Job",
        filters={"status": ["in", ["Queued", "Extracting Text", "Chunking", "Embedding", "Storing Vectors", "Completed"]]},
        pluck="source_drive_file"
    )

    queued_count = 0
    skipped_count = 0
    max_size = (settings.max_file_size_mb or 50) * 1024 * 1024

    for df in drive_files:
        # Skip if already processed or in progress
        if df.name in existing_jobs:
            skipped_count += 1
            continue

        # Skip if file type not supported
        if not df.path or not TextExtractor.is_supported(df.path):
            skipped_count += 1
            continue

        # Skip if file too large
        file_size = df.file_size or 0
        if file_size > max_size:
            skipped_count += 1
            continue

        # Create embedding job
        job = frappe.get_doc({
            "doctype": "Embedding Job",
            "source_drive_file": df.name,
            "file_path": df.path,
            "file_size_bytes": file_size,
            "file_mime_type": df.mime_type,
            "status": "Queued",
            "priority": "Normal"
        })
        job.insert(ignore_permissions=True)

        # Enqueue background job
        frappe.enqueue(
            "frappe_data_pipelines.tasks.process_embedding.process_embedding_job",
            queue="long",
            embedding_job_name=job.name,
            enqueue_after_commit=True
        )

        queued_count += 1

    frappe.db.commit()

    return {
        "success": True,
        "message": f"Queued {queued_count} files for processing. Skipped {skipped_count} files (already processed, unsupported type, or too large)."
    }


@frappe.whitelist()
def get_processing_stats():
    """Get statistics about embedding processing."""
    stats = {
        "total_drive_files": frappe.db.count("Drive File", {"is_group": 0}),
        "total_jobs": frappe.db.count("Embedding Job"),
        "queued_jobs": frappe.db.count("Embedding Job", {"status": "Queued"}),
        "processing_jobs": frappe.db.count("Embedding Job", {"status": ["in", ["Extracting Text", "Chunking", "Embedding", "Storing Vectors"]]}),
        "completed_jobs": frappe.db.count("Embedding Job", {"status": "Completed"}),
        "failed_jobs": frappe.db.count("Embedding Job", {"status": "Failed"}),
        "total_chunks": frappe.db.count("Document Chunk"),
    }

    return stats
