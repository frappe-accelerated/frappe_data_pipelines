"""
Utility functions for frappe_data_pipelines.
"""
import frappe


def ensure_settings_exist():
    """
    Ensure Data Pipeline Settings singleton exists with sensible defaults.

    Returns:
        Data Pipeline Settings document or None if table doesn't exist
    """
    if not frappe.db.table_exists("Singles"):
        return None

    # Check if settings already exist
    existing = frappe.db.get_value(
        "Singles",
        {"doctype": "Data Pipeline Settings", "field": "enable_auto_processing"},
        "value"
    )

    if existing is None:
        # Create default settings
        try:
            doc = frappe.new_doc("Data Pipeline Settings")
            doc.enable_auto_processing = 1
            doc.enable_smart_pipeline = 1
            doc.max_file_size_mb = 50
            doc.chunk_size = 1000
            doc.chunk_overlap = 200
            doc.embedding_provider = "Local (Ollama)"
            doc.ollama_model = "nomic-embed-text"
            doc.ollama_url = "http://localhost:11434"
            doc.qdrant_mode = "Local"
            doc.qdrant_host = "localhost"
            doc.qdrant_port = 6333
            doc.collection_name = "drive_documents"
            doc.enabled_file_types = "pdf\ntxt\ndocx\nmd\njpg\njpeg\npng"
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            frappe.logger("frappe_data_pipelines").info(
                "Created default Data Pipeline Settings"
            )
        except Exception as e:
            frappe.log_error(
                title="Failed to create Data Pipeline Settings",
                message=str(e)
            )
            return None

    return frappe.get_single("Data Pipeline Settings")


def get_or_create_settings():
    """
    Get Data Pipeline Settings, creating defaults if needed.

    This is a safe wrapper that handles all edge cases:
    - Table doesn't exist (migration not run)
    - Settings not initialized
    - Any other unexpected errors

    Returns:
        Data Pipeline Settings document or None on failure
    """
    try:
        return ensure_settings_exist()
    except Exception as e:
        frappe.log_error(
            title="Pipeline Settings Init Failed",
            message=f"Failed to get or create Data Pipeline Settings: {str(e)}"
        )
        return None
