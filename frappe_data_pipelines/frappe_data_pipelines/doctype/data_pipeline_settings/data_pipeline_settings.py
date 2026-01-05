"""
Data Pipeline Settings - Configuration for document embedding pipelines
"""
import frappe
from frappe.model.document import Document


class DataPipelineSettings(Document):
    pass


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
