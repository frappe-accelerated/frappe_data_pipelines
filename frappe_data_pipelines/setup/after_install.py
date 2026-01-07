"""
Post-installation setup for frappe_data_pipelines.
"""
import frappe


def execute():
    """
    Initialize default settings after app installation or migration.

    This ensures that Data Pipeline Settings exist with sensible defaults
    so that the Drive file upload hooks can function properly.
    """
    from frappe_data_pipelines.utils import ensure_settings_exist

    try:
        settings = ensure_settings_exist()
        if settings:
            frappe.logger("frappe_data_pipelines").info(
                "Data Pipeline Settings initialized successfully"
            )
    except Exception as e:
        frappe.log_error(
            title="Pipeline Setup Failed",
            message=f"Failed to initialize Data Pipeline Settings: {str(e)}"
        )
