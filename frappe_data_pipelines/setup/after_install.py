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

    # Setup workspace sidebar
    try:
        setup_workspace_sidebar()
        frappe.logger("frappe_data_pipelines").info(
            "Workspace sidebar configured successfully"
        )
    except Exception as e:
        frappe.log_error(
            title="Workspace Sidebar Setup Failed",
            message=f"Failed to configure workspace sidebar: {str(e)}"
        )

    # Setup desktop icon for desk page
    try:
        setup_desktop_icon()
        frappe.logger("frappe_data_pipelines").info(
            "Desktop icon configured successfully"
        )
    except Exception as e:
        frappe.log_error(
            title="Desktop Icon Setup Failed",
            message=f"Failed to configure desktop icon: {str(e)}"
        )


def setup_workspace_sidebar():
    """
    Configure the Workspace Sidebar for Frappe Data Pipelines.

    This adds sidebar navigation items for the Data Pipelines workspace,
    making Document Chunks, Embedding Jobs, and Settings accessible from the sidebar.
    """
    sidebar_name = "Frappe Data Pipelines"

    # Check if the sidebar exists
    if not frappe.db.exists("Workspace Sidebar", sidebar_name):
        frappe.logger("frappe_data_pipelines").info(
            f"Workspace Sidebar '{sidebar_name}' does not exist yet, skipping setup"
        )
        return

    sidebar = frappe.get_doc("Workspace Sidebar", sidebar_name)

    # Define the expected sidebar items
    expected_items = [
        {"label": "Home", "link_to": "Frappe Data Pipelines", "type": "Link", "link_type": "Workspace"},
        {"label": "Document Chunks", "link_to": "Document Chunk", "type": "Link", "link_type": "DocType"},
        {"label": "Embedding Jobs", "link_to": "Embedding Job", "type": "Link", "link_type": "DocType"},
        {"label": "Data Pipeline Settings", "link_to": "Data Pipeline Settings", "type": "Link", "link_type": "DocType"},
    ]

    # Check if items already exist (by comparing labels)
    existing_labels = {item.label for item in sidebar.items}
    expected_labels = {item["label"] for item in expected_items}

    # If all expected items exist, no need to update
    if expected_labels.issubset(existing_labels):
        return

    # Clear and re-add items in correct order
    sidebar.items = []
    for item in expected_items:
        sidebar.append("items", item)

    sidebar.save()
    frappe.db.commit()


def setup_desktop_icon():
    """
    Configure the Desktop Icon for Data Pipelines on the desk page.

    This creates/updates the Desktop Icon to show Data Pipelines with
    a custom icon on the Frappe desk home page, grouped under Framework.
    """
    icon_name = "Frappe Data Pipelines"

    # Check if the icon exists
    if not frappe.db.exists("Desktop Icon", icon_name):
        # Create new Desktop Icon
        icon = frappe.get_doc({
            "doctype": "Desktop Icon",
            "name": icon_name,
            "label": "Data Pipelines",
            "icon_type": "App",
            "link_type": "External",
            "link": "/app/data-pipeline-settings",
            "standard": 1,
            "app": "frappe_data_pipelines",
            "logo_url": "/assets/frappe_data_pipelines/icons/data-pipelines.svg",
            "hidden": 0,
            "parent_icon": "Framework",
            "idx": 11,
        })
        icon.insert(ignore_permissions=True)
        frappe.logger("frappe_data_pipelines").info(
            f"Created Desktop Icon: {icon_name}"
        )
    else:
        # Update existing icon
        frappe.db.set_value("Desktop Icon", icon_name, {
            "label": "Data Pipelines",
            "icon_type": "App",
            "link_type": "External",
            "link": "/app/data-pipeline-settings",
            "logo_url": "/assets/frappe_data_pipelines/icons/data-pipelines.svg",
            "hidden": 0,
            "parent_icon": "Framework",
            "idx": 11,
        })
        frappe.logger("frappe_data_pipelines").info(
            f"Updated Desktop Icon: {icon_name}"
        )

    # Clear desktop icons cache
    frappe.cache.delete_keys("desktop_icons*")
    frappe.db.commit()
