app_name = "frappe_data_pipelines"
app_title = "Frappe Data Pipelines"
app_publisher = "Frappe Accelerated"
app_description = "Document embedding and vector search for Frappe Drive with Insights integration"
app_email = "admin@frappe-accelerated.com"
app_license = "MIT"
app_icon = "octicon octicon-database"
app_color = "blue"

# Required apps
required_apps = ["frappe", "drive"]

# Setup hooks - initialize settings after install/migrate
after_install = "frappe_data_pipelines.setup.after_install.execute"
after_migrate = "frappe_data_pipelines.setup.after_install.execute"

# Document Events - Hook into Drive File uploads
doc_events = {
    "Drive File": {
        "after_insert": "frappe_data_pipelines.handlers.drive_file_handler.on_file_upload",
        "on_trash": "frappe_data_pipelines.handlers.drive_file_handler.on_file_delete",
    }
}

# RLS Permissions - Document Chunks inherit permissions from source Drive File
has_permission = {
    "Document Chunk": "frappe_data_pipelines.permissions.document_chunk_has_permission",
}

permission_query_conditions = {
    "Document Chunk": "frappe_data_pipelines.permissions.document_chunk_query_conditions",
}

# Scheduled Tasks
scheduler_events = {
    "hourly": [
        "frappe_data_pipelines.tasks.process_embedding.retry_failed_jobs"
    ],
    "daily": [
        "frappe_data_pipelines.tasks.process_embedding.cleanup_old_jobs"
    ],
}

# Fixtures
fixtures = [
    {"doctype": "Custom Field", "filters": [["module", "=", "Frappe Data Pipelines"]]},
]
