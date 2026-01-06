"""
Event handlers for Drive File document events.
"""
import frappe
from frappe import _


def on_file_upload(doc, method):
    """
    Handler for Drive File after_insert event.
    Queues the file for embedding if it's a supported type.
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")
    except Exception:
        # Settings don't exist yet, skip processing
        return

    # Check if auto-processing is enabled
    if not settings.enable_auto_processing:
        return

    # Check if file is a folder (is_group)
    if getattr(doc, "is_group", False):
        return

    # Import here to avoid circular imports
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    # Get file path
    file_path = getattr(doc, "path", None)

    # Check if file type is supported
    if not file_path or not TextExtractor.is_supported(file_path):
        return

    # Check file size limit
    max_size = (settings.max_file_size_mb or 50) * 1024 * 1024
    file_size = getattr(doc, "file_size", 0) or 0
    if file_size > max_size:
        frappe.log_error(
            title="File too large for embedding",
            message=f"File {doc.title} ({file_size} bytes) exceeds max size ({max_size} bytes)"
        )
        return

    # Create embedding job
    job = frappe.get_doc({
        "doctype": "Embedding Job",
        "source_drive_file": doc.name,
        "file_path": file_path,
        "file_size_bytes": file_size,
        "file_mime_type": getattr(doc, "mime_type", None),
        "team": getattr(doc, "team", None),
        "status": "Queued",
        "priority": "Normal"
    })
    job.insert(ignore_permissions=True)
    frappe.db.commit()

    # Enqueue background job
    frappe.enqueue(
        "frappe_data_pipelines.tasks.process_embedding.process_embedding_job",
        queue="long",
        embedding_job_name=job.name,
        enqueue_after_commit=True
    )


def on_file_delete(doc, method):
    """
    Handler for Drive File on_trash event.
    Removes associated vectors from Qdrant and cleans up chunks.
    """
    from frappe_data_pipelines.services.qdrant_service import QdrantService

    # Get all chunks for this document
    chunks = frappe.get_all(
        "Document Chunk",
        filters={"source_drive_file": doc.name},
        fields=["name", "collection_name"],
        ignore_permissions=True
    )

    if not chunks:
        return

    # Delete vectors from Qdrant
    collection_name = chunks[0].collection_name if chunks else None
    if collection_name:
        try:
            QdrantService.delete_by_document(doc.name, collection_name)
        except Exception as e:
            frappe.log_error(
                title="Failed to delete vectors",
                message=f"Error deleting vectors for {doc.name}: {str(e)}"
            )

    # Delete chunk records
    for chunk in chunks:
        frappe.delete_doc("Document Chunk", chunk.name, ignore_permissions=True)

    # Delete related embedding jobs
    jobs = frappe.get_all(
        "Embedding Job",
        filters={"source_drive_file": doc.name},
        ignore_permissions=True
    )
    for job in jobs:
        frappe.delete_doc("Embedding Job", job.name, ignore_permissions=True)

    frappe.db.commit()
