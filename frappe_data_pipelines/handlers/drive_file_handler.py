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
    logger = frappe.logger("frappe_data_pipelines")
    logger.debug(f"on_file_upload triggered for: {doc.name} ({getattr(doc, 'title', 'untitled')})")

    # Get settings using robust utility function
    from frappe_data_pipelines.utils import get_or_create_settings

    settings = get_or_create_settings()
    if not settings:
        logger.warning(
            f"Skipping file {doc.name}: Data Pipeline Settings not available. "
            "Run 'bench migrate' to initialize."
        )
        return

    # Check if auto-processing is enabled
    if not settings.enable_auto_processing:
        logger.debug(f"Skipping file {doc.name}: auto-processing disabled")
        return

    # Check if file is a folder (is_group)
    if getattr(doc, "is_group", False):
        logger.debug(f"Skipping {doc.name}: is a folder")
        return

    # Import here to avoid circular imports
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    # Get file path
    file_path = getattr(doc, "path", None)

    # Check if file type is supported
    if not file_path:
        logger.debug(f"Skipping {doc.name}: no file path")
        return

    if not TextExtractor.is_supported(file_path):
        logger.debug(f"Skipping {doc.name}: unsupported file type ({file_path})")
        return

    # Check file size limit
    max_size = (settings.max_file_size_mb or 50) * 1024 * 1024
    file_size = getattr(doc, "file_size", 0) or 0
    if file_size > max_size:
        frappe.log_error(
            title="File too large for embedding",
            message=f"File {doc.title} ({file_size} bytes) exceeds max size ({max_size} bytes)"
        )
        logger.warning(f"Skipping {doc.name}: file too large ({file_size} > {max_size})")
        return

    # Create embedding job
    try:
        job = frappe.get_doc({
            "doctype": "Embedding Job",
            "source_drive_file": doc.name,
            "file_path": file_path,
            "file_title": getattr(doc, "title", None),
            "file_size_bytes": file_size,
            "file_mime_type": getattr(doc, "mime_type", None),
            "team": getattr(doc, "team", None),
            "status": "Queued",
            "priority": "Normal"
        })
        job.insert(ignore_permissions=True)
        frappe.db.commit()

        logger.info(f"Created embedding job {job.name} for file {doc.title}")

        # Enqueue background job
        frappe.enqueue(
            "frappe_data_pipelines.tasks.process_embedding.process_embedding_job",
            queue="long",
            embedding_job_name=job.name,
            enqueue_after_commit=True
        )
        logger.debug(f"Enqueued processing job for {job.name}")

    except Exception as e:
        frappe.log_error(
            title="Failed to create embedding job",
            message=f"Error creating job for {doc.name}: {str(e)}"
        )
        logger.error(f"Failed to create embedding job for {doc.name}: {e}")


def on_file_delete(doc, method):
    """
    Handler for Drive File on_trash event.
    Removes associated vectors from Qdrant and cleans up chunks.
    """
    logger = frappe.logger("frappe_data_pipelines")
    logger.debug(f"on_file_delete triggered for: {doc.name}")

    from frappe_data_pipelines.services.qdrant_service import QdrantService

    # Get all chunks for this document
    chunks = frappe.get_all(
        "Document Chunk",
        filters={"source_drive_file": doc.name},
        fields=["name", "collection_name"],
        ignore_permissions=True
    )

    if not chunks:
        logger.debug(f"No chunks found for {doc.name}")
        return

    logger.info(f"Cleaning up {len(chunks)} chunks for deleted file {doc.name}")

    # Delete vectors from Qdrant
    collection_name = chunks[0].collection_name if chunks else None
    if collection_name:
        try:
            QdrantService.delete_by_document(doc.name, collection_name)
            logger.debug(f"Deleted vectors from Qdrant for {doc.name}")
        except Exception as e:
            frappe.log_error(
                title="Failed to delete vectors",
                message=f"Error deleting vectors for {doc.name}: {str(e)}"
            )
            logger.error(f"Failed to delete vectors for {doc.name}: {e}")

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
    logger.debug(f"Cleanup complete for {doc.name}")
