"""
Background task for processing embedding jobs.
Handles text extraction, chunking, embedding, and vector storage.
"""
import frappe
from frappe import _
from typing import Optional
import traceback


def process_embedding_job(job_name: str):
    """
    Main background task to process an embedding job.

    Steps:
    1. Extract text from file
    2. Chunk text
    3. Generate embeddings (batched)
    4. Store vectors in Qdrant
    5. Create Document Chunk records
    """
    job = frappe.get_doc("Embedding Job", job_name)

    if job.status not in ("Queued", "Failed"):
        return

    try:
        # Update status to processing
        job.status = "Extracting"
        job.started_at = frappe.utils.now_datetime()
        job.save(ignore_permissions=True)
        frappe.db.commit()

        # Get settings
        settings = frappe.get_single("Data Pipeline Settings")

        # Step 1: Extract text
        text = extract_text(job)
        if not text:
            raise ValueError("No text could be extracted from file")

        # Step 2: Chunk text
        job.status = "Chunking"
        job.save(ignore_permissions=True)
        frappe.db.commit()

        chunks = chunk_text(text, settings)
        job.total_chunks = len(chunks)
        job.save(ignore_permissions=True)
        frappe.db.commit()

        if not chunks:
            raise ValueError("No chunks generated from text")

        # Step 3: Generate embeddings
        job.status = "Embedding"
        job.save(ignore_permissions=True)
        frappe.db.commit()

        embeddings = generate_embeddings(chunks, settings, job)

        # Step 4: Store in Qdrant
        job.status = "Storing"
        job.save(ignore_permissions=True)
        frappe.db.commit()

        point_ids = store_vectors(job, chunks, embeddings, settings)

        # Step 5: Create Document Chunk records
        create_chunk_records(job, chunks, point_ids, settings)

        # Mark complete
        job.status = "Completed"
        job.completed_at = frappe.utils.now_datetime()
        job.processed_chunks = len(chunks)
        job.progress_percent = 100
        job.save(ignore_permissions=True)
        frappe.db.commit()

    except Exception as e:
        job.reload()
        job.status = "Failed"
        job.error_message = str(e)[:500]
        job.retry_count = (job.retry_count or 0) + 1
        job.save(ignore_permissions=True)
        frappe.db.commit()

        frappe.log_error(
            title=f"Embedding Job Failed: {job_name}",
            message=traceback.format_exc()
        )


def extract_text(job) -> str:
    """Extract text from the source file."""
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    # Get the Drive File
    drive_file = frappe.get_doc("Drive File", job.source_drive_file)

    # Get file path - Drive stores files in private/files
    file_path = drive_file.path
    if not file_path:
        raise ValueError("Drive File has no path")

    # Construct full path
    site_path = frappe.get_site_path()

    # Drive files are typically stored relative to site
    if not file_path.startswith("/"):
        full_path = f"{site_path}/{file_path}"
    else:
        full_path = file_path

    # Extract text
    text = TextExtractor.extract(full_path)
    return text


def chunk_text(text: str, settings) -> list:
    """Chunk the extracted text."""
    from frappe_data_pipelines.services.chunking_service import ChunkingService

    chunk_size = settings.chunk_size or 1000
    chunk_overlap = settings.chunk_overlap or 200

    chunks = ChunkingService.chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunks


def generate_embeddings(chunks: list, settings, job) -> list:
    """Generate embeddings for chunks in batches."""
    from frappe_data_pipelines.services.embedding_service import get_embedding_provider

    provider = get_embedding_provider()
    batch_size = 50
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = provider.embed(batch)
        all_embeddings.extend(batch_embeddings)

        # Update progress
        job.processed_chunks = min(i + batch_size, len(chunks))
        job.progress_percent = int((job.processed_chunks / len(chunks)) * 80)  # 80% for embedding
        job.save(ignore_permissions=True)
        frappe.db.commit()

    return all_embeddings


def store_vectors(job, chunks: list, embeddings: list, settings) -> list:
    """Store vectors in Qdrant."""
    from frappe_data_pipelines.services.qdrant_service import QdrantService

    collection_name = settings.collection_name or "drive_documents"

    # Get Drive File info for payloads
    drive_file = frappe.get_doc("Drive File", job.source_drive_file)

    # Create payloads with metadata
    payloads = []
    for i, chunk in enumerate(chunks):
        payloads.append({
            "source_document": job.source_drive_file,
            "source_title": drive_file.title or drive_file.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_text": chunk[:1000],  # Store first 1000 chars for preview
            "team": job.team or "",
            "mime_type": job.file_mime_type or "",
            "owner": drive_file.owner
        })

    # Upsert vectors
    point_ids = QdrantService.upsert_vectors(
        vectors=embeddings,
        payloads=payloads,
        collection_name=collection_name
    )

    return point_ids


def create_chunk_records(job, chunks: list, point_ids: list, settings):
    """Create Document Chunk records for tracking."""
    collection_name = settings.collection_name or "drive_documents"

    for i, (chunk, point_id) in enumerate(zip(chunks, point_ids)):
        chunk_doc = frappe.get_doc({
            "doctype": "Document Chunk",
            "source_drive_file": job.source_drive_file,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_text": chunk,
            "qdrant_point_id": point_id,
            "collection_name": collection_name,
            "embedding_status": "Completed",
            "team": job.team
        })
        chunk_doc.insert(ignore_permissions=True)

    frappe.db.commit()


def retry_failed_jobs():
    """
    Scheduler task to retry failed jobs.
    Runs hourly, retries jobs with retry_count < 3.
    """
    failed_jobs = frappe.get_all(
        "Embedding Job",
        filters={
            "status": "Failed",
            "retry_count": ["<", 3]
        },
        pluck="name"
    )

    for job_name in failed_jobs:
        # Reset status for retry
        frappe.db.set_value("Embedding Job", job_name, "status", "Queued")

        # Re-enqueue
        frappe.enqueue(
            "frappe_data_pipelines.tasks.process_embedding.process_embedding_job",
            queue="long",
            job_name=job_name
        )

    if failed_jobs:
        frappe.db.commit()


def cleanup_old_jobs():
    """
    Scheduler task to clean up old completed jobs.
    Runs daily, removes jobs older than 30 days.
    """
    from frappe.utils import add_days, nowdate

    cutoff_date = add_days(nowdate(), -30)

    old_jobs = frappe.get_all(
        "Embedding Job",
        filters={
            "status": "Completed",
            "completed_at": ["<", cutoff_date]
        },
        pluck="name"
    )

    for job_name in old_jobs:
        frappe.delete_doc("Embedding Job", job_name, ignore_permissions=True)

    if old_jobs:
        frappe.db.commit()
        frappe.log_error(
            title="Embedding Jobs Cleanup",
            message=f"Cleaned up {len(old_jobs)} old embedding jobs"
        )
