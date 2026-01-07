"""
Background task for processing embedding jobs.

Supports both legacy pipeline and Smart Pipeline v2 with:
- Decision agent for intelligent routing
- Semantic chunking
- Contextual enrichment
- Vision/OCR processing
"""
import frappe
from frappe import _
from typing import Optional, List
import traceback


def process_embedding_job(embedding_job_name: str):
    """
    Main background task to process an embedding job.

    Smart Pipeline Steps (when enabled):
    1. Analyze file with decision agent
    2. Extract text (with OCR/vision if needed)
    3. Semantic chunking with structure detection
    4. Contextual enrichment (situating context)
    5. Generate embeddings
    6. Store vectors in Qdrant
    7. Create Document Chunk records

    Legacy Steps (when smart pipeline disabled):
    1. Extract text from file
    2. Chunk text
    3. Generate embeddings
    4. Store vectors in Qdrant
    5. Create Document Chunk records
    """
    job = frappe.get_doc("Embedding Job", embedding_job_name)

    if job.status not in ("Queued", "Failed"):
        return

    try:
        # Update status to processing
        job.status = "Extracting Text"
        job.started_at = frappe.utils.now_datetime()
        job.save(ignore_permissions=True)
        frappe.db.commit()

        # Get settings
        settings = frappe.get_single("Data Pipeline Settings")

        # Determine if we should use smart pipeline
        use_smart_pipeline = getattr(settings, 'enable_smart_pipeline', False)

        if use_smart_pipeline:
            # Smart Pipeline v2
            process_smart_pipeline(job, settings)
        else:
            # Legacy pipeline
            process_legacy_pipeline(job, settings)

        # Mark complete
        job.reload()
        job.status = "Completed"
        job.completed_at = frappe.utils.now_datetime()
        job.progress_percent = 100
        job.save(ignore_permissions=True)
        frappe.db.commit()

    except Exception as e:
        job.reload()
        job.status = "Failed"
        job.error_message = str(e)[:500]
        job.error_traceback = traceback.format_exc()[:2000]
        job.retry_count = (job.retry_count or 0) + 1
        job.save(ignore_permissions=True)
        frappe.db.commit()

        frappe.log_error(
            title=f"Embedding Job Failed: {embedding_job_name}",
            message=traceback.format_exc()
        )


def process_smart_pipeline(job, settings):
    """
    Process using Smart Pipeline v2 with Docling-based chunking.
    """
    from frappe_data_pipelines.services.decision_agent import DecisionAgent
    from frappe_data_pipelines.services.chunking_service import DoclingChunker, SemanticChunker, get_chunker
    from frappe_data_pipelines.services.context_service import get_context_service
    from frappe_data_pipelines.services.embedding_service import get_embedding_provider

    file_path = get_full_file_path(job)

    # Step 1: Analyze with decision agent
    agent = DecisionAgent()
    plan = agent.analyze(
        file_path=file_path,
        mime_type=job.file_mime_type
    )

    # Step 2: Extract text and chunk
    job.status = "Extracting Text"
    job.save(ignore_permissions=True)
    frappe.db.commit()

    chunk_texts = []
    section_paths = []
    text = ""

    # Handle images specially - no chunking needed, vision description is a single unit
    if plan.content_type == 'visual' and plan.document_type == 'image':
        # Extract image description using vision service
        text, visual_chunks = extract_text_smart(job, plan, settings)
        if text:
            # Image description is a single chunk - don't split it
            chunk_texts = [text]
            section_paths = ["Image Description"]
        elif visual_chunks:
            # Use visual chunks directly
            for vc in visual_chunks:
                chunk_texts.append(vc.combined)
                section_paths.append("Visual Content")

        if not chunk_texts:
            raise ValueError("No content could be extracted from image")
    else:
        # For documents: use Docling or semantic chunking
        chunker = get_chunker(use_semantic=settings.enable_semantic_chunking, use_docling=True)

        if isinstance(chunker, DoclingChunker):
            # Use Docling for direct document processing (best quality)
            job.status = "Chunking"
            job.save(ignore_permissions=True)
            frappe.db.commit()

            try:
                semantic_chunks = chunker.chunk_document(file_path)
                chunk_texts = [c.text for c in semantic_chunks]
                section_paths = [c.section_path for c in semantic_chunks]
                text = "\n\n".join(chunk_texts)  # Reconstruct for context enrichment
            except Exception as e:
                frappe.log_error(
                    title="Docling document chunking failed",
                    message=f"Falling back to text extraction: {str(e)}"
                )
                # Fall back to text extraction + chunking
                text, visual_chunks = extract_text_smart(job, plan, settings)
                if text:
                    semantic_chunks = chunker.chunk_text(text, plan.document_type)
                    chunk_texts = [c.text for c in semantic_chunks]
                    section_paths = [c.section_path for c in semantic_chunks]
        else:
            # Use legacy extraction + chunking
            text, visual_chunks = extract_text_smart(job, plan, settings)
            if not text and not visual_chunks:
                raise ValueError("No content could be extracted from file")

            job.status = "Chunking"
            job.save(ignore_permissions=True)
            frappe.db.commit()

            if isinstance(chunker, SemanticChunker):
                semantic_chunks = chunker.chunk(text, plan.document_type)
                chunk_texts = [c.text for c in semantic_chunks]
                section_paths = [c.section_path for c in semantic_chunks]
            else:
                chunk_texts = chunker.chunk_text(text)
                section_paths = ["Document"] * len(chunk_texts)

            # Add visual content chunks if any
            visual_info = []
            if visual_chunks:
                for vc in visual_chunks:
                    chunk_texts.append(vc.combined)
                    section_paths.append("Visual Content")
                    visual_info.append(True)
                visual_info = [False] * (len(chunk_texts) - len(visual_chunks)) + visual_info

    if not chunk_texts:
        raise ValueError("No chunks generated from content")

    job.total_chunks = len(chunk_texts)
    job.save(ignore_permissions=True)
    frappe.db.commit()

    # Step 4: Contextual enrichment (if enabled)
    # Skip enrichment for images - vision description already provides context
    context_prefixes = [""] * len(chunk_texts)
    is_image = plan.content_type == 'visual' and plan.document_type == 'image'
    if settings.enable_contextual_enrichment and not is_image:
        context_service = get_context_service()
        if context_service:
            job.status = "Enriching Context"
            job.save(ignore_permissions=True)
            frappe.db.commit()

            # Get document title
            doc_title = frappe.db.get_value(
                "Drive File", job.source_drive_file, "title"
            ) or job.source_drive_file

            try:
                enriched = context_service.enrich_chunks(
                    chunks=chunk_texts,
                    full_document=text[:50000],  # Truncate for context
                    document_title=doc_title,
                    section_paths=section_paths
                )

                # Update chunks with enriched text for embedding
                for i, ec in enumerate(enriched):
                    chunk_texts[i] = ec.embedded_text
                    context_prefixes[i] = ec.context_prefix

            except Exception as e:
                frappe.log_error(
                    title="Contextual enrichment failed",
                    message=str(e)
                )
                # Continue without enrichment

    # Step 5: Generate embeddings
    job.status = "Embedding"
    job.save(ignore_permissions=True)
    frappe.db.commit()

    embeddings = generate_embeddings(chunk_texts, settings, job)

    # Step 6: Store in Qdrant
    job.status = "Storing Vectors"
    job.save(ignore_permissions=True)
    frappe.db.commit()

    point_ids = store_vectors_smart(
        job, chunk_texts, embeddings, settings,
        section_paths=section_paths,
        context_prefixes=context_prefixes,
        processing_strategy=plan.strategy,
        detected_languages=plan.detected_languages
    )

    # Step 7: Create Document Chunk records
    create_chunk_records_smart(
        job, chunk_texts, point_ids, settings,
        section_paths=section_paths,
        context_prefixes=context_prefixes,
        visual_flags=visual_info if visual_chunks else None,
        processing_strategy=plan.strategy,
        detected_languages=plan.detected_languages
    )


def process_legacy_pipeline(job, settings):
    """
    Process using legacy pipeline (backward compatible).
    """
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
    job.status = "Storing Vectors"
    job.save(ignore_permissions=True)
    frappe.db.commit()

    point_ids = store_vectors(job, chunks, embeddings, settings)

    # Step 5: Create Document Chunk records
    create_chunk_records(job, chunks, point_ids, settings)

    job.processed_chunks = len(chunks)


def get_full_file_path(job) -> str:
    """Get the full file path for a job."""
    file_path = frappe.db.get_value("Drive File", job.source_drive_file, "path")
    if not file_path:
        raise ValueError("Drive File has no path")
    site_path = frappe.get_site_path()
    return f"{site_path}/private/files/{file_path}"


def extract_text_smart(job, plan, settings) -> tuple:
    """
    Extract text using smart pipeline with OCR/vision if needed.

    Returns:
        Tuple of (text, visual_chunks)
    """
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    full_path = get_full_file_path(job)
    visual_chunks = []

    # Standard text extraction
    text = TextExtractor.extract(full_path)

    # If OCR is needed and enabled
    if plan.requires_ocr and settings.enable_vision_processing:
        try:
            from frappe_data_pipelines.services.ocr_service import get_ocr_service
            ocr_service = get_ocr_service(plan.detected_languages)
            ocr_result = ocr_service.extract_text(full_path)
            if ocr_result.text:
                text = ocr_result.text if not text else f"{text}\n\n{ocr_result.text}"
        except Exception as e:
            frappe.log_error(title="OCR extraction failed", message=str(e))

    # If vision processing is needed and enabled
    if plan.requires_vision and settings.enable_vision_processing:
        try:
            from frappe_data_pipelines.services.vision_service import get_vision_service
            vision_service = get_vision_service()
            if vision_service:
                visual_content = vision_service.process_image(full_path)
                if visual_content and visual_content.combined:
                    visual_chunks.append(visual_content)
        except Exception as e:
            frappe.log_error(title="Vision processing failed", message=str(e))

    return text, visual_chunks


def extract_text(job) -> str:
    """Extract text from the source file (legacy)."""
    from frappe_data_pipelines.services.text_extraction import TextExtractor

    full_path = get_full_file_path(job)
    text = TextExtractor.extract(full_path)
    return text


def chunk_text(text: str, settings) -> list:
    """Chunk the extracted text (legacy)."""
    from frappe_data_pipelines.services.chunking_service import ChunkingService

    chunks = ChunkingService.chunk_text(text)
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
        job.progress_percent = int((job.processed_chunks / len(chunks)) * 80)
        job.save(ignore_permissions=True)
        frappe.db.commit()

    return all_embeddings


def store_vectors(job, chunks: list, embeddings: list, settings) -> list:
    """Store vectors in Qdrant (legacy)."""
    from frappe_data_pipelines.services.qdrant_service import QdrantService

    collection_name = settings.collection_name or "drive_documents"

    drive_file_info = frappe.db.get_value(
        "Drive File",
        job.source_drive_file,
        ["title", "name", "owner"],
        as_dict=True
    )

    payloads = []
    for i, chunk in enumerate(chunks):
        payloads.append({
            "source_drive_file": job.source_drive_file,
            "source_document": job.source_drive_file,
            "source_title": drive_file_info.title or drive_file_info.name if drive_file_info else job.source_drive_file,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk[:1000],
            "original_text": chunk,
            "team": job.team or "",
            "mime_type": job.file_mime_type or "",
            "owner": drive_file_info.owner if drive_file_info else ""
        })

    point_ids = QdrantService.upsert_vectors(
        vectors=embeddings,
        payloads=payloads,
        collection_name=collection_name
    )

    return point_ids


def store_vectors_smart(
    job, chunks: list, embeddings: list, settings,
    section_paths: list = None,
    context_prefixes: list = None,
    processing_strategy: str = "text",
    detected_languages: list = None
) -> list:
    """Store vectors in Qdrant with smart pipeline metadata."""
    from frappe_data_pipelines.services.qdrant_service import QdrantService

    collection_name = settings.collection_name or "drive_documents"

    drive_file_info = frappe.db.get_value(
        "Drive File",
        job.source_drive_file,
        ["title", "name", "owner"],
        as_dict=True
    )

    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "source_drive_file": job.source_drive_file,
            "source_document": job.source_drive_file,
            "source_title": drive_file_info.title or drive_file_info.name if drive_file_info else job.source_drive_file,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk[:1000],
            "original_text": chunk,
            "team": job.team or "",
            "mime_type": job.file_mime_type or "",
            "owner": drive_file_info.owner if drive_file_info else "",
            # Smart pipeline fields
            "section_path": section_paths[i] if section_paths else "Document",
            "context_prefix": context_prefixes[i] if context_prefixes else "",
            "processing_strategy": processing_strategy,
            "detected_languages": ",".join(detected_languages) if detected_languages else ""
        }
        payloads.append(payload)

    point_ids = QdrantService.upsert_vectors(
        vectors=embeddings,
        payloads=payloads,
        collection_name=collection_name
    )

    return point_ids


def create_chunk_records(job, chunks: list, point_ids: list, settings):
    """Create Document Chunk records for tracking (legacy)."""
    collection_name = settings.collection_name or "drive_documents"

    for i, (chunk, point_id) in enumerate(zip(chunks, point_ids)):
        chunk_doc = frappe.get_doc({
            "doctype": "Document Chunk",
            "source_drive_file": job.source_drive_file,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_text": chunk,
            "character_count": len(chunk),
            "qdrant_point_id": point_id,
            "collection_name": collection_name,
            "embedding_status": "Completed",
            "team": job.team,
            "file_mime_type": job.file_mime_type
        })
        chunk_doc.insert(ignore_permissions=True)

    frappe.db.commit()


def create_chunk_records_smart(
    job, chunks: list, point_ids: list, settings,
    section_paths: list = None,
    context_prefixes: list = None,
    visual_flags: list = None,
    processing_strategy: str = "text",
    detected_languages: list = None
):
    """Create Document Chunk records with smart pipeline metadata."""
    collection_name = settings.collection_name or "drive_documents"

    for i, (chunk, point_id) in enumerate(zip(chunks, point_ids)):
        chunk_doc = frappe.get_doc({
            "doctype": "Document Chunk",
            "source_drive_file": job.source_drive_file,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_text": chunk,
            "character_count": len(chunk),
            "qdrant_point_id": point_id,
            "collection_name": collection_name,
            "embedding_status": "Completed",
            "team": job.team,
            "file_mime_type": job.file_mime_type,
            # Smart pipeline fields
            "section_path": section_paths[i] if section_paths else None,
            "context_prefix": context_prefixes[i] if context_prefixes else None,
            "has_visual_content": visual_flags[i] if visual_flags else False,
            "processing_strategy": processing_strategy,
            "detected_languages": ",".join(detected_languages) if detected_languages else None
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
            embedding_job_name=job_name
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
