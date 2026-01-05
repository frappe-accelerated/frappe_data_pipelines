"""
Qdrant vector database service.
Handles collection management, vector upsert, and similarity search.
"""
import frappe
import uuid
from typing import List, Dict, Any, Optional


class QdrantService:
    """Wrapper for Qdrant operations."""

    _client = None

    @classmethod
    def get_client(cls):
        """Get or create Qdrant client based on settings."""
        from qdrant_client import QdrantClient

        if cls._client is not None:
            return cls._client

        settings = frappe.get_single("Data Pipeline Settings")

        if settings.qdrant_mode == "Embedded (In-Memory)":
            cls._client = QdrantClient(":memory:")
        elif settings.qdrant_mode == "Embedded (Persistent)":
            path = settings.qdrant_path or "./sites/{site_name}/private/qdrant"
            path = path.replace("{site_name}", frappe.local.site)
            # Ensure directory exists
            import os
            os.makedirs(path, exist_ok=True)
            cls._client = QdrantClient(path=path)
        else:  # Server mode
            api_key = None
            if settings.qdrant_api_key:
                api_key = settings.get_password("qdrant_api_key")
            cls._client = QdrantClient(
                host=settings.qdrant_host or "localhost",
                port=settings.qdrant_port or 6333,
                api_key=api_key
            )

        return cls._client

    @classmethod
    def reset_client(cls):
        """Reset the client (useful after settings change)."""
        cls._client = None

    @classmethod
    def get_collection_name(cls) -> str:
        """Get collection name from settings."""
        try:
            settings = frappe.get_single("Data Pipeline Settings")
            return settings.collection_name or "drive_documents"
        except Exception:
            return "drive_documents"

    @classmethod
    def ensure_collection(cls, collection_name: Optional[str] = None) -> None:
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        client = cls.get_client()
        collection_name = collection_name or cls.get_collection_name()

        try:
            settings = frappe.get_single("Data Pipeline Settings")
            dimension = settings.embedding_dimension or 384
        except Exception:
            dimension = 384

        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

    @classmethod
    def upsert_vectors(
        cls,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Upsert vectors with payloads into collection."""
        from qdrant_client.models import PointStruct

        collection_name = collection_name or cls.get_collection_name()
        cls.ensure_collection(collection_name)
        client = cls.get_client()

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )

        return ids

    @classmethod
    def search(
        cls,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection_name = collection_name or cls.get_collection_name()
        client = cls.get_client()

        search_filter = None
        if filters:
            must_conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            search_filter = Filter(must=must_conditions)

        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )

        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]

    @classmethod
    def delete_by_document(
        cls,
        source_document: str,
        collection_name: Optional[str] = None
    ) -> None:
        """Delete all vectors for a source document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection_name = collection_name or cls.get_collection_name()
        client = cls.get_client()

        try:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_document",
                            match=MatchValue(value=source_document)
                        )
                    ]
                )
            )
        except Exception as e:
            frappe.log_error(
                title="Qdrant Delete Error",
                message=f"Failed to delete vectors for {source_document}: {str(e)}"
            )

    @classmethod
    def get_collection_info(cls, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a collection."""
        collection_name = collection_name or cls.get_collection_name()
        client = cls.get_client()

        try:
            info = client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "config": {
                    "vector_size": info.config.params.vectors.size if info.config.params.vectors else None,
                    "distance": info.config.params.vectors.distance.value if info.config.params.vectors else None
                }
            }
        except Exception as e:
            return {
                "name": collection_name,
                "error": str(e)
            }

    @classmethod
    def get_all_collections(cls) -> List[Dict[str, Any]]:
        """Get list of all collections with their info."""
        client = cls.get_client()
        collections = client.get_collections().collections

        result = []
        for col in collections:
            info = cls.get_collection_info(col.name)
            result.append(info)

        return result

    @classmethod
    def test_connection(cls) -> Dict[str, Any]:
        """Test connection to Qdrant."""
        try:
            client = cls.get_client()
            collections = client.get_collections()
            return {
                "success": True,
                "collections_count": len(collections.collections),
                "message": "Connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e)
            }
