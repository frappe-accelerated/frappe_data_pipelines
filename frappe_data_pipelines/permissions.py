"""
Row-Level Security (RLS) permissions for Document Chunks.
Document Chunks inherit permissions from their source Drive File.
"""
import frappe


def document_chunk_has_permission(doc, user=None, permission_type=None):
    """
    Custom permission logic for Document Chunks.

    Users can only access chunks if they have permission to access
    the source Drive File.
    """
    if not user:
        user = frappe.session.user

    # Admins and System Managers bypass checks
    roles = frappe.get_roles(user)
    if "Administrator" in roles or "System Manager" in roles:
        return True

    # Get source Drive File
    source_file = getattr(doc, "source_drive_file", None)
    if not source_file:
        return False

    # Check permission on source Drive File
    try:
        # Use Frappe's permission system to check Drive File access
        return frappe.has_permission(
            doctype="Drive File",
            doc=source_file,
            ptype=permission_type or "read",
            user=user
        )
    except frappe.DoesNotExistError:
        return False
    except Exception:
        return False


def document_chunk_query_conditions(user):
    """
    Filter Document Chunks in list queries based on Drive File access.

    Only returns chunks from Drive Files the user can access.
    This improves performance by filtering at query time.
    """
    if not user:
        user = frappe.session.user

    # Admins and System Managers see all chunks
    roles = frappe.get_roles(user)
    if "Administrator" in roles or "System Manager" in roles:
        return ""

    # Build SQL condition for accessible Drive Files
    # Users can see chunks if they:
    # 1. Own the Drive File
    # 2. Have explicit sharing via Drive DocShare
    # 3. Are members of a team that has access

    escaped_user = frappe.db.escape(user)

    condition = f"""
        `tabDocument Chunk`.source_drive_file IN (
            SELECT name FROM `tabDrive File`
            WHERE owner = {escaped_user}

            UNION

            SELECT entity FROM `tabDrive DocShare`
            WHERE user = {escaped_user}
            AND entity_type = 'Drive File'
        )
    """

    return condition
