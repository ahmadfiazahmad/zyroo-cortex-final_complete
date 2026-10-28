"""Shared FastAPI dependencies — auth, admin, workspaces, quotas."""
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import User, Workspace, WorkspaceMember
from .plans import check_quota, get_plan
from .security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """JWT -> User. 401 for bad/expired tokens, 403 for suspended accounts."""
    try:
        user_id = decode_access_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if user.is_suspended:
        raise HTTPException(status_code=403, detail="Account is suspended")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Admin-only dependency: 403 for non-admins."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def get_workspace(db: Session, workspace_id: str) -> Workspace:
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def is_workspace_member(db: Session, workspace_id: str, user_id: str) -> bool:
    return (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
        is not None
    )


def verify_workspace_access(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Workspace:
    """Reusable dependency: the user must be a member of the workspace.

    Returns the workspace. 403 for non-members.
    """
    workspace = get_workspace(db, workspace_id)
    if not is_workspace_member(db, workspace_id, current_user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this workspace")
    return workspace


def enforce_personal_quota(db: Session, user: User, kind: str) -> None:
    """Block with 429 when the user's own plan quota is reached."""
    used = (
        user.documents_uploaded_count if kind == "document" else user.messages_sent_count
    )
    check_quota(kind, used, user.plan_tier)


def enforce_workspace_quota(workspace: Workspace, kind: str) -> None:
    """Block with 429 when the workspace's shared Team quota is reached."""
    used = (
        workspace.documents_uploaded_count
        if kind == "document"
        else workspace.messages_sent_count
    )
    check_quota(kind, used, workspace.plan_tier or "team")


def require_team_plan(user: User) -> None:
    """Creating/operating a shared workspace needs an active Team plan."""
    if (user.plan_tier or "free").lower() != "team":
        raise HTTPException(
            status_code=402,
            detail=(
                "A Team plan is required to create a workspace. "
                "Upgrade here: GET /api/billing/plans"
            ),
        )
