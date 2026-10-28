"""Team workspaces — Weeks 7-8: create, invite, remove, shared lists."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import (
    get_current_user,
    get_workspace,
    is_workspace_member,
    require_team_plan,
    verify_workspace_access,
)
from ..models import ChatSession, Document, User, Workspace, WorkspaceMember
from ..schemas import (
    CreateWorkspaceRequest,
    DocumentOut,
    InviteRequest,
    SessionOut,
    WorkspaceMemberOut,
    WorkspaceOut,
)

router = APIRouter(prefix="/api/workspaces", tags=["Workspaces"])


def _workspace_out(db: Session, workspace: Workspace) -> WorkspaceOut:
    members = []
    for m in db.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace.id).all():
        u = db.query(User).filter(User.id == m.user_id).first()
        if u:
            members.append(
                WorkspaceMemberOut(
                    user_id=u.id, name=u.name, email=u.email, role=m.role, joined_at=m.joined_at
                )
            )
    return WorkspaceOut(
        id=workspace.id,
        name=workspace.name,
        owner_id=workspace.owner_id,
        plan_tier=workspace.plan_tier,
        documents_uploaded_count=workspace.documents_uploaded_count,
        messages_sent_count=workspace.messages_sent_count,
        storage_used_bytes=workspace.storage_used_bytes,
        created_at=workspace.created_at,
        members=members,
    )


@router.post("", response_model=WorkspaceOut, summary="Create a workspace (Team plan)")
def create_workspace(
    data: CreateWorkspaceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The creator is automatically added as owner. Requires a Team plan."""
    require_team_plan(current_user)
    workspace = Workspace(name=data.name, owner_id=current_user.id, plan_tier="team")
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=current_user.id, role="owner"))
    db.commit()
    return _workspace_out(db, workspace)


@router.get("", response_model=list[WorkspaceOut], summary="List my workspaces")
def list_workspaces(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    memberships = (
        db.query(WorkspaceMember).filter(WorkspaceMember.user_id == current_user.id).all()
    )
    workspaces = [
        db.query(Workspace).filter(Workspace.id == m.workspace_id).first() for m in memberships
    ]
    return [_workspace_out(db, w) for w in workspaces if w]


@router.get("/{workspace_id}", response_model=WorkspaceOut, summary="Get one workspace")
def get_one_workspace(
    workspace: Workspace = Depends(verify_workspace_access), db: Session = Depends(get_db)
):
    return _workspace_out(db, workspace)


@router.post("/{workspace_id}/invite", response_model=WorkspaceOut, summary="Invite a member by email")
def invite_member(
    workspace_id: str,
    data: InviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Only the workspace owner can invite. Prevents duplicate memberships."""
    workspace = get_workspace(db, workspace_id)
    if workspace.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the workspace owner can invite members")
    invited = db.query(User).filter(User.email == data.email.lower()).first()
    if not invited:
        raise HTTPException(status_code=404, detail="No user with that email")
    if is_workspace_member(db, workspace_id, invited.id):
        raise HTTPException(status_code=400, detail="User is already a member of this workspace")
    db.add(WorkspaceMember(workspace_id=workspace_id, user_id=invited.id, role="member"))
    db.commit()
    return _workspace_out(db, workspace)


@router.delete("/{workspace_id}/members/{user_id}", summary="Remove a member")
def remove_member(
    workspace_id: str,
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Only the owner can remove members, and the owner cannot remove
    themselves."""
    workspace = get_workspace(db, workspace_id)
    if workspace.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the workspace owner can remove members")
    if user_id == workspace.owner_id:
        raise HTTPException(status_code=400, detail="The workspace owner cannot be removed")
    member = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="User is not a member of this workspace")
    db.delete(member)
    db.commit()
    return {"message": "Member removed", "workspace_id": workspace_id, "user_id": user_id}


@router.get("/{workspace_id}/documents", response_model=list[DocumentOut], summary="List workspace documents")
def list_workspace_documents(
    workspace: Workspace = Depends(verify_workspace_access), db: Session = Depends(get_db)
):
    return (
        db.query(Document)
        .filter(Document.workspace_id == workspace.id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )


@router.get("/{workspace_id}/chat-sessions", response_model=list[SessionOut], summary="List workspace chat sessions")
def list_workspace_sessions(
    workspace: Workspace = Depends(verify_workspace_access), db: Session = Depends(get_db)
):
    return (
        db.query(ChatSession)
        .filter(ChatSession.workspace_id == workspace.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
