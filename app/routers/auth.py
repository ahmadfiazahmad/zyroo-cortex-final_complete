"""Auth — Week 2: register, login (JWT), me."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import LoginRequest, RegisterRequest, TokenOut, UserOut
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/register", response_model=TokenOut, summary="Register a new user")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    """Create a user (bcrypt-hashed password) and return a JWT."""
    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    role = "admin" if data.email.lower() in settings.admin_email_set else "user"
    user = User(
        name=data.name,
        email=data.email.lower(),
        password_hash=hash_password(data.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut, summary="Log in and get a JWT")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Verify credentials and return a JWT. Generic error on failure."""
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user.is_suspended:
        raise HTTPException(status_code=403, detail="Account is suspended")
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut, summary="Get the logged-in user")
def me(current_user: User = Depends(get_current_user)):
    """Protected route — returns the user behind the JWT."""
    return current_user
