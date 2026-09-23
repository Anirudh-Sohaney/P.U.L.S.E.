"""Small persistent account store and bearer-token authentication."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def _database() -> Path:
    path = Path(settings.DATA_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path / "accounts.sqlite3"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database())
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)")
    return connection


def _user(username: str):
    with _connect() as connection:
        return connection.execute("SELECT username, password_hash FROM accounts WHERE username = ?", (username,)).fetchone()


def _verify(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


class User(BaseModel):
    username: str
    role: str = "user"


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)


class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if not username:
            raise unauthorized
    except JWTError:
        raise unauthorized
    if _user(username) is None:
        raise unauthorized
    return User(username=username)


@router.post("/register", response_model=Token, status_code=201)
async def register(account: Registration):
    if _user(account.username):
        raise HTTPException(status_code=409, detail="That username is already registered")
    hashed = bcrypt.hashpw(account.password.encode(), bcrypt.gensalt()).decode()
    try:
        with _connect() as connection:
            connection.execute("INSERT INTO accounts(username, password_hash, created_at) VALUES (?, ?, ?)", (account.username, hashed, datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already registered")
    return Token(access_token=create_access_token({"sub": account.username}), token_type="bearer")


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = _user(form_data.username)
    if user is None or not _verify(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_access_token({"sub": user["username"]}), token_type="bearer")


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
