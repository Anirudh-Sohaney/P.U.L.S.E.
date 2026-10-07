"""Small persistent account store and bearer-token authentication."""
from __future__ import annotations

import sqlite3
import hashlib
import hmac
import ipaddress
import os
import secrets
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from jwt.exceptions import PyJWTError
from pydantic import BaseModel, Field

from ..config import settings
from ..database_locks import exclusive_file_lock

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)
PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=19 * 1024,
                                 parallelism=1, hash_len=32, salt_len=16,
                                 type=Type.ID)
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))
AUTH_SCHEMA_VERSION = 1


def _database() -> Path:
    path = Path(settings.DATA_PATH)
    path.mkdir(parents=True, exist_ok=True)
    database = path / "accounts.sqlite3"
    try:
        descriptor = os.open(database, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(descriptor)
    database.chmod(0o600)
    return database


def initialize() -> None:
    """Initialize and version the account database before serving requests."""
    connection = _connect()
    connection.close()


def _connect() -> sqlite3.Connection:
    path = _database()
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    # Switching journal modes on concurrent fresh connections can fail even
    # with a busy timeout; the WAL setting persists after the first switch.
    with exclusive_file_lock(path.with_name(path.name + ".journal.lock")):
        if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            connection.execute("PRAGMA journal_mode=WAL")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > AUTH_SCHEMA_VERSION:
        connection.close()
        raise RuntimeError(f"Account database schema {version} is newer than this service")
    if version < AUTH_SCHEMA_VERSION:
        connection.close()
        _migrate_database(path)
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _migrate_database(path: Path) -> None:
    """Apply the account schema once, serialized across API worker processes."""
    with exclusive_file_lock(path.with_name(path.name + ".schema.lock")):
        connection = sqlite3.connect(path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > AUTH_SCHEMA_VERSION:
                raise RuntimeError(f"Account database schema {version} is newer than this service")
            if version == AUTH_SCHEMA_VERSION:
                return
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)")
            connection.execute("""CREATE TABLE IF NOT EXISTS login_attempts
                (username TEXT NOT NULL, ip TEXT NOT NULL, failed_at TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts ON login_attempts(username, ip, failed_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip, failed_at)")
            connection.execute("""CREATE TABLE IF NOT EXISTS registration_attempts
                (ip TEXT NOT NULL, attempted_at TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_registration_attempts ON registration_attempts(ip, attempted_at)")
            connection.execute("""CREATE TABLE IF NOT EXISTS sessions
                (jti_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at TEXT NOT NULL,
                 created_at TEXT NOT NULL, FOREIGN KEY(username) REFERENCES accounts(username))""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username, expires_at)")
            connection.execute(f"PRAGMA user_version={AUTH_SCHEMA_VERSION}")
            connection.commit()
        finally:
            connection.close()


def _user(username: str):
    with _connect() as connection:
        return connection.execute("SELECT username, password_hash FROM accounts WHERE username = ?", (username,)).fetchone()


def _verify(password: str, hashed: str) -> bool:
    if hashed.startswith("$argon2id$"):
        try:
            return PASSWORD_HASHER.verify(hashed, password)
        except (VerificationError, InvalidHashError):
            return False
    try:
        if hashed.startswith("scrypt$"):
            _, n, r, p, salt_hex, digest_hex = hashed.split("$")
            if (n, r, p) != ("16384", "8", "1"):
                return False
            candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                                       n=2**14, r=8, p=1, dklen=32)
            return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
        return hashed.startswith("$2") and bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


def _hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = peer in settings.TRUSTED_PROXY_IPS
    if not trusted:
        for hostname in settings.TRUSTED_PROXY_HOSTNAMES:
            try:
                addresses = {entry[4][0] for entry in socket.getaddrinfo(
                    hostname, None, type=socket.SOCK_STREAM)}
            except OSError:
                continue
            if peer in addresses:
                trusted = True
                break
    if trusted:
        forwarded = request.headers.get("X-Real-IP") or ""
        try:
            return str(ipaddress.ip_address(forwarded))
        except ValueError:
            pass
    return peer


def _require_browser_origin(request: Request, *, cookie_session: bool = False) -> None:
    """Check auth origins; production cookie issuance requires an Origin header."""
    site = (request.headers.get("Sec-Fetch-Site") or "").lower()
    if site in {"cross-site", "same-site"}:
        raise HTTPException(status_code=403, detail="Cross-origin authentication is not allowed")
    origin = request.headers.get("Origin")
    if cookie_session and settings.ENVIRONMENT == "production" and not origin:
        raise HTTPException(status_code=403, detail="Authentication origin is required")
    allowed_origins = {settings.PUBLIC_ORIGIN}
    if settings.ENVIRONMENT != "production":
        allowed_origins.update(settings.CORS_ORIGINS)
        allowed_origins.add(str(request.base_url).rstrip("/"))
    if origin and origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="Authentication origin is not allowed")


def _reserve_login_attempt(username: str, ip: str) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM login_attempts WHERE failed_at < ?", (cutoff,))
        account_count = connection.execute("""SELECT COUNT(*) FROM login_attempts
            WHERE username=? AND ip=?""", (username, ip)).fetchone()[0]
        ip_count = connection.execute("SELECT COUNT(*) FROM login_attempts WHERE ip=?",
                                      (ip,)).fetchone()[0]
        if account_count >= 5 or ip_count >= 30:
            raise HTTPException(status_code=429, detail="Too many attempts. Try again in 15 minutes")
        connection.execute("INSERT INTO login_attempts VALUES (?, ?, ?)",
                           (username, ip, datetime.now(timezone.utc).isoformat()))


def _reserve_registration_attempt(ip: str) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM registration_attempts WHERE attempted_at < ?", (cutoff,))
        count = connection.execute("SELECT COUNT(*) FROM registration_attempts WHERE ip=?",
                                   (ip,)).fetchone()[0]
        if count >= 5:
            raise HTTPException(status_code=429, detail="Too many registrations. Try again in 15 minutes")
        connection.execute("INSERT INTO registration_attempts VALUES (?, ?)",
                           (ip, datetime.now(timezone.utc).isoformat()))


def _create_session(username: str) -> tuple[str, str]:
    csrf = secrets.token_urlsafe(24)
    jti = secrets.token_urlsafe(24)
    token = create_access_token({"sub": username, "jti": jti,
                                 "csrf_hash": hashlib.sha256(csrf.encode()).hexdigest()})
    max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    now = datetime.now(timezone.utc)
    with _connect() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
        connection.execute("""INSERT INTO sessions(jti_hash, username, expires_at, created_at)
            VALUES (?, ?, ?, ?)""", (hashlib.sha256(jti.encode()).hexdigest(), username,
                                     (now + timedelta(seconds=max_age)).isoformat(), now.isoformat()))
    return token, csrf


def _set_session(response: Response, username: str) -> str:
    token, csrf = _create_session(username)
    max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie("pulse_session", token, max_age=max_age, path="/api",
                        secure=settings.COOKIE_SECURE, httponly=True, samesite="strict")
    response.set_cookie("pulse_csrf", csrf, max_age=max_age, path="/",
                        secure=settings.COOKIE_SECURE, httponly=False, samesite="strict")
    return token


class User(BaseModel):
    username: str
    role: str = "user"


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=128)


class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def get_current_user(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> User:
    unauthorized = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        bearer = bool(token)
        session_token = token or request.cookies.get("pulse_session")
        if not session_token:
            raise unauthorized
        payload = jwt.decode(session_token, settings.SECRET_KEY, algorithms=["HS256"])
        if not bearer and request.method not in {"GET", "HEAD", "OPTIONS"}:
            csrf_cookie = request.cookies.get("pulse_csrf") or ""
            csrf_header = request.headers.get("X-CSRF-Token") or ""
            expected = payload.get("csrf_hash") or ""
            if not csrf_cookie or not hmac.compare_digest(csrf_cookie, csrf_header) or not hmac.compare_digest(hashlib.sha256(csrf_cookie.encode()).hexdigest(), expected):
                raise HTTPException(403, detail="CSRF validation failed")
        username = payload.get("sub")
        jti = payload.get("jti")
        if not username or not jti:
            raise unauthorized
    except PyJWTError:
        raise unauthorized
    with _connect() as connection:
        session = connection.execute("""SELECT 1 FROM sessions
            WHERE jti_hash=? AND username=? AND expires_at>?""",
            (hashlib.sha256(jti.encode()).hexdigest(), username,
             datetime.now(timezone.utc).isoformat())).fetchone()
    if _user(username) is None or not session:
        raise unauthorized
    return User(username=username)


@router.post("/register", response_model=User, status_code=201)
def register(account: Registration, request: Request, response: Response):
    _require_browser_origin(request, cookie_session=True)
    _reserve_registration_attempt(_client_ip(request))
    if _user(account.username):
        raise HTTPException(status_code=409, detail="That username is already registered")
    hashed = _hash_password(account.password)
    try:
        with _connect() as connection:
            connection.execute("INSERT INTO accounts(username, password_hash, created_at) VALUES (?, ?, ?)", (account.username, hashed, datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already registered")
    _set_session(response, account.username)
    return User(username=account.username)


def _authenticate_login(request: Request, form_data: OAuth2PasswordRequestForm,
                        *, cookie_session: bool = False) -> str:
    _require_browser_origin(request, cookie_session=cookie_session)
    ip = _client_ip(request)
    _reserve_login_attempt(form_data.username, ip)
    user = _user(form_data.username)
    if user is None:
        _verify(form_data.password, DUMMY_PASSWORD_HASH)
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    if not _verify(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    with _connect() as connection:
        connection.execute("DELETE FROM login_attempts WHERE username=? AND ip=?", (form_data.username, ip))
    if (not user["password_hash"].startswith("$argon2id$") or
            PASSWORD_HASHER.check_needs_rehash(user["password_hash"])):
        with _connect() as connection:
            connection.execute("""UPDATE accounts SET password_hash=?
                WHERE username=? AND password_hash=?""",
                (_hash_password(form_data.password), user["username"],
                 user["password_hash"]))
    return user["username"]


@router.post("/session", response_model=User)
def login_session(request: Request, response: Response,
                  form_data: OAuth2PasswordRequestForm = Depends()):
    username = _authenticate_login(request, form_data, cookie_session=True)
    _set_session(response, username)
    return User(username=username)


@router.post("/token", response_model=Token)
def login(request: Request,
          form_data: OAuth2PasswordRequestForm = Depends()):
    username = _authenticate_login(request, form_data)
    token, _ = _create_session(username)
    return Token(access_token=token, token_type="bearer")


@router.post("/logout")
def logout(request: Request, response: Response,
           current_user: User = Depends(get_current_user)):
    authorization = request.headers.get("Authorization") or ""
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    raw = bearer or request.cookies.get("pulse_session")
    try:
        jti = jwt.decode(raw, settings.SECRET_KEY, algorithms=["HS256"]).get("jti")
    except PyJWTError:
        jti = None
    if jti:
        with _connect() as connection:
            connection.execute("DELETE FROM sessions WHERE jti_hash=?",
                               (hashlib.sha256(jti.encode()).hexdigest(),))
    response.delete_cookie("pulse_session", path="/api")
    response.delete_cookie("pulse_csrf", path="/")
    return {"logged_out": True}


@router.post("/logout-all")
def logout_all(response: Response, current_user: User = Depends(get_current_user)):
    """Revoke every server session for the authenticated account."""
    with _connect() as connection:
        revoked = connection.execute("DELETE FROM sessions WHERE username=?",
                                     (current_user.username,)).rowcount
    response.delete_cookie("pulse_session", path="/api")
    response.delete_cookie("pulse_csrf", path="/")
    return {"logged_out": True, "sessions_revoked": revoked}


@router.get("/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
