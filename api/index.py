import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel, ConfigDict, Field
from psycopg2 import Error as PsycopgError
from psycopg2.pool import PoolError, ThreadedConnectionPool

from config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cukur_ai")

app = FastAPI(
    title="Cukur AI",
    version="1.0.0",
    description="Serverless conversational assistant backed by Neon Postgres.",
)

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()
_schema_ready = False
_schema_lock = threading.Lock()

_RESTRICTED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:run|execute)\s+(?:a\s+)?(?:shell|terminal|system)\s+commands?\b",
        r"\b(?:execute|run)\s+(?:arbitrary\s+)?(?:python|javascript|code)\b",
        r"\b(?:delete|drop|truncate)\s+(?:the\s+)?(?:database|table|all\s+data)\b",
        r"\b(?:access|reveal|dump)\s+(?:the\s+)?(?:secret|credentials|api\s+key|passwords?)\b",
        r"\b(?:bypass|disable)\s+(?:security|authentication|auth)\b",
        r"\bsudo\b",
    )
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    developer_mode: bool = False


class VerifyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=256)


@contextmanager
def _database_connection() -> Iterator[Any]:
    global _pool

    if not settings.neon_database_url:
        raise HTTPException(
            status_code=503,
            detail="Neon database is not configured. Set NEON_DATABASE_URL.",
        )

    try:
        if _pool is None:
            with _pool_lock:
                if _pool is None:
                    _pool = ThreadedConnectionPool(
                        minconn=1,
                        maxconn=5,
                        dsn=settings.neon_database_url,
                        connect_timeout=10,
                        sslmode="require",
                    )
        connection = _pool.getconn()
    except (PsycopgError, PoolError) as exc:
        logger.exception("Could not establish a Neon database connection.")
        raise HTTPException(
            status_code=503,
            detail="Neon database is unavailable. Check NEON_DATABASE_URL and try again.",
        ) from exc

    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        _pool.putconn(connection)


def _ensure_schema() -> None:
    global _schema_ready

    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return
        with _database_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chats_history (
                            id BIGSERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                            text TEXT NOT NULL,
                            developer_mode_active BOOLEAN NOT NULL DEFAULT FALSE,
                            timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS chats_history_session_timestamp_idx
                        ON chats_history (session_id, timestamp DESC)
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS developer_sessions (
                            session_id TEXT PRIMARY KEY,
                            developer_mode_active BOOLEAN NOT NULL DEFAULT FALSE,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                connection.commit()
                _schema_ready = True
            except PsycopgError as exc:
                connection.rollback()
                logger.exception("Could not initialize the Neon database schema.")
                raise HTTPException(
                    status_code=503,
                    detail="Could not initialize the Neon database schema.",
                ) from exc


def _restricted_request(message: str) -> bool:
    return any(pattern.search(message) for pattern in _RESTRICTED_PATTERNS)


def _get_session_state(session_id: str) -> bool:
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT developer_mode_active
                    FROM developer_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not read Developer Mode session state.")
            raise HTTPException(
                status_code=503,
                detail="Could not read session state from Neon.",
            ) from exc
    return bool(row and row[0])


def _save_messages(
    session_id: str,
    user_message: str,
    assistant_message: str,
    developer_mode_active: bool,
) -> None:
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chats_history
                        (session_id, role, text, developer_mode_active)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        (session_id, "user", user_message, developer_mode_active),
                        (session_id, "assistant", assistant_message, developer_mode_active),
                    ),
                )
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not save chat messages to Neon.")
            raise HTTPException(
                status_code=503,
                detail="Could not save this conversation to Neon.",
            ) from exc


def _generate_reply(session_id: str, message: str) -> str:
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, text
                    FROM chats_history
                    WHERE session_id = %s
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 12
                    """,
                    (session_id,),
                )
                history = list(reversed(cursor.fetchall()))
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not load conversation context from Neon.")
            raise HTTPException(
                status_code=503,
                detail="Could not load conversation history from Neon.",
            ) from exc

    messages = [
        {
            "role": "system",
            "content": (
                "You are Cukur AI, a clear, capable virtual assistant. "
                "Do not claim to have executed actions or accessed systems. "
                "Developer Mode is an authorization state, not a grant of tools."
            ),
        }
    ]
    messages.extend({"role": role, "content": text} for role, text in history)
    messages.append({"role": "user", "content": message})

    try:
        if settings.ollama_host:
            response = requests.post(
                f"{settings.ollama_host.rstrip('/')}/api/chat",
                json={"model": "llama3", "messages": messages, "stream": False},
                timeout=(5, 60),
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
        elif settings.openai_api_key:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": "gpt-4o-mini", "messages": messages},
                timeout=(5, 60),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        else:
            raise HTTPException(
                status_code=503,
                detail="No AI provider is configured. Set OLLAMA_HOST or OPENAI_API_KEY.",
            )
    except HTTPException:
        raise
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.exception("The configured AI provider could not complete the request.")
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider is unavailable or returned an invalid response.",
        ) from exc

    if not isinstance(content, str) or not content.strip():
        logger.error("The configured AI provider returned an empty response.")
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider returned an empty response.",
        )
    return content.strip()


@app.on_event("startup")
def initialize_database() -> None:
    try:
        _ensure_schema()
    except HTTPException:
        logger.warning("Database initialization deferred; database-backed requests will report errors.")


@app.get("/api/health")
def health() -> dict[str, str]:
    _ensure_schema()
    return {"status": "ok"}


@app.get("/api/session/{session_id}")
def session_status(session_id: str) -> dict[str, Any]:
    _ensure_schema()
    return {
        "session_id": session_id,
        "developer_mode_active": _get_session_state(session_id),
    }


@app.post("/api/verify")
def verify_developer_mode(request: VerifyRequest) -> dict[str, Any]:
    _ensure_schema()
    import secrets

    if not secrets.compare_digest(request.code, settings.developer_bypass_code):
        raise HTTPException(status_code=401, detail="Invalid authentication code.")

    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO developer_sessions (session_id, developer_mode_active)
                    VALUES (%s, TRUE)
                    ON CONFLICT (session_id) DO UPDATE
                    SET developer_mode_active = TRUE, updated_at = CURRENT_TIMESTAMP
                    """,
                    (request.session_id,),
                )
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not activate Developer Mode.")
            raise HTTPException(
                status_code=503,
                detail="Could not save Developer Mode status to Neon.",
            ) from exc

    return {"session_id": request.session_id, "developer_mode_active": True}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    _ensure_schema()
    developer_mode_active = _get_session_state(request.session_id)

    if _restricted_request(request.message) and not developer_mode_active:
        reply = "Authentication required. Verify your developer code to continue."
        _save_messages(request.session_id, request.message, reply, False)
        return {
            "session_id": request.session_id,
            "reply": reply,
            "developer_mode_active": False,
            "auth_required": True,
        }

    reply = _generate_reply(request.session_id, request.message)
    _save_messages(
        request.session_id,
        request.message,
        reply,
        developer_mode_active,
    )
    return {
        "session_id": request.session_id,
        "reply": reply,
        "developer_mode_active": developer_mode_active,
        "auth_required": False,
    }


handler = Mangum(app, lifespan="auto")
