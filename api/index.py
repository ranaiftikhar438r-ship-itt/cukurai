import base64
import binascii
import json
import logging
import secrets
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, ConfigDict, Field
from psycopg2 import Error as PsycopgError
from psycopg2.pool import PoolError, ThreadedConnectionPool
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cukur_ai")
DEVELOPER_MODE_CODE = "Cukur301r"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-3.5-flash-lite"
GEMINI_MODELS = (GEMINI_MODEL, GEMINI_MODEL_FALLBACK)
GEMINI_LIVE_MODEL = "gemini-3.8-live"
NIKO_PERSONA_INSTRUCTIONS = (
    "You are Niko, a highly analytical, observant, and dependable conversational "
    "companion. Speak clearly, calmly, and naturally. Do not claim a measured IQ "
    "or abilities you do not have. Do not claim to operate devices or take actions "
    "outside this conversation. "
    "For medical questions or human medical imagery/data: first list only raw, "
    "directly observable features and uncertainty; never give a definitive diagnosis; "
    "describe at least three distinct possibilities using cautious language such as "
    "'could resemble' or 'may be consistent with'; give at least three distinct "
    "general management options for wellness advice; ask the user to verify physical "
    "product labels and consult a qualified clinician; and always end with this exact "
    "sentence: 'Consult a licensed medical professional before making any health decisions.'"
)
MEDICAL_IMAGE_INSTRUCTIONS = (
    "When the user asks about medical imagery or human medical data, start by listing "
    "the raw visual features observed before interpretation. Do not present a definitive "
    "diagnosis; use hedged language and list at least three distinct possible explanations. "
    "Any wellness or remediation advice must give at least three distinct general options. "
    "Tell the user to double-check physical product labels. State that the information is "
    "general only and append exactly: 'Consult a licensed medical professional before "
    "making any health decisions.'"
)


def _is_gemini_model_not_found(exc: genai_errors.ClientError) -> bool:
    if exc.code != 404:
        return False
    error_detail = str(exc).lower()
    return "model" in error_detail and any(
        phrase in error_detail
        for phrase in ("not found", "not_found", "does not exist", "unsupported")
    )


app = FastAPI(
    title="Cukur AI",
    version="1.0.0",
    description="Serverless conversational assistant backed by Neon Postgres.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "X-Developer-Code"],
)

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()
_schema_ready = False
_schema_lock = threading.Lock()

class ImageContext(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    media_type: str = Field(pattern=r"^image/(?:png|jpeg|webp|gif)$")
    data: str = Field(min_length=1, max_length=8_000_000)
    description: str | None = Field(default=None, max_length=1000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    developer_mode: bool = False
    images: list[ImageContext] = Field(default_factory=list, max_length=4)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=256)


class LiveTokenRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)


class PromptRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=16000)
    system_prompt: str = Field(
        default="You are Cukur AI, a capable and clear virtual assistant.",
        max_length=4000,
    )


class ImageAnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=8000)
    images: list[ImageContext] = Field(min_length=1, max_length=4)


class CodeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    task: str = Field(min_length=1, max_length=8000)
    code: str = Field(min_length=1, max_length=24000)
    language: str = Field(default="text", min_length=1, max_length=64)


class AgentTaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AgentStepStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class AgentStep(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=1000)
    status: AgentStepStatus = AgentStepStatus.pending
    result: str | None = Field(default=None, max_length=8000)


class AgentTaskCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1, max_length=4000)
    steps: list[AgentStep] = Field(default_factory=list, max_length=50)


class AgentStepUpdate(BaseModel):
    status: AgentStepStatus
    result: str | None = Field(default=None, max_length=8000)


class AgentTaskStateUpdate(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    status: AgentTaskStatus


@dataclass(frozen=True)
class AICompletion:
    text: str
    grounding_metadata: dict[str, Any] | None = None
    sources: list[dict[str, str]] | None = None
    web_search_queries: list[str] | None = None


def require_developer_code(
    x_developer_code: str | None = Header(default=None, alias="X-Developer-Code"),
) -> None:
    if not x_developer_code or not secrets.compare_digest(
        x_developer_code,
        DEVELOPER_MODE_CODE,
    ):
        raise HTTPException(status_code=403, detail="Developer authorization required.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled error while handling %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


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
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_tasks (
                            task_id UUID PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            goal TEXT NOT NULL,
                            status TEXT NOT NULL CHECK (
                                status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
                            ),
                            steps JSONB NOT NULL DEFAULT '[]'::jsonb,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS agent_tasks_session_updated_idx
                        ON agent_tasks (session_id, updated_at DESC)
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


def _generate_reply(
    session_id: str,
    message: str,
    gemini_images: list[tuple[str, str]] | None = None,
) -> AICompletion:
    return _call_llm_completion(
        _chat_messages(session_id, message),
        ollama_model="llama3",
        gemini_images=gemini_images,
    )


def _chat_messages(session_id: str, message: str) -> list[dict[str, str]]:
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
                f"{NIKO_PERSONA_INSTRUCTIONS} "
                "Do not claim to have executed actions or accessed systems. "
                "Developer Mode is an authorization state, not a grant of tools. "
                f"{MEDICAL_IMAGE_INSTRUCTIONS}"
            ),
        }
    ]
    messages.extend({"role": role, "content": text} for role, text in history)
    messages.append({"role": "user", "content": message})
    return messages


def _call_llm(
    messages: list[dict[str, Any]],
    *,
    ollama_model: str,
    gemini_images: list[tuple[str, str]] | None = None,
) -> str:
    return _call_llm_completion(
        messages,
        ollama_model=ollama_model,
        gemini_images=gemini_images,
    ).text


def _call_llm_completion(
    messages: list[dict[str, Any]],
    *,
    ollama_model: str,
    gemini_images: list[tuple[str, str]] | None = None,
) -> AICompletion:
    try:
        if gemini_images and not settings.gemini_api_key:
            raise HTTPException(
                status_code=503,
                detail="Image chat requires GEMINI_API_KEY to be configured.",
            )
        if settings.gemini_api_key:
            return _call_gemini(messages, gemini_images=gemini_images)
        if settings.ollama_host:
            response = requests.post(
                f"{settings.ollama_host.rstrip('/')}/api/chat",
                json={"model": ollama_model, "messages": messages, "stream": False},
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
                detail=(
                    "No AI provider is configured. Set GEMINI_API_KEY, OLLAMA_HOST, "
                    "or OPENAI_API_KEY."
                ),
            )
    except HTTPException:
        raise
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ) as exc:
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
    return AICompletion(text=content.strip())


def _call_gemini(
    messages: list[dict[str, Any]],
    *,
    gemini_images: list[tuple[str, str]] | None = None,
) -> AICompletion:
    api_key = settings.gemini_api_key
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini is not configured on this server.")

    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        text = message.get("content")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="AI message content must be text.")
        if role == "system":
            system_parts.append({"text": text})
        elif role in ("user", "assistant"):
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": text}],
                }
            )

    if gemini_images:
        user_content = next(
            (content for content in reversed(contents) if content["role"] == "user"),
            None,
        )
        if user_content is None:
            raise HTTPException(status_code=422, detail="Image analysis requires a user message.")
        user_content["parts"].extend(
            {"inline_data": {"mime_type": media_type, "data": encoded_image}}
            for media_type, encoded_image in gemini_images
        )

    if not contents:
        raise HTTPException(status_code=422, detail="At least one user message is required.")

    system_instruction = "\n".join(part["text"] for part in system_parts) or None
    client: genai.Client | None = None
    try:
        configs = (
            genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            ),
            genai_types.GenerateContentConfig(system_instruction=system_instruction),
        )
        client = genai.Client(api_key=api_key)
        response = None
        for model_index, model in enumerate(GEMINI_MODELS):
            for config_index, config in enumerate(configs):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    break
                except genai_errors.ClientError as exc:
                    error_detail = str(exc).lower()
                    is_search_tool_error = exc.code == 400 and any(
                        marker in error_detail
                        for marker in ("google_search", "google search", "grounding", "tool")
                    )
                    if config_index == 0 and is_search_tool_error:
                        logger.warning(
                            "Gemini rejected Google Search grounding; retrying without it."
                        )
                        continue
                    if model_index == 0 and _is_gemini_model_not_found(exc):
                        logger.warning(
                            "Gemini model %s was not found; retrying with %s.",
                            model,
                            GEMINI_MODEL_FALLBACK,
                        )
                        break
                    raise
            if response is not None:
                break
        if response is None:
            raise HTTPException(
                status_code=502,
                detail="Gemini models were unavailable. Check API access and try again.",
            )
    except genai_errors.APIError as exc:
        status_code = exc.code
        logger.error("Gemini rejected the request with HTTP %s: %s", status_code, exc)
        provider_detail = exc.message or "Gemini rejected the configured models."
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini rejected the request (HTTP {status_code}): {provider_detail}"
            ),
        ) from exc
    except (
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        logger.exception("Gemini could not initialize or complete the request.")
        raise HTTPException(
            status_code=502,
            detail="Gemini is unavailable or returned an invalid response.",
        ) from exc
    finally:
        if client is not None:
            client.close()

    try:
        candidates = response.candidates or []
        if not candidates:
            logger.error("Gemini returned no response candidates.")
            raise HTTPException(
                status_code=502,
                detail="Gemini did not return a response. Check the prompt and model configuration.",
            )
        candidate = candidates[0]
        parts = candidate.content.parts if candidate.content else []
        text = "\n".join(part.text for part in parts if isinstance(part.text, str))
        grounding_metadata = candidate.grounding_metadata
        sources_by_uri: dict[str, dict[str, str]] = {}
        if grounding_metadata:
            for chunk in grounding_metadata.grounding_chunks or []:
                web_source = chunk.web
                if web_source is None:
                    continue
                uri = web_source.uri
                title = web_source.title
                if (
                    isinstance(uri, str)
                    and uri.startswith(("https://", "http://"))
                ):
                    sources_by_uri[uri] = {
                        "title": title if isinstance(title, str) and title else uri,
                        "uri": uri,
                    }
        web_search_queries = (
            [
                query
                for query in (grounding_metadata.web_search_queries or [])
                if isinstance(query, str)
            ]
            if grounding_metadata
            else []
        )
    except HTTPException:
        raise
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
    ) as exc:
        logger.exception("Gemini could not complete the request.")
        raise HTTPException(
            status_code=502,
            detail="Gemini is unavailable or returned an invalid response.",
        ) from exc

    if not text.strip():
        logger.error("Gemini returned an empty response.")
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
    return AICompletion(
        text=text.strip(),
        grounding_metadata=grounding_metadata,
        sources=list(sources_by_uri.values()),
        web_search_queries=web_search_queries,
    )


def _stream_gemini_completion(
    messages: list[dict[str, Any]],
    *,
    gemini_images: list[tuple[str, str]] | None = None,
) -> Iterator[str | AICompletion]:
    api_key = settings.gemini_api_key
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini is not configured on this server.")

    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        text = message.get("content")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="AI message content must be text.")
        if role == "system":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": text}],
                }
            )

    if gemini_images:
        user_content = next(
            (content for content in reversed(contents) if content["role"] == "user"),
            None,
        )
        if user_content is None:
            raise HTTPException(status_code=422, detail="Image analysis requires a user message.")
        user_content["parts"].extend(
            {"inline_data": {"mime_type": media_type, "data": encoded_image}}
            for media_type, encoded_image in gemini_images
        )
    if not contents:
        raise HTTPException(status_code=422, detail="At least one user message is required.")

    system_instruction = "\n".join(system_parts) or None
    configs = (
        genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
        ),
        genai_types.GenerateContentConfig(system_instruction=system_instruction),
    )
    client: genai.Client | None = None
    response_text: list[str] = []
    grounding_metadata: Any = None
    try:
        client = genai.Client(api_key=api_key)
        stream_completed = False
        for model_index, model in enumerate(GEMINI_MODELS):
            for config_index, config in enumerate(configs):
                try:
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=config,
                    ):
                        candidates = chunk.candidates or []
                        if candidates and candidates[0].grounding_metadata:
                            grounding_metadata = candidates[0].grounding_metadata
                        text = chunk.text
                        if isinstance(text, str) and text:
                            response_text.append(text)
                            yield text
                    stream_completed = True
                    break
                except genai_errors.ClientError as exc:
                    error_detail = str(exc).lower()
                    is_search_tool_error = exc.code == 400 and any(
                        marker in error_detail
                        for marker in ("google_search", "google search", "grounding", "tool")
                    )
                    if config_index == 0 and is_search_tool_error and not response_text:
                        logger.warning(
                            "Gemini rejected Google Search grounding; retrying stream without it."
                        )
                        continue
                    if (
                        model_index == 0
                        and not response_text
                        and _is_gemini_model_not_found(exc)
                    ):
                        logger.warning(
                            "Gemini streaming model %s was not found; retrying with %s.",
                            model,
                            GEMINI_MODEL_FALLBACK,
                        )
                        break
                    raise
            if stream_completed:
                break
        if not "".join(response_text).strip():
            logger.error("Gemini returned an empty streamed response.")
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )
        sources_by_uri: dict[str, dict[str, str]] = {}
        if grounding_metadata:
            for chunk in grounding_metadata.grounding_chunks or []:
                web_source = chunk.web
                if web_source is None:
                    continue
                uri = web_source.uri
                title = web_source.title
                if isinstance(uri, str) and uri.startswith(("https://", "http://")):
                    sources_by_uri[uri] = {
                        "title": title if isinstance(title, str) and title else uri,
                        "uri": uri,
                    }
        web_search_queries = (
            [
                query
                for query in (grounding_metadata.web_search_queries or [])
                if isinstance(query, str)
            ]
            if grounding_metadata
            else []
        )
        yield AICompletion(
            text="".join(response_text).strip(),
            grounding_metadata=grounding_metadata,
            sources=list(sources_by_uri.values()),
            web_search_queries=web_search_queries,
        )
    except HTTPException:
        raise
    except genai_errors.APIError as exc:
        logger.error("Gemini streaming request failed with HTTP %s: %s", exc.code, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini rejected the request (HTTP {exc.code}): "
                f"{exc.message or 'Gemini rejected the configured models.'}"
            ),
        ) from exc
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Gemini streaming request returned an invalid response.")
        raise HTTPException(
            status_code=502,
            detail="Gemini is unavailable or returned an invalid response.",
        ) from exc
    finally:
        if client is not None:
            client.close()


def _stream_completion(
    messages: list[dict[str, Any]],
    *,
    ollama_model: str,
    gemini_images: list[tuple[str, str]] | None = None,
) -> Iterator[str | AICompletion]:
    if settings.gemini_api_key:
        yield from _stream_gemini_completion(messages, gemini_images=gemini_images)
        return
    if gemini_images:
        raise HTTPException(
            status_code=503,
            detail="Image chat requires GEMINI_API_KEY to be configured.",
        )

    response_text: list[str] = []
    try:
        if settings.ollama_host:
            response = requests.post(
                f"{settings.ollama_host.rstrip('/')}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": messages,
                    "stream": True,
                },
                timeout=(5, 60),
                stream=True,
            )
            is_openai = False
        elif settings.openai_api_key:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "stream": True,
                },
                timeout=(5, 60),
                stream=True,
            )
            is_openai = True
        else:
            raise HTTPException(
                status_code=503,
                detail="No AI provider is configured. Set GEMINI_API_KEY, OLLAMA_HOST, "
                "or OPENAI_API_KEY.",
            )

        with response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line:
                    continue
                if is_openai:
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    choices = data.get("choices", [])
                    delta = choices[0].get("delta", {}).get("content") if choices else None
                else:
                    data = json.loads(line)
                    delta = data.get("message", {}).get("content")
                if isinstance(delta, str) and delta:
                    response_text.append(delta)
                    yield delta
    except HTTPException:
        raise
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ) as exc:
        logger.exception("The configured AI provider could not stream a response.")
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider is unavailable or returned an invalid response.",
        ) from exc

    text = "".join(response_text).strip()
    if not text:
        logger.error("The configured AI provider returned an empty streamed response.")
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider returned an empty response.",
        )
    yield AICompletion(text=text)


def _completion_response_fields(completion: AICompletion) -> dict[str, Any]:
    return {
        "grounding_metadata": completion.grounding_metadata,
        "sources": completion.sources or [],
        "web_search_queries": completion.web_search_queries or [],
    }


def _decode_image(image: ImageContext) -> str:
        encoded = image.data
        if encoded.startswith("data:"):
            header, separator, encoded = encoded.partition(",")
            expected_header = f"data:{image.media_type};base64"
            if not separator or header != expected_header:
                raise HTTPException(
                    status_code=422,
                    detail="Image data URI must match media_type and use base64 encoding.",
                )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Image data must be valid base64.") from exc
        if not raw or len(raw) > 6_000_000:
            raise HTTPException(
                status_code=413,
                detail="Each decoded image must be between 1 byte and 6 MB.",
            )
        return base64.b64encode(raw).decode("ascii")


@app.on_event("startup")
def initialize_database() -> None:
    try:
        _ensure_schema()
    except HTTPException:
        logger.warning("Database initialization deferred; database-backed requests will report errors.")


@app.get("/")
def root() -> HTMLResponse:
    dashboard_path = Path(__file__).resolve().parent.parent / "public" / "index.html"
    try:
        dashboard_html = dashboard_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not read dashboard HTML from %s.", dashboard_path)
        raise HTTPException(status_code=500, detail="Dashboard is unavailable.") from exc
    return HTMLResponse(content=dashboard_html)


@app.get("/app", include_in_schema=False)
def dashboard() -> FileResponse:
    dashboard_path = Path(__file__).resolve().parent.parent / "public" / "index.html"
    if not dashboard_path.is_file():
        logger.error("Dashboard HTML was not found at %s.", dashboard_path)
        raise HTTPException(status_code=404, detail="Dashboard is unavailable.")
    return FileResponse(dashboard_path)


@app.get("/api/health")
def health() -> dict[str, str]:
    _ensure_schema()
    return {"status": "ok"}


@app.get("/api/session/{session_id}")
def session_status(session_id: str) -> dict[str, Any]:
    if not settings.neon_database_url:
        return {
            "session_id": session_id,
            "developer_mode_active": False,
            "persistence_enabled": False,
        }
    _ensure_schema()
    return {
        "session_id": session_id,
        "developer_mode_active": _get_session_state(session_id),
        "persistence_enabled": True,
    }


@app.post("/api/verify")
def verify_developer_mode(request: VerifyRequest) -> dict[str, Any]:
    if not secrets.compare_digest(request.code, DEVELOPER_MODE_CODE):
        raise HTTPException(status_code=401, detail="Invalid authentication code.")

    if not settings.neon_database_url:
        return {
            "session_id": request.session_id,
            "developer_mode_active": True,
            "persistence_enabled": False,
        }

    _ensure_schema()

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

    return {
        "session_id": request.session_id,
        "developer_mode_active": True,
        "persistence_enabled": True,
    }


@app.post("/api/live/token", dependencies=[Depends(require_developer_code)])
def create_live_token(request: LiveTokenRequest) -> dict[str, str]:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Live voice requires GEMINI_API_KEY to be configured.",
        )

    now = datetime.now(timezone.utc)
    live_config = genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=NIKO_PERSONA_INSTRUCTIONS,
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
        output_audio_transcription=genai_types.AudioTranscriptionConfig(),
    )
    client: genai.Client | None = None
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        token = client.auth_tokens.create(
            config=genai_types.CreateAuthTokenConfig(
                uses=1,
                expire_time=now + timedelta(minutes=30),
                new_session_expire_time=now + timedelta(seconds=60),
                live_connect_constraints=genai_types.LiveConnectConstraints(
                    model=f"models/{GEMINI_LIVE_MODEL}",
                    config=live_config,
                ),
            )
        )
    except genai_errors.APIError as exc:
        logger.exception("Could not create a Gemini Live session token.")
        raise HTTPException(
            status_code=502,
            detail="Could not initialize Gemini Live voice.",
        ) from exc
    except (ValueError, TypeError) as exc:
        logger.exception("Gemini Live token configuration was invalid.")
        raise HTTPException(
            status_code=502,
            detail="Could not initialize Gemini Live voice.",
        ) from exc
    finally:
        if client is not None:
            client.close()

    if not token.name:
        logger.error("Gemini returned an empty Live API token.")
        raise HTTPException(status_code=502, detail="Gemini returned an invalid Live token.")
    return {
        "token": token.name,
        "model": GEMINI_LIVE_MODEL,
        "persona": NIKO_PERSONA_INSTRUCTIONS,
        "session_id": request.session_id,
    }


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    _: None = Depends(require_developer_code),
) -> StreamingResponse:
    developer_mode_active = True
    gemini_images = [
        (image.media_type, _decode_image(image))
        for image in request.images
    ]
    if sum(len(encoded_image) for _, encoded_image in gemini_images) > 4_000_000:
        raise HTTPException(
            status_code=413,
            detail="Combined image data must not exceed 3 MB before base64 encoding.",
        )
    persistence_enabled = bool(settings.neon_database_url)
    if persistence_enabled:
        _ensure_schema()
        messages = _chat_messages(request.session_id, request.message)
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{NIKO_PERSONA_INSTRUCTIONS} "
                    "Do not claim to have executed actions or accessed systems. "
                    "Developer Mode is an authorization state, not a grant of tools. "
                    f"{MEDICAL_IMAGE_INSTRUCTIONS}"
                ),
            },
            {"role": "user", "content": request.message},
        ]

    def event_stream() -> Iterator[str]:
        chunks: list[str] = []
        completion: AICompletion | None = None
        try:
            for item in _stream_completion(
                messages,
                ollama_model="llama3",
                gemini_images=gemini_images,
            ):
                if isinstance(item, str):
                    chunks.append(item)
                    event = {"type": "delta", "text": item}
                else:
                    completion = item
                    event = None
                if event is not None:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if completion is None:
                completion = AICompletion(text="".join(chunks).strip())
            if persistence_enabled:
                _save_messages(
                    request.session_id,
                    request.message,
                    completion.text,
                    developer_mode_active,
                )
            done_event = {
                "type": "done",
                "session_id": request.session_id,
                "reply": completion.text,
                "developer_mode_active": developer_mode_active,
                "auth_required": False,
                "persistence_enabled": persistence_enabled,
                **_completion_response_fields(completion),
            }
            yield f"data: {json.dumps(jsonable_encoder(done_event), ensure_ascii=False)}\n\n"
        except HTTPException as exc:
            error_event = {"type": "error", "detail": exc.detail}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Chat response stream failed.")
            error_event = {
                "type": "error",
                "detail": "The assistant could not complete this request.",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ai/prompt", dependencies=[Depends(require_developer_code)])
def process_prompt(request: PromptRequest) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": f"{request.system_prompt}\n\n{MEDICAL_IMAGE_INSTRUCTIONS}",
        },
        {"role": "user", "content": request.prompt},
    ]
    return {"response": _call_llm(messages, ollama_model="llama3")}


@app.post("/api/ai/images/analyze", dependencies=[Depends(require_developer_code)])
def analyze_images(request: ImageAnalysisRequest) -> dict[str, str]:
    encoded_images = [_decode_image(image) for image in request.images]
    if sum(len(encoded) for encoded in encoded_images) > 4_000_000:
        raise HTTPException(
            status_code=413,
            detail="Combined image data must not exceed 3 MB before base64 encoding.",
        )
    image_context = "\n".join(
        f"Image {index}: {image.description or 'No additional description.'}"
        for index, image in enumerate(request.images, start=1)
    )
    prompt = f"{request.prompt}\n\nImage context:\n{image_context}"

    if settings.gemini_api_key:
        result = _call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Analyze the supplied images and answer accurately. State uncertainty. "
                        f"{MEDICAL_IMAGE_INSTRUCTIONS}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            ollama_model="llava",
            gemini_images=[
                (image.media_type, encoded)
                for image, encoded in zip(request.images, encoded_images)
            ],
        )
    elif settings.ollama_host:
        messages = [
            {
                "role": "system",
                "content": (
                    "Analyze the supplied images and answer accurately. State uncertainty. "
                    f"{MEDICAL_IMAGE_INSTRUCTIONS}"
                ),
            },
            {"role": "user", "content": prompt, "images": encoded_images},
        ]
        result = _call_llm(messages, ollama_model="llava")
    elif settings.openai_api_key:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.media_type};base64,{encoded}",
                    "detail": "auto",
                },
            }
            for image, encoded in zip(request.images, encoded_images)
        )
        result = _call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Analyze the supplied images and answer accurately. State uncertainty. "
                        f"{MEDICAL_IMAGE_INSTRUCTIONS}"
                    ),
                },
                {"role": "user", "content": content},
            ],
            ollama_model="llava",
        )
    else:
        raise HTTPException(
            status_code=503,
            detail=(
                "No AI provider is configured. Set GEMINI_API_KEY, OLLAMA_HOST, "
                "or OPENAI_API_KEY."
            ),
        )
    return {"analysis": result}


@app.post("/api/ai/code", dependencies=[Depends(require_developer_code)])
def generate_or_refactor_code(request: CodeRequest) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior software engineer. Analyze the supplied source, then "
                "write, refactor, or repair it according to the request. Preserve intended "
                "behavior, explain assumptions briefly, and return the complete revised "
                "code in a fenced code block. Never claim to have run code or tests."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {request.language}\nTask: {request.task}\n\n"
                f"Source code:\n```{request.language}\n{request.code}\n```"
            ),
        },
    ]
    return {"result": _call_llm(messages, ollama_model="codellama")}


@app.get(
    "/api/agents/tools",
    dependencies=[Depends(require_developer_code)],
)
def list_agent_tools() -> dict[str, list[dict[str, Any]]]:
    return {
        "tools": [
            {
                "name": "prompt.process",
                "description": "Generate a response from the configured language model.",
                "endpoint": "POST /api/ai/prompt",
                "execution_mode": "authenticated_api",
            },
            {
                "name": "image.analyze",
                "description": "Analyze validated base64 image context with a vision-capable model.",
                "endpoint": "POST /api/ai/images/analyze",
                "execution_mode": "authenticated_api",
            },
            {
                "name": "code.assist",
                "description": "Generate, refactor, or repair source code without executing it.",
                "endpoint": "POST /api/ai/code",
                "execution_mode": "authenticated_api",
            },
        ]
    }


@app.post(
    "/api/agents/tasks",
    dependencies=[Depends(require_developer_code)],
)
def create_agent_task(request: AgentTaskCreate) -> dict[str, Any]:
    _ensure_schema()
    task_id = str(uuid.uuid4())
    steps = [step.model_dump(mode="json") for step in request.steps]
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO agent_tasks (task_id, session_id, goal, status, steps)
                    VALUES (%s, %s, %s, 'pending', %s::jsonb)
                    RETURNING task_id::text, session_id, goal, status, steps,
                              created_at, updated_at
                    """,
                    (task_id, request.session_id, request.goal, json.dumps(steps)),
                )
                row = cursor.fetchone()
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not create agent task.")
            raise HTTPException(
                status_code=503,
                detail="Could not save the agent task to Neon.",
            ) from exc
    return _agent_task_payload(row)


@app.get(
    "/api/agents/tasks",
    dependencies=[Depends(require_developer_code)],
)
def list_agent_tasks(
    session_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, list[dict[str, Any]]]:
    _ensure_schema()
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT task_id::text, session_id, goal, status, steps,
                           created_at, updated_at
                    FROM agent_tasks
                    WHERE session_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cursor.fetchall()
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not list agent tasks.")
            raise HTTPException(
                status_code=503,
                detail="Could not load agent tasks from Neon.",
            ) from exc
    return {"tasks": [_agent_task_payload(row) for row in rows]}


@app.get(
    "/api/agents/tasks/{task_id}",
    dependencies=[Depends(require_developer_code)],
)
def get_agent_task(
    task_id: uuid.UUID,
    session_id: str = Query(min_length=1, max_length=128),
) -> dict[str, Any]:
    _ensure_schema()
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT task_id::text, session_id, goal, status, steps,
                           created_at, updated_at
                    FROM agent_tasks
                    WHERE task_id = %s AND session_id = %s
                    """,
                    (str(task_id), session_id),
                )
                row = cursor.fetchone()
            connection.commit()
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not load agent task.")
            raise HTTPException(
                status_code=503,
                detail="Could not load agent task from Neon.",
            ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Agent task not found.")
    return _agent_task_payload(row)


@app.patch(
    "/api/agents/tasks/{task_id}",
    dependencies=[Depends(require_developer_code)],
)
def update_agent_task_state(
    task_id: uuid.UUID,
    request: AgentTaskStateUpdate,
) -> dict[str, Any]:
    _ensure_schema()
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT task_id::text, session_id, goal, status, steps,
                           created_at, updated_at
                    FROM agent_tasks
                    WHERE task_id = %s AND session_id = %s
                    FOR UPDATE
                    """,
                    (str(task_id), request.session_id),
                )
                row = cursor.fetchone()
                if row is None:
                    connection.commit()
                    raise HTTPException(status_code=404, detail="Agent task not found.")
                current_status = AgentTaskStatus(row[3])
                transitions = {
                    AgentTaskStatus.pending: {
                        AgentTaskStatus.running,
                        AgentTaskStatus.cancelled,
                    },
                    AgentTaskStatus.running: {
                        AgentTaskStatus.completed,
                        AgentTaskStatus.failed,
                        AgentTaskStatus.cancelled,
                    },
                }
                if request.status not in transitions.get(current_status, set()):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Cannot transition task from {current_status} to {request.status}.",
                    )
                cursor.execute(
                    """
                    UPDATE agent_tasks
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s AND session_id = %s
                    RETURNING task_id::text, session_id, goal, status, steps,
                              created_at, updated_at
                    """,
                    (request.status.value, str(task_id), request.session_id),
                )
                updated = cursor.fetchone()
            connection.commit()
        except HTTPException:
            connection.rollback()
            raise
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not update agent task state.")
            raise HTTPException(
                status_code=503,
                detail="Could not update agent task in Neon.",
            ) from exc
    return _agent_task_payload(updated)


@app.patch(
    "/api/agents/tasks/{task_id}/steps/{step_index}",
    dependencies=[Depends(require_developer_code)],
)
def update_agent_task_step(
    task_id: uuid.UUID,
    step_index: int,
    request: AgentStepUpdate,
    session_id: str = Query(min_length=1, max_length=128),
) -> dict[str, Any]:
    _ensure_schema()
    with _database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status, steps
                    FROM agent_tasks
                    WHERE task_id = %s AND session_id = %s
                    FOR UPDATE
                    """,
                    (str(task_id), session_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Agent task not found.")
                if AgentTaskStatus(row[0]) not in {
                    AgentTaskStatus.pending,
                    AgentTaskStatus.running,
                }:
                    raise HTTPException(
                        status_code=409,
                        detail="Steps cannot be changed after the task is terminal.",
                    )
                steps = row[1]
                if step_index < 0 or step_index >= len(steps):
                    raise HTTPException(status_code=404, detail="Agent task step not found.")
                steps[step_index]["status"] = request.status.value
                steps[step_index]["result"] = request.result
                cursor.execute(
                    """
                    UPDATE agent_tasks
                    SET steps = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s AND session_id = %s
                    RETURNING task_id::text, session_id, goal, status, steps,
                              created_at, updated_at
                    """,
                    (json.dumps(steps), str(task_id), session_id),
                )
                updated = cursor.fetchone()
            connection.commit()
        except HTTPException:
            connection.rollback()
            raise
        except PsycopgError as exc:
            connection.rollback()
            logger.exception("Could not update agent task step.")
            raise HTTPException(
                status_code=503,
                detail="Could not update agent task step in Neon.",
            ) from exc
    return _agent_task_payload(updated)


def _agent_task_payload(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "task_id": row[0],
        "session_id": row[1],
        "goal": row[2],
        "status": row[3],
        "steps": row[4],
        "created_at": row[5].isoformat(),
        "updated_at": row[6].isoformat(),
    }


handler = Mangum(app, lifespan="auto")
