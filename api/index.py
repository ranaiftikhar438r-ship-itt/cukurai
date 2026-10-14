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
from enum import StrEnum
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
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

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    developer_mode: bool = False
    images: list[ImageContext] = Field(default_factory=list, max_length=4)


class ImageContext(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    media_type: str = Field(pattern=r"^image/(?:png|jpeg|webp|gif)$")
    data: str = Field(min_length=1, max_length=8_000_000)
    description: str | None = Field(default=None, max_length=1000)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=256)


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

    return _call_llm_completion(
        messages,
        ollama_model="llama3",
        gemini_images=gemini_images,
    )


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

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    payload["tools"] = [{"google_search": {}}]

    try:
        response = requests.post(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.gemini_model}:generateContent"
            ),
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=(5, 60),
        )
        response.raise_for_status()
        response_body = response.json()
        if not isinstance(response_body, dict):
            raise ValueError("Gemini response must be a JSON object.")
        candidates = response_body.get("candidates", [])
        if not candidates:
            logger.error("Gemini returned no response candidates.")
            raise HTTPException(
                status_code=502,
                detail="Gemini did not return a response. Check the prompt and model configuration.",
            )
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ValueError("Gemini response candidate must be a JSON object.")
        parts = candidate.get("content", {}).get("parts", [])
        text = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        grounding_metadata = candidate.get("groundingMetadata")
        if not isinstance(grounding_metadata, dict):
            grounding_metadata = None
        sources_by_uri: dict[str, dict[str, str]] = {}
        if grounding_metadata:
            grounding_chunks = grounding_metadata.get("groundingChunks", [])
            if not isinstance(grounding_chunks, list):
                grounding_chunks = []
            for chunk in grounding_chunks:
                if not isinstance(chunk, dict):
                    continue
                web_source = chunk.get("web")
                if not isinstance(web_source, dict):
                    continue
                uri = web_source.get("uri")
                title = web_source.get("title")
                if (
                    isinstance(uri, str)
                    and uri.startswith(("https://", "http://"))
                ):
                    sources_by_uri[uri] = {
                        "title": title if isinstance(title, str) and title else uri,
                        "uri": uri,
                    }
        raw_queries = grounding_metadata.get("webSearchQueries", []) if grounding_metadata else []
        web_search_queries = (
            [query for query in raw_queries if isinstance(query, str)]
            if isinstance(raw_queries, list)
            else []
        )
    except HTTPException:
        raise
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        logger.exception("Gemini rejected the request with HTTP %s.", status_code)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Gemini rejected the request (HTTP {status_code}). "
                "Check the Gemini API key, model, and request configuration."
            ),
        ) from exc
    except (
        requests.RequestException,
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
def root() -> dict[str, str]:
    return {"message": "Welcome to Cukur AI", "status": "ok"}


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


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    _: None = Depends(require_developer_code),
) -> dict[str, Any]:
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
    if not settings.neon_database_url:
        completion = _call_llm_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Cukur AI, a clear, capable virtual assistant. "
                        "Do not claim to have executed actions or accessed systems. "
                        "Developer Mode is an authorization state, not a grant of tools."
                    ),
                },
                {"role": "user", "content": request.message},
            ],
            ollama_model="llama3",
            gemini_images=gemini_images,
        )
        return {
            "session_id": request.session_id,
            "reply": completion.text,
            "developer_mode_active": developer_mode_active,
            "auth_required": False,
            "persistence_enabled": False,
            **_completion_response_fields(completion),
        }

    _ensure_schema()
    completion = _generate_reply(request.session_id, request.message, gemini_images)
    _save_messages(
        request.session_id,
        request.message,
        completion.text,
        developer_mode_active,
    )
    return {
        "session_id": request.session_id,
        "reply": completion.text,
        "developer_mode_active": developer_mode_active,
        "auth_required": False,
        "persistence_enabled": True,
        **_completion_response_fields(completion),
    }


@app.post("/api/ai/prompt", dependencies=[Depends(require_developer_code)])
def process_prompt(request: PromptRequest) -> dict[str, str]:
    messages = [
        {"role": "system", "content": request.system_prompt},
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
                    "content": "Analyze the supplied images and answer accurately. State uncertainty.",
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
                "content": "Analyze the supplied images and answer accurately. State uncertainty.",
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
                    "content": "Analyze the supplied images and answer accurately. State uncertainty.",
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
