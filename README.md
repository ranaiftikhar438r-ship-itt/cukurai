# Cukur AI

## Configuration

Configure these environment variables in the deployment environment (and in a
local `.env` file when running locally):

- `GEMINI_API_KEY` — Google AI Studio API key. Gemini requests use the
  production `gemini-2.5-flash` model.
- `NEON_DATABASE_URL` — optional for chat; required for persistent chat
  history, developer sessions, and agent task state.

`OLLAMA_HOST` and `OPENAI_API_KEY` remain supported as alternative AI providers
when `GEMINI_API_KEY` is not configured. Gemini takes precedence when its key
is present.

The chat and AI/agent API routes require the `X-Developer-Code` header. The
browser dashboard requests the code at sign-in and holds it only in memory for
the current page session. Developer authorization accepts only the fixed
server-side code `Cukur301r`; no environment override is supported.

Open the browser dashboard at `/` or `/app`. Chat messages and their image previews, source links,
and search queries are saved in browser local storage for the current session;
use **Clear chat** to remove that local history. Gemini requests enable Google
Search grounding and chat responses include any returned grounding metadata,
web source links, and search queries.

Chat replies stream to the dashboard as they are generated. Browser speech input
and optional spoken replies use the Web Speech APIs when supported. Fenced
Python, JSON, and text code blocks include a local download action.

Real-time Niko voice uses a one-use, short-lived Gemini Live token issued by the
developer-authorized `/api/live/token` endpoint; the browser connects directly to
Gemini for microphone streaming. Medical-image and health responses are instructed
to distinguish observations from possibilities, provide general information only,
verify physical product labels, and include a clinician disclaimer.