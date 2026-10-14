# Cukur AI

## Configuration

Configure these environment variables in the deployment environment (and in a
local `.env` file when running locally):

- `GEMINI_API_KEY` — Google AI Studio API key. The default model is
  `gemini-2.5-flash`; override it with `GEMINI_MODEL` if needed.
- `NEON_DATABASE_URL` — optional for chat; required for persistent chat
  history, developer sessions, and agent task state.

`OLLAMA_HOST` and `OPENAI_API_KEY` remain supported as alternative AI providers
when `GEMINI_API_KEY` is not configured. Gemini takes precedence when its key
is present.

The chat and AI/agent API routes require the `X-Developer-Code` header. The
browser dashboard requests the code at sign-in and holds it only in memory for
the current page session. Developer authorization accepts only the fixed
server-side code `Cukur301r`; no environment override is supported.

Open the browser dashboard at `/app`. The root route `/` remains the JSON
service status endpoint.