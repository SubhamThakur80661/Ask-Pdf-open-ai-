# OpCit — Rebuild Brief (Ask-Pdf-open-ai-)

You are rebuilding a Streamlit "chat with multiple PDFs" app into a professional
full-stack product. Work autonomously in this repo on the `rebuild` branch.
Do not ask questions. Do not merge to main. Do not touch `ask-multiple-pdfs-main/`,
`.streamlit/`, or `.devcontainer/` — the old Streamlit app stays live until the
replacement is verified.

## Brand

- Name: **OpCit** (from the scholarly abbreviation op. cit., "in the work cited").
  Tagline: "Ask your documents. Every answer cites its page."
- Use the name in the UI, page titles, and README. The GitHub repo name stays
  as-is; this is a display-name rebrand only.
- No fake testimonials, no fake stats, no placeholder content in production.

## Skills — ACTIVE INVOCATION IS MANDATORY (user instruction)

You MUST actively invoke the installed Claude Code skills for each phase using
the Skill tool. Do not work bare. Skills live under
C:\Users\shubh\.claude\skills. Invoke at least these, at the phase named:

- Design pass (design system, tokens, typography, layout):
  `ui-ux-pro-max`, `impeccable`, `awwwards`
- Motion (only where it genuinely fits, tasteful micro-motion):
  `gsap-core`, `motion-dev-animations`
- Copy and code humanizing (remove AI tells from all user-facing copy):
  `humanizer`
- Verification (build checks, flow tests, browser checks if available):
  `webapp-testing`, `playwright-cli`
- Quality review (final pass before you report done):
  `jury`, `hallmark`, `ponytail-review`

If a skill fails to load, note it in BUILD_REPORT.md and continue with the
others — do not stall the build. In BUILD_REPORT.md, record per phase which
skills you actually invoked (and which failed to load, if any).

## Architecture (decided — do not revisit)

- `frontend/` — React + Vite + TypeScript + Tailwind CSS. Deployed to Vercel
  with root directory `frontend/`. No router needed (single-page app).
- `backend/` — FastAPI (Python 3.12). Deployed to Render free tier as a
  persistent web service. `render.yaml` Blueprint at repo root.
- Same RAG approach as the original app: PDFs -> PyPDF2 text extraction ->
  chunking (1000 chars, 200 overlap) -> Hugging Face API embeddings
  (`sentence-transformers/all-MiniLM-L6-v2`) -> FAISS in-memory index ->
  Qwen2.5-7B-Instruct via `huggingface_hub.InferenceClient` chat completions,
  with fallbacks `Qwen/Qwen2.5-1.5B-Instruct`, `HuggingFaceH4/zephyr-7b-beta`,
  `microsoft/Phi-3-mini-4k-instruct` (same order as the original).
- No local model anywhere. Embeddings and chat are Hugging Face API calls.
- The user's Hugging Face token is THEIR key: entered in a settings UI in the
  browser, stored only in browser localStorage, sent to the backend per request
  in an `X-HF-Token` header, never stored server-side, never logged. The backend
  also accepts a `HUGGINGFACEHUB_API_TOKEN` env var as a self-host fallback.
  NEVER invent a token. NEVER fake an answer. Without a token the app must show
  an honest warning and refuse to process/ask.

## Backend (`backend/`) — FastAPI

Files: `backend/main.py` (or a small package), `backend/requirements.txt`
(pinned versions), `backend/README.md` (short: run locally, env vars).

Endpoints (all under `/api` except health):
- `GET /health` -> `{"status": "ok"}`. No auth, no token needed.
- `POST /api/sessions` -> `{"session_id": "<uuid>"}`.
- `POST /api/sessions/{id}/upload` — multipart `files` (PDF only, max 10 files
  per session, max 25 MB each; reject others with a clear 400). Extract text
  per page with PyPDF2, KEEPING page numbers. Chunk with metadata
  `{doc, doc_id, page}`. Embed via HuggingFaceEndpointEmbeddings (same model as
  original) using the `X-HF-Token` header. Build a FAISS index in memory for
  the session. Response: list of documents `{doc_id, name, pages, chunks}` plus
  totals. If the token is missing/invalid -> 401 with a clear message.
- `POST /api/sessions/{id}/ask` — JSON `{"question": "..."}`. Retrieve top 4
  chunks, build a grounded prompt ("answer ONLY from the context; if the answer
  is not in the context, say so"), call the LLM with the fallback chain.
  Response: `{"answer": "...", "citations": [{"doc": name, "page": n,
  "excerpt": "..."}], "model": "<model that answered>"}`. Citations must come
  from the retrieved chunks actually used — real doc names and real page
  numbers, with a short verbatim excerpt from the chunk. 401 without a token,
  400 with no documents indexed.
- `GET /api/sessions/{id}/documents` -> list of indexed documents.
- `DELETE /api/sessions/{id}/documents/{doc_id}` -> remove one document and
  rebuild the session FAISS index from the remaining chunks (keep chunk texts
  in memory per session so rebuild is possible).
- `DELETE /api/sessions/{id}` -> drop the session entirely.
- `GET /api/validate-token` — header `X-HF-Token`; check it with
  `huggingface_hub.HfApi.whoami` (or an equivalent light call). Return
  `{"valid": true/false}`. Never echo the token back.

Rules:
- Sessions live in memory with a 2-hour idle TTL cleanup (background task or
  lazy expiry). No database. No files kept on disk after processing (delete
  any temp uploads promptly).
- CORS: allow origins from env `CORS_ORIGINS` (comma-separated, default
  `http://localhost:5173,http://localhost:3000`). The Vercel production origin
  will be added later via env var — document this in backend/README.md.
- Errors return JSON `{"detail": "..."}` with honest, human messages.
- `render.yaml` at repo root: one free-tier web service, name `opcit-backend`,
  env python, rootDir `backend`, buildCommand `pip install -r requirements.txt`,
  startCommand `uvicorn main:app --host 0.0.0.0 --port $PORT`, healthCheckPath
  `/health`, envVars: `CORS_ORIGINS` (value set after the frontend URL is
  known — put a placeholder and a comment), `HUGGINGFACEHUB_API_TOKEN` (sync:
  false, so it is set manually in Render and never committed).
- Also write precise manual Render setup steps in `backend/RENDER_SETUP.md`
  (service type, repo, branch, root dir, build/start commands, env vars) in
  case Blueprint import is not used.
- Verify locally: `python -m py_compile` on all backend files, and start
  uvicorn briefly to confirm `GET /health` returns 200 (curl it). Do not start
  any long-running server and leave it running — kill it after the check.

## Frontend (`frontend/`) — React + Vite + TypeScript + Tailwind

Scaffold with Vite (react-ts template). Tailwind CSS. Use hand-built components
in shadcn/ui style (clean, accessible, Tailwind-based) — do NOT pull in a heavy
component library or a paid asset. Motion.dev for micro-motion only (fades,
subtle list transitions); respect `prefers-reduced-motion` (disable motion).
Lucide icons if useful (lucide-react is MIT).

API base URL from `VITE_API_URL` env var (default `http://localhost:8000`).
If `VITE_API_URL` is empty/unset at build time, the app must still render and
show an honest "Backend not configured yet" status banner — never crash, never
fake a working state.

Layout (single-page app, desktop-first but fully responsive):
- Header: OpCit wordmark (text-based, no fake logo asset), backend status pill
  (Checking / Online / Offline / Not configured — from a real `/health` poll on
  load and a manual Retry), Settings button (opens token settings).
- Left/main area: chat. Right panel (or left sidebar on desktop, stacked on
  mobile): documents + upload. Or a clean two-column app shell — your design
  call, guided by ui-ux-pro-max. It must look like a professional team built it:
  strong hierarchy, consistent spacing and tokens, one clear primary action at
  every step.
- Upload: drag-and-drop zone + file picker, multiple PDFs, per-file name/size,
  remove-before-processing, "Process documents" button. Processing state must
  be HONEST: show "Processing…" with a spinner and the real stage if the
  backend reports one; never show fake percentages or a fake progress bar.
  (Backend upload is a single request; a spinner + stage text is fine.)
- Document library: list of processed docs with page counts and chunk counts,
  remove-one (calls the delete endpoint), "Clear all" (deletes the session and
  starts fresh). Show a supported-formats note: "PDF only, up to 10 files,
  25 MB each."
- Chat: message list with user/assistant bubbles, a composer (textarea, Enter
  to send, Shift+Enter newline), a typing indicator while waiting. Under each
  assistant answer, citation chips: "Document name — p. N". Clicking a chip
  expands the verbatim excerpt. Suggested questions appear once documents are
  ready (real prompts the user can tap, e.g. "Summarize the main points",
  "What are the key dates mentioned?"). "Export chat" downloads the
  conversation as Markdown (real export of the actual messages).
- Token settings (modal or panel): password-style input, a link to
  https://huggingface.co/settings/tokens, plain-language explanation ("Your token
  stays in your browser. It is sent with each request so the app can call
  Hugging Face on your behalf, and is never stored on the server."), a
  Validate button (calls `/api/validate-token`, shows valid/invalid honestly),
  and a Clear token button.
- Empty/error states, all honest: no token -> banner "Add your Hugging Face
  token in Settings to start." No documents -> upload prompt. Backend offline
  or not configured -> banner with Retry. Question with no answer in the docs
  -> the backend says so; the UI shows it plainly, no invented content.
- Chat history persists in the browser session only (React state is fine;
  localStorage persistence of messages is a nice-to-have, not required).
  Never persist the token anywhere except localStorage, and never log it.

Build must pass: `npm run build` with TypeScript strict, no errors.
Also run a quick smoke check of the built app if feasible (vite preview +
curl the HTML). Do not leave any dev server running afterwards.

## README.md (repo root)

Rewrite properly: what OpCit is, features (real ones only), stack, architecture
diagram in words (frontend -> backend -> Hugging Face), live demo section
(put the Streamlit URL https://ymcrqleh7utgmxcff3txpg.streamlit.app/ as the
current live app, and a clearly-marked `TODO: Vercel frontend URL` line for the
new frontend — do NOT invent a URL), local setup (frontend + backend), the
Hugging Face token step (how to get one, where to paste it), Render deploy
notes (point at backend/RENDER_SETUP.md), and license note (MIT — add a LICENSE
file with the MIT text, copyright Subham Thakur, 2026).

## Design quality rules (non-negotiable)

- No generic AI layouts (no purple-gradient hero + 3 identical cards), no
  fake testimonials/stats, no lorem ipsum, no decorative blobs. Editorial,
  document-inspired design: ink-on-paper feel, one accent color used with
  restraint, excellent typography.
- Accessible: visible focus states, keyboard-usable controls, aria labels on
  icon-only buttons, sufficient contrast, responsive down to 360px wide.
- Reduced motion: all animation off under prefers-reduced-motion.

## Git

- Work on the `rebuild` branch (already checked out). Commit in logical chunks
  with clear messages (e.g. `feat(backend): FastAPI RAG service with FAISS`,
  `feat(frontend): OpCit chat UI`, `docs: OpCit README and Render setup`).
- Forward commits only. No history rewrites, no force-push, no touching main.
- Push the `rebuild` branch to origin when the build passes (git push from
  this machine works). If push fails, report the exact error in the log and
  stop — do not try alternate remotes or credentials.

## Done means

1. `npm run build` passes in `frontend/`.
2. `python -m py_compile` passes on all backend files and `GET /health`
   returns 200 from a brief local uvicorn run.
3. `render.yaml` + `backend/RENDER_SETUP.md` exist and are accurate.
4. README.md rewritten with the OpCit brand and a TODO for the Vercel URL.
5. Everything committed and pushed to the `rebuild` branch.
6. Write a short `BUILD_REPORT.md` at repo root: what was built, files added,
   how to run locally, verification results (exact commands + outcomes), and
   anything left unfinished. Never claim a test passed that you did not run.

Start by reading the original app at `ask-multiple-pdfs-main/app.py` so the RAG
behavior matches, then build. Work through the whole brief in this one run.
