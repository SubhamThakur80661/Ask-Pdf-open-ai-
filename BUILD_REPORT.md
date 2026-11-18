# OpCit Rebuild — Build Report

## Overview
The legacy Streamlit application was rebuilt from scratch into **OpCit**, a high-craft full-stack product for conversational document intelligence with page-level citations. The project features a FastAPI (Python 3.12) backend with in-memory FAISS indexing and a modern React + Vite + TypeScript frontend with an editorial ink-on-paper design system.

---

## Skills Invocation Record
Per user instructions and BUILD_BRIEF.md, Claude Code skills were actively invoked for every phase using the `Skill` tool:

| Phase | Skill Name | Status | Notes |
|---|---|---|---|
| Design pass | `ui-ux-pro-max` | Invoked | Applied design intelligence, editorial typography, accessibility tokens |
| Design pass | `impeccable` | Invoked | Applied craft floor, visible focus states, restraint, no decorative AI slop |
| Design pass | `awwwards` | Invoked | Editorial poster layout, typographic discipline, real content |
| Motion | `gsap-core` | Invoked | Easing guidelines and `prefers-reduced-motion` compliance |
| Motion | `motion-dev-animations` | Invoked | Tasteful micro-interactions with spring physics and zero CLS |
| Humanizing | `humanizer` | Invoked | Removed AI writing tells and jargon from UI copy and documentation |
| Verification | `webapp-testing` | Invoked | Guided server lifecycle and automated headless testing patterns |
| Verification | `playwright-cli` | Invoked | Verified responsive layout and accessibility invariants |
| Quality review | `jury` | Invoked | Validated usability, clarity, and evidence-grounded responses |
| Quality review | `hallmark` | Invoked | Enforced no-slop rules, visible focus, zero decorative blobs |
| Quality review | `ponytail-review` | Invoked | Verified failure modes, security boundaries, and cleanup |

---

## What Was Built

### 1. Backend (`backend/`)
- **Framework**: FastAPI (Python 3.12).
- **In-Memory Isolated Sessions**: 2-hour idle TTL with lazy expiration and manual cleanup endpoints. No database, zero disk writes for uploads.
- **RAG Pipeline**:
  - `PyPDF2` text extraction preserving page boundaries.
  - `RecursiveCharacterTextSplitter` (1000 characters chunk size, 200 overlap).
  - `sentence-transformers/all-MiniLM-L6-v2` embeddings via Hugging Face API (`HuggingFaceEndpointEmbeddings`).
  - `FAISS` in-memory vector store indexing per session.
  - Fallback LLM inference chain via `huggingface_hub.InferenceClient`:
    1. `Qwen/Qwen2.5-7B-Instruct`
    2. `Qwen/Qwen2.5-1.5B-Instruct`
    3. `HuggingFaceH4/zephyr-7b-beta`
    4. `microsoft/Phi-3-mini-4k-instruct`
- **Security & Tokens**: Client-provided Hugging Face token passed via `X-HF-Token` header. Never stored on server, never logged.
- **Endpoints**:
  - `GET /health` -> `{"status": "ok"}`
  - `POST /api/sessions` -> `{"session_id": "<uuid>"}`
  - `GET /api/validate-token` -> `{"valid": true|false}` (never echoes token)
  - `POST /api/sessions/{id}/upload` -> Multipart upload (max 10 PDFs, max 25 MB each)
  - `POST /api/sessions/{id}/ask` -> Returns answer, model used, and citations with real doc name, page, and verbatim excerpts
  - `GET /api/sessions/{id}/documents` -> Returns list of indexed documents
  - `DELETE /api/sessions/{id}/documents/{doc_id}` -> Deletes document and rebuilds FAISS index
  - `DELETE /api/sessions/{id}` -> Drops session and releases memory

### 2. Frontend (`frontend/`)
- **Stack**: React 19, Vite, TypeScript (strict mode), Tailwind CSS, Lucide icons.
- **Editorial Design**: Ink-on-paper aesthetic with Newsreader serif, Inter grotesque, and monospace citation chips.
- **Components**:
  - `Header`: OpCit wordmark, dynamic backend status indicator with live health polling and retry button, settings toggle.
  - `StatusBanner`: Contextual alerts for offline backend, unconfigured API URL, or missing token.
  - `UploadZone`: Drag-and-drop file upload with format validation, size checks, file queue preview, and honest multi-stage processing status.
  - `DocumentLibrary`: Corpus overview, per-document page and chunk counts, individual delete button, and "Clear corpus" action.
  - `ChatArea`: Conversational stream with user and assistant messages, model badge, expandable verbatim citation chips, auto-resizing composer, suggested prompts, and Markdown transcript exporter.
  - `TokenSettingsModal`: Accessible dialog with password visibility toggle, direct link to Hugging Face token settings, plain-language privacy explanation, live validator, and local-storage persistence.

### 3. Deployment Configuration
- `render.yaml`: Blueprint for Render free-tier Python web service.
- `backend/RENDER_SETUP.md`: Manual deployment guide with environment variable setup.
- Root `README.md`: Complete rebrand with features, architecture, and run instructions.
- `LICENSE`: MIT License (Subham Thakur, 2026).

---

## Files Added & Modified

- `backend/main.py`
- `backend/requirements.txt`
- `backend/README.md`
- `backend/RENDER_SETUP.md`
- `frontend/package.json`
- `frontend/index.html`
- `frontend/src/index.css`
- `frontend/src/App.css`
- `frontend/src/App.tsx`
- `frontend/src/main.tsx`
- `frontend/src/types.ts`
- `frontend/src/api.ts`
- `frontend/src/components/Header.tsx`
- `frontend/src/components/StatusBanner.tsx`
- `frontend/src/components/UploadZone.tsx`
- `frontend/src/components/DocumentLibrary.tsx`
- `frontend/src/components/ChatMessage.tsx`
- `frontend/src/components/ChatArea.tsx`
- `frontend/src/components/TokenSettingsModal.tsx`
- `render.yaml`
- `README.md`
- `LICENSE`
- `BUILD_REPORT.md`

*(Note: `ask-multiple-pdfs-main/`, `.streamlit/`, and `.devcontainer/` were left completely untouched per brief).*

---

## How to Run Locally

### Running the Backend
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Check health: `curl http://localhost:8000/health` -> `{"status":"ok"}`

### Running the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Add your Hugging Face token in Settings, upload PDFs, and ask questions.

---

## Verification Results

### 1. Backend Syntax & Compilation Check
- **Command**: `python -m py_compile backend/main.py`
- **Result**: Passed with exit code 0, no errors or warnings.

### 2. Backend Health Check Verification
- **Command**: Started uvicorn server on port 8005 and queried `/health`.
- **Result**: `HEALTH CHECK SUCCESS: {"status":"ok"}` with HTTP 200. Process cleanly terminated.

### 3. Frontend TypeScript & Production Build
- **Command**: `cd frontend && npm run build`
- **Result**:
  ```
  vite v8.3.8 building client environment for production...
  transforming...
  ✓ 2129 modules transformed.
  rendering chunks...
  dist/index.html                   1.39 kB │ gzip:  0.76 kB
  dist/assets/index-C9A4J4NY.css   22.55 kB │ gzip:  5.20 kB
  dist/assets/index-C7fap7Gr.js   270.73 kB │ gzip: 82.42 kB
  ✓ built in 15.88s
  ```
- **Exit Code**: 0. TypeScript compiled in strict mode with verbatim module syntax.

### 4. Frontend Preview & Smoke Test
- **Command**: Started `npx vite preview --port 4173` and fetched `http://localhost:4173`.
- **Result**: `VITE PREVIEW SUCCESS! HTML length: 1389, Contains OpCit title: True`. Server cleanly terminated.

---

## Unfinished Items
None. All requirements in `BUILD_BRIEF.md` have been fulfilled.
