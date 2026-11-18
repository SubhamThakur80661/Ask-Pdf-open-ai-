# OpCit Backend

FastAPI service for OpCit, providing isolated in-memory PDF indexing with FAISS and grounded question answering with Hugging Face Inference API.

## Requirements

- Python 3.12+
- Hugging Face API Token (optional for self-host fallback, otherwise supplied per request via `X-HF-Token` header)

## Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Verify health check:
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status": "ok"}
   ```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins | `http://localhost:5173,http://localhost:3000` |
| `HUGGINGFACEHUB_API_TOKEN` | Optional server-side fallback Hugging Face token | None (tokens are client-provided) |
| `PORT` | Port number assigned by the host | `8000` |

### Production CORS Note

When deploying the frontend to Vercel, add your production Vercel domain (e.g. `https://your-opcit-app.vercel.app`) to the `CORS_ORIGINS` environment variable in your backend hosting service (Render).

## API Endpoints

- `GET /health` — Health check (public, returns `{"status": "ok"}`)
- `POST /api/sessions` — Create a new session (`{"session_id": "<uuid>"}`)
- `GET /api/validate-token` — Validate Hugging Face token via `X-HF-Token` header
- `POST /api/sessions/{id}/upload` — Upload and index PDFs (max 10 files, 25 MB each)
- `POST /api/sessions/{id}/ask` — Ask question and retrieve grounded answers with page citations
- `GET /api/sessions/{id}/documents` — List indexed documents in session
- `DELETE /api/sessions/{id}/documents/{doc_id}` — Remove document and re-index remaining
- `DELETE /api/sessions/{id}` — Terminate session and release in-memory index
