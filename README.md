# OpCit - Ask your documents. Every answer cites its page.

## What OpCit is
OpCit is a professional full-stack web application that allows users to chat with multiple PDF documents. It provides accurate answers by citing the exact page and document from which the information was extracted.

## Features
- Upload multiple PDF documents (up to 10 files, 25 MB each)
- Ask natural language questions about the content of your PDFs
- Get answers with precise citations (document name, page number, verbatim excerpt)
- Supports a fallback chain of Hugging Face models for robust responses
- User-provided Hugging Face API token, never stored server-side
- Clean, accessible, and responsive user interface
- Chat history persists within the browser session
- Export chat conversation as Markdown

## Stack
- **Frontend**: React, Vite, TypeScript, Tailwind CSS, Motion.dev (for micro-motion), Lucide icons
- **Backend**: FastAPI (Python 3.12)
- **RAG (Retrieval-Augmented Generation)**:
    - PDF text extraction: PyPDF2
    - Text chunking: 1000 characters with 200 overlap
    - Embeddings: Hugging Face API (`sentence-transformers/all-MiniLM-L6-v2`)
    - Vector store: FAISS (in-memory per session)
    - LLM Chat Completions: Qwen2.5-7B-Instruct (via `huggingface_hub.InferenceClient`), with fallbacks Qwen/Qwen2.5-1.5B-Instruct, HuggingFaceH4/zephyr-7b-beta, microsoft/Phi-3-mini-4k-instruct

## Architecture Diagram (in words)
Frontend (React app) sends requests to Backend (FastAPI).
Backend processes requests:
1. Receives PDF uploads from Frontend.
2. Extracts text, chunks it, and generates embeddings via Hugging Face API.
3. Builds an in-memory FAISS index.
4. For questions, retrieves relevant chunks from FAISS.
5. Sends chunks and question to Hugging Face LLM for chat completions.
6. Returns answer and citations to Frontend.

## Live Demo
Current Streamlit app: https://ymcrqleh7utgmxcff3txpg.streamlit.app/
New React app: https://opcit.vercel.app

## Local Setup

### Prerequisites
- Python 3.12+
- Node.js 20+ and npm
- A Hugging Face API token (get one at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### Backend Setup (`backend/`)
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   ./venv/Scripts/activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your Hugging Face API token. You can set it as an environment variable (for development only, never commit your token):
   ```bash
   # On Windows (Command Prompt):
   set HUGGINGFACEHUB_API_TOKEN="hf_YOUR_TOKEN_HERE"
   # On Windows (PowerShell):
   $env:HUGGINGFACEHUB_API_TOKEN="hf_YOUR_TOKEN_HERE"
   # On macOS/Linux:
   export HUGGINGFACEHUB_API_TOKEN="hf_YOUR_TOKEN_HERE"
   ```
   Alternatively, the backend also accepts the token via the `X-HF-Token` header from the frontend, which is the recommended secure approach for production.
5. Run the FastAPI application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   The backend will be available at `http://localhost:8000`.

### Frontend Setup (`frontend/`)
1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Configure the backend API URL. Create a `.env.local` file in the `frontend/` directory with the following content (if your backend is running on `http://localhost:8000`):
   ```
   VITE_API_URL=http://localhost:8000
   ```
   If `VITE_API_URL` is not set, the frontend will show a "Backend not configured yet" banner.
4. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`.

## Hugging Face Token Step
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Create a new token with "Read" permissions.
3. Copy the token.
4. In the OpCit frontend, open the "Settings" modal (usually via a button in the header) and paste your Hugging Face API token there. Click "Validate" to ensure it's correct. The token is stored only in your browser's local storage and sent with each request to the backend, never stored server-side.

## Render Deploy Notes
The `render.yaml` file at the repository root defines the infrastructure for deploying the backend to Render. For detailed manual setup steps, refer to `backend/RENDER_SETUP.md`.

## License
MIT License

Copyright (c) 2026 Subham Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
