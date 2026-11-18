"""
OpCit Backend API
FastAPI backend for document Q&A with grounded citations and in-memory FAISS indexing.
"""

import os
import io
import time
import uuid
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient, HfApi

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opcit")

# Constants
MAX_FILES_PER_SESSION = 10
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SESSION_TTL_SECONDS = 7200  # 2 hours idle TTL
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FALLBACK_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
]


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    pages: int
    chunks: int


class ChunkItem:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


class SessionState:
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.last_active: float = time.time()
        self.documents: Dict[str, DocumentInfo] = {}
        self.chunks: List[ChunkItem] = []
        self.vectorstore: Optional[FAISS] = None

    def touch(self):
        self.last_active = time.time()


# In-memory storage for active sessions
sessions: Dict[str, SessionState] = {}


def cleanup_expired_sessions():
    """Remove sessions that have been idle longer than the TTL."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in expired:
        logger.info(f"Expiring idle session: {sid}")
        sessions.pop(sid, None)


def get_token(x_hf_token: Optional[str] = Header(None, alias="X-HF-Token")) -> str:
    """Resolve token from header or environment variable fallback."""
    token = (x_hf_token or "").strip()
    if not token:
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Hugging Face API token is required. Add your token in Settings to start.",
        )
    return token


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure clean state
    logger.info("OpCit backend starting up...")
    yield
    # Shutdown: clear sessions
    sessions.clear()
    logger.info("OpCit backend shut down.")


app = FastAPI(
    title="OpCit API",
    description="Ask your documents. Every answer cites its page.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
raw_cors = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
cors_origins = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    doc: str
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: str


class UploadSummary(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_pages: int
    total_chunks: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_pages: int
    total_chunks: int


@app.get("/health")
def health_check():
    """Health check endpoint. No auth required."""
    return {"status": "ok"}


@app.post("/api/sessions")
def create_session():
    """Create a new isolated session."""
    cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionState(session_id)
    return {"session_id": session_id}


@app.get("/api/validate-token")
def validate_token(x_hf_token: Optional[str] = Header(None, alias="X-HF-Token")):
    """
    Validate Hugging Face API token using whoami.
    Never echoes the token back.
    """
    token = (x_hf_token or "").strip()
    if not token:
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()

    if not token:
        return {"valid": False}

    try:
        api = HfApi()
        api.whoami(token=token)
        return {"valid": True}
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return {"valid": False}


@app.get("/api/sessions/{session_id}/documents", response_model=DocumentListResponse)
def get_documents(session_id: str):
    """List indexed documents for a session."""
    cleanup_expired_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired. Please refresh or create a new session.",
        )
    session.touch()

    doc_list = list(session.documents.values())
    total_pages = sum(d.pages for d in doc_list)
    total_chunks = sum(d.chunks for d in doc_list)

    return DocumentListResponse(
        documents=doc_list,
        total_documents=len(doc_list),
        total_pages=total_pages,
        total_chunks=total_chunks,
    )


@app.post("/api/sessions/{session_id}/upload", response_model=UploadSummary)
async def upload_documents(
    session_id: str,
    files: List[UploadFile] = File(...),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
):
    """
    Upload and process PDFs:
    - Enforce PDF format, max 10 files per session, max 25 MB per file.
    - Extract text per page with PyPDF2, keeping page numbers.
    - Chunk with metadata {doc, doc_id, page}.
    - Embed via HuggingFaceEndpointEmbeddings and index with FAISS.
    """
    cleanup_expired_sessions()
    token = get_token(x_hf_token)

    session = sessions.get(session_id)
    if not session:
        # Create session if it doesn't exist yet
        session = SessionState(session_id)
        sessions[session_id] = session
    session.touch()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded. Please upload at least one PDF file.",
        )

    current_count = len(session.documents)
    if current_count + len(files) > MAX_FILES_PER_SESSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_FILES_PER_SESSION} files allowed per session. Current: {current_count}, Attempted to add: {len(files)}.",
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    new_chunks: List[ChunkItem] = []
    new_doc_infos: List[DocumentInfo] = []

    for file in files:
        # Validate file extension
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for '{filename}'. Only PDF files are supported.",
            )

        # Read content with 25MB boundary check
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{filename}' exceeds maximum allowed size of 25 MB ({len(content) / (1024 * 1024):.1f} MB).",
            )

        # Validate PDF header
        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{filename}' is not a valid PDF document.",
            )

        doc_id = str(uuid.uuid4())[:8]

        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse PDF '{filename}': {str(e)}",
            )

        total_pages = len(reader.pages)
        doc_chunk_count = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from {filename} page {page_idx}: {e}")
                page_text = ""

            # Sanitize text
            page_text = page_text.encode("ascii", "ignore").decode("ascii")
            if not page_text.strip():
                continue

            page_chunks = splitter.split_text(page_text)
            for chunk_str in page_chunks:
                chunk_clean = chunk_str.strip()
                if chunk_clean:
                    new_chunks.append(
                        ChunkItem(
                            text=chunk_clean,
                            metadata={
                                "doc": filename,
                                "doc_id": doc_id,
                                "page": page_idx,
                            },
                        )
                    )
                    doc_chunk_count += 1

        if doc_chunk_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No selectable text could be extracted from '{filename}'. Please ensure the PDF contains digital text rather than scanned images.",
            )

        doc_info = DocumentInfo(
            doc_id=doc_id,
            name=filename,
            pages=total_pages,
            chunks=doc_chunk_count,
        )
        session.documents[doc_id] = doc_info
        new_doc_infos.append(doc_info)

    session.chunks.extend(new_chunks)

    # Build / rebuild the FAISS vector store
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=token,
        )
        all_texts = [c.text for c in session.chunks]
        all_metas = [c.metadata for c in session.chunks]
        session.vectorstore = FAISS.from_texts(
            texts=all_texts,
            embedding=embeddings,
            metadatas=all_metas,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg or "invalid token" in err_msg:
            # Revert document additions on auth failure
            for doc in new_doc_infos:
                session.documents.pop(doc.doc_id, None)
            session.chunks = [c for c in session.chunks if c.metadata["doc_id"] not in [d.doc_id for d in new_doc_infos]]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Hugging Face API token or unauthorized access. Please check your token in Settings.",
            )
        logger.error(f"Vector store creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )

    all_docs = list(session.documents.values())
    return UploadSummary(
        documents=all_docs,
        total_documents=len(all_docs),
        total_pages=sum(d.pages for d in all_docs),
        total_chunks=sum(d.chunks for d in all_docs),
    )


@app.post("/api/sessions/{session_id}/ask", response_model=AskResponse)
def ask_question(
    session_id: str,
    req: AskRequest,
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
):
    """
    Ask a question against indexed documents:
    - Retrieve top 4 relevant chunks.
    - Construct grounded prompt.
    - Query Hugging Face with fallback model chain.
    - Return answer, citations with real doc names, pages, and excerpts.
    """
    cleanup_expired_sessions()
    token = get_token(x_hf_token)

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired. Please upload your documents again.",
        )
    session.touch()

    if not session.vectorstore or not session.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents indexed in this session. Please upload PDFs first.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    # Retrieve top 4 chunks
    try:
        retrieved_docs = session.vectorstore.similarity_search(question, k=4)
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search document index: {str(e)}",
        )

    if not retrieved_docs:
        return AskResponse(
            answer="No relevant content found in the uploaded documents to answer your question.",
            citations=[],
            model="none",
        )

    # Format context and citations
    context_blocks = []
    citations: List[Citation] = []

    for d in retrieved_docs:
        doc_name = d.metadata.get("doc", "Document")
        page_num = d.metadata.get("page", 1)
        raw_text = d.page_content.strip()

        # Build clean verbatim excerpt (around 220 characters)
        excerpt = raw_text[:240].rsplit(" ", 1)[0] if len(raw_text) > 240 else raw_text
        if len(raw_text) > len(excerpt):
            excerpt += "..."

        citations.append(
            Citation(
                doc=doc_name,
                page=page_num,
                excerpt=excerpt,
            )
        )
        context_blocks.append(f"[Document: {doc_name} | Page {page_num}]\n{raw_text}")

    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are OpCit, an exacting research assistant. "
        "Answer the user's question using ONLY the provided context excerpts. "
        "If the answer cannot be determined from the context, state honestly: "
        "'The provided documents do not contain information to answer this question.' "
        "Do not invent facts, speculate, or draw from outside knowledge. "
        "Be concise, clear, and objective."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer:"

    client = InferenceClient(api_key=token)

    last_error = None
    successful_model = None
    answer_text = None

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"Attempting inference with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            answer_text = response.choices[0].message.content.strip()
            successful_model = model
            break
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "unauthorized" in err_str:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Hugging Face API token. Please check your token in Settings.",
                )
            logger.warning(f"Model {model} failed: {e}. Trying fallback...")
            last_error = e
            continue

    if not answer_text or not successful_model:
        logger.error(f"All models failed. Last error: {last_error}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"All Hugging Face models failed to generate an answer. Error: {str(last_error)}",
        )

    return AskResponse(
        answer=answer_text,
        citations=citations,
        model=successful_model,
    )


@app.delete("/api/sessions/{session_id}/documents/{doc_id}", response_model=DocumentListResponse)
def delete_document(
    session_id: str,
    doc_id: str,
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
):
    """
    Remove one document from the session and rebuild FAISS index with remaining chunks.
    """
    cleanup_expired_sessions()
    token = get_token(x_hf_token)

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired.",
        )
    session.touch()

    if doc_id not in session.documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in this session.",
        )

    # Remove document
    session.documents.pop(doc_id)
    # Filter out chunks
    session.chunks = [c for c in session.chunks if c.metadata.get("doc_id") != doc_id]

    # Rebuild vector store or clear if empty
    if session.chunks:
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
                model=EMBEDDING_MODEL,
                huggingfacehub_api_token=token,
            )
            all_texts = [c.text for c in session.chunks]
            all_metas = [c.metadata for c in session.chunks]
            session.vectorstore = FAISS.from_texts(
                texts=all_texts,
                embedding=embeddings,
                metadatas=all_metas,
            )
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to rebuild index after document deletion: {str(e)}",
            )
    else:
        session.vectorstore = None

    doc_list = list(session.documents.values())
    return DocumentListResponse(
        documents=doc_list,
        total_documents=len(doc_list),
        total_pages=sum(d.pages for d in doc_list),
        total_chunks=sum(d.chunks for d in doc_list),
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Drop a session entirely."""
    if session_id in sessions:
        sessions.pop(session_id, None)
    return {"status": "ok", "message": f"Session '{session_id}' deleted."}
