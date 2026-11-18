export interface DocumentInfo {
  doc_id: string;
  name: string;
  pages: number;
  chunks: number;
}

export interface Citation {
  doc: string;
  page: number;
  excerpt: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  model?: string;
  timestamp: string;
}

export type BackendStatus = 'checking' | 'online' | 'offline' | 'unconfigured';

export type UploadStage =
  | 'idle'
  | 'reading'
  | 'extracting'
  | 'embedding'
  | 'indexing'
  | 'done';

export interface UploadSummaryResponse {
  documents: DocumentInfo[];
  total_documents: number;
  total_pages: number;
  total_chunks: number;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  model: string;
}
