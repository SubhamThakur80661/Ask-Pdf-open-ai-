import type { AskResponse, DocumentInfo, UploadSummaryResponse } from './types';

// API Base URL from environment or default
export const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl !== undefined && envUrl !== '') {
    return envUrl.replace(/\/+$/, '');
  }
  // Default to localhost:8000
  return 'http://localhost:8000';
};

export const isApiConfigured = (): boolean => {
  // If explicitly set to empty string, it's considered unconfigured
  const envUrl = import.meta.env.VITE_API_URL;
  return envUrl !== '';
};

// Token storage in browser localStorage only
const TOKEN_STORAGE_KEY = 'opcit_hf_token';

export const getStoredToken = (): string => {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || '';
  } catch {
    return '';
  }
};

export const setStoredToken = (token: string): void => {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token.trim());
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // LocalStorage quota or access issue
  }
};

export const clearStoredToken = (): void => {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // Ignore
  }
};

const getHeaders = (token?: string, isJson = true): HeadersInit => {
  const headers: Record<string, string> = {};
  if (isJson) {
    headers['Content-Type'] = 'application/json';
  }
  const activeToken = token || getStoredToken();
  if (activeToken) {
    headers['X-HF-Token'] = activeToken;
  }
  return headers;
};

export async function checkHealth(): Promise<boolean> {
  if (!isApiConfigured()) {
    return false;
  }
  try {
    const res = await fetch(`${getApiBaseUrl()}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'ok';
  } catch {
    return false;
  }
}

export async function validateToken(tokenToTest: string): Promise<boolean> {
  try {
    const res = await fetch(`${getApiBaseUrl()}/api/validate-token`, {
      method: 'GET',
      headers: {
        'X-HF-Token': tokenToTest.trim(),
      },
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) return false;
    const data = await res.json();
    return Boolean(data?.valid);
  } catch {
    return false;
  }
}

export async function createSession(): Promise<string> {
  const res = await fetch(`${getApiBaseUrl()}/api/sessions`, {
    method: 'POST',
    headers: getHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to create session (${res.status})`);
  }
  const data = await res.json();
  return data.session_id;
}

export async function uploadDocuments(
  sessionId: string,
  files: File[]
): Promise<UploadSummaryResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const res = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}/upload`, {
    method: 'POST',
    headers: getHeaders(undefined, false),
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to upload documents (${res.status})`);
  }
  return res.json();
}

export async function askQuestion(
  sessionId: string,
  question: string
): Promise<AskResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}/ask`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to get answer (${res.status})`);
  }
  return res.json();
}

export async function getDocuments(sessionId: string): Promise<DocumentInfo[]> {
  const res = await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}/documents`, {
    method: 'GET',
    headers: getHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to retrieve documents (${res.status})`);
  }
  const data = await res.json();
  return data.documents || [];
}

export async function deleteDocument(
  sessionId: string,
  docId: string
): Promise<DocumentInfo[]> {
  const res = await fetch(
    `${getApiBaseUrl()}/api/sessions/${sessionId}/documents/${docId}`,
    {
      method: 'DELETE',
      headers: getHeaders(),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to delete document (${res.status})`);
  }
  const data = await res.json();
  return data.documents || [];
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${getApiBaseUrl()}/api/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: getHeaders(),
  }).catch(() => {
    // Best-effort cleanup
  });
}
