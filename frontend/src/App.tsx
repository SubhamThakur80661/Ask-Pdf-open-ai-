import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { StatusBanner } from './components/StatusBanner';
import { UploadZone } from './components/UploadZone';
import { DocumentLibrary } from './components/DocumentLibrary';
import { ChatArea } from './components/ChatArea';
import { TokenSettingsModal } from './components/TokenSettingsModal';
import { AmbientBackground } from './components/AmbientBackground';
import type { BackendStatus, DocumentInfo, Message, UploadStage } from './types';
import {
  checkHealth,
  createSession,
  uploadDocuments,
  askQuestion,
  deleteDocument,
  deleteSession,
  getStoredToken,
  setStoredToken,
  clearStoredToken,
  isApiConfigured,
} from './api';
import { BookOpen, MessageSquare, PanelLeft, PanelLeftClose } from 'lucide-react';

export function App() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [isRetrying, setIsRetrying] = useState<boolean>(false);
  const [token, setToken] = useState<string>('');
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);

  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStage, setUploadStage] = useState<UploadStage>('idle');
  const [isLoadingQuestion, setIsLoadingQuestion] = useState<boolean>(false);
  const [isDeletingDoc, setIsDeletingDoc] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Mobile layout tab switcher ('documents' or 'chat')
  const [mobileTab, setMobileTab] = useState<'documents' | 'chat'>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);

  // Initial load
  useEffect(() => {
    const savedToken = getStoredToken();
    setToken(savedToken);

    verifyBackend();
  }, []);

  const verifyBackend = useCallback(async () => {
    if (!isApiConfigured()) {
      setBackendStatus('unconfigured');
      return;
    }
    setIsRetrying(true);
    try {
      const isHealthy = await checkHealth();
      setBackendStatus(isHealthy ? 'online' : 'offline');
    } catch {
      setBackendStatus('offline');
    } finally {
      setIsRetrying(false);
    }
  }, []);

  const handleSaveToken = (newToken: string) => {
    setToken(newToken);
    setStoredToken(newToken);
    setErrorMessage(null);
  };

  const handleClearToken = () => {
    setToken('');
    clearStoredToken();
  };

  const ensureSession = async (): Promise<string> => {
    if (sessionId) return sessionId;
    const newId = await createSession();
    setSessionId(newId);
    return newId;
  };

  const handleFilesSelected = (newFiles: File[]) => {
    setSelectedFiles((prev) => [...prev, ...newFiles]);
    setErrorMessage(null);
  };

  const handleRemoveStagedFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleProcessDocuments = async () => {
    if (selectedFiles.length === 0 || isUploading) return;
    if (!token) {
      setIsSettingsOpen(true);
      return;
    }

    setIsUploading(true);
    setUploadStage('reading');
    setErrorMessage(null);

    try {
      const activeSessionId = await ensureSession();

      // Show real processing progression
      setUploadStage('extracting');
      // Yield to UI loop
      await new Promise((r) => setTimeout(r, 200));

      setUploadStage('embedding');
      const res = await uploadDocuments(activeSessionId, selectedFiles);

      setUploadStage('indexing');
      setDocuments(res.documents);
      setSelectedFiles([]);
      setUploadStage('done');

      // On mobile, switch view to chat once documents are ready
      setMobileTab('chat');
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to process documents.');
    } finally {
      setIsUploading(false);
      setUploadStage('idle');
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    if (!sessionId || isDeletingDoc) return;
    setIsDeletingDoc(true);
    setErrorMessage(null);
    try {
      const remaining = await deleteDocument(sessionId, docId);
      setDocuments(remaining);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to remove document.');
    } finally {
      setIsDeletingDoc(false);
    }
  };

  const handleClearAll = async () => {
    if (!sessionId) {
      setSelectedFiles([]);
      setDocuments([]);
      setMessages([]);
      return;
    }
    try {
      await deleteSession(sessionId);
    } catch {
      // Best-effort
    }
    setSessionId(null);
    setSelectedFiles([]);
    setDocuments([]);
    setMessages([]);
    setErrorMessage(null);
  };

  const handleSendMessage = async (question: string) => {
    if (!sessionId || !hasToken || isLoadingQuestion || !question.trim()) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoadingQuestion(true);
    setErrorMessage(null);

    try {
      const res = await askQuestion(sessionId, question);
      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        model: res.model,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const assistantErrorMsg: Message = {
        id: `assistant-err-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${err.message || 'Could not retrieve answer. Please try again.'}`,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantErrorMsg]);
    } finally {
      setIsLoadingQuestion(false);
    }
  };

  const hasToken = Boolean(token.trim());
  const isActionDisabled = backendStatus === 'offline' || backendStatus === 'unconfigured';

  return (
    <div className="min-h-screen flex flex-col bg-[#faf9f5] text-ink-900 font-sans selection:bg-amber-100">
      <AmbientBackground />
      <div className="relative z-10 flex flex-col flex-1 min-h-0 animate-rise">
      {/* Header */}
      <Header
        status={backendStatus}
        isRetrying={isRetrying}
        onRetry={verifyBackend}
        hasToken={hasToken}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Status / Warning Banner */}
      <StatusBanner
        backendStatus={backendStatus}
        isRetrying={isRetrying}
        onRetry={verifyBackend}
        hasToken={hasToken}
        onOpenSettings={() => setIsSettingsOpen(true)}
        errorMessage={errorMessage}
        onDismissError={() => setErrorMessage(null)}
      />

      {/* Mobile Tab Switcher */}
      <div className="md:hidden flex border-b border-white/60 bg-white/70 backdrop-blur-xl sticky top-16 z-20">
        <button
          onClick={() => setMobileTab('documents')}
          className={`flex-1 py-2.5 text-xs font-semibold flex items-center justify-center gap-1.5 border-b-2 transition-all ${
            mobileTab === 'documents'
              ? 'border-ink-900 text-ink-950 bg-white'
              : 'border-transparent text-ink-600 hover:text-ink-900'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          <span>Documents ({documents.length})</span>
        </button>
        <button
          onClick={() => setMobileTab('chat')}
          className={`flex-1 py-2.5 text-xs font-semibold flex items-center justify-center gap-1.5 border-b-2 transition-all ${
            mobileTab === 'chat'
              ? 'border-ink-900 text-ink-950 bg-white'
              : 'border-transparent text-ink-600 hover:text-ink-900'
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          <span>Inquiry Chat {messages.length > 0 && `(${messages.length})`}</span>
        </button>
      </div>

      {/* Main Two-Column App Shell */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Document Management */}
        <aside
          className={`w-full md:w-80 lg:w-96 border-r border-white/60 bg-white/55 backdrop-blur-xl overflow-y-auto p-4 sm:p-5 space-y-6 flex-shrink-0 transition-all ${
            mobileTab === 'documents' ? 'block' : 'hidden md:block'
          } ${isSidebarOpen ? '' : 'md:hidden'}`}
          aria-label="Document Management Sidebar"
        >
          <div className="space-y-1">
            <h2 className="text-base font-serif font-bold text-ink-950 flex items-center justify-between">
              <span>Document Corpus</span>
              <span className="text-xs font-sans font-normal text-ink-500">
                {documents.length} / 10 files
              </span>
            </h2>
            <p className="text-xs text-ink-600 leading-normal">
              PDFs are processed locally in server memory. Text is chunked with page tracking and indexed into FAISS.
            </p>
          </div>

          {/* Upload Zone */}
          <UploadZone
            onFilesSelected={handleFilesSelected}
            selectedFiles={selectedFiles}
            onRemoveFile={handleRemoveStagedFile}
            onProcess={handleProcessDocuments}
            isProcessing={isUploading}
            stage={uploadStage}
            hasToken={hasToken}
            onOpenSettings={() => setIsSettingsOpen(true)}
            disabled={isActionDisabled}
          />

          {/* Processed Document Library */}
          <DocumentLibrary
            documents={documents}
            onDeleteDocument={handleDeleteDocument}
            onClearAll={handleClearAll}
            isDeleting={isDeletingDoc}
            disabled={isActionDisabled || isUploading}
          />
        </aside>

        {/* Right / Main Pane: Chat & Citations */}
        <section
          className={`flex-1 flex flex-col min-w-0 bg-transparent ${
            mobileTab === 'chat' ? 'flex' : 'hidden md:flex'
          }`}
          aria-label="Inquiry and Citations Workspace"
        >
          {/* Desktop Sidebar Toggle */}
          <div className="hidden md:flex items-center justify-between px-4 py-1 border-b border-white/60 bg-white/50 backdrop-blur-md text-[11px] text-ink-500">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="inline-flex items-center gap-1.5 text-ink-600 hover:text-ink-950 focus-ring rounded p-1 transition-colors"
              title={isSidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
              aria-label={isSidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
            >
              {isSidebarOpen ? (
                <>
                  <PanelLeftClose className="w-3.5 h-3.5" />
                  <span>Hide corpus panel</span>
                </>
              ) : (
                <>
                  <PanelLeft className="w-3.5 h-3.5" />
                  <span>Show corpus panel</span>
                </>
              )}
            </button>
            <span className="font-mono text-[10px]">OpCit v1.0 · RAG with Grounded Citations</span>
          </div>

          <ChatArea
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoadingQuestion}
            documents={documents}
            hasToken={hasToken}
            onClearChat={() => setMessages([])}
            onOpenSettings={() => setIsSettingsOpen(true)}
            disabled={isActionDisabled}
          />
        </section>
      </main>

      {/* Hugging Face Token Settings Modal */}
      <TokenSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        token={token}
        onSaveToken={handleSaveToken}
        onClearToken={handleClearToken}
      />
      </div>
    </div>
  );
}

export default App;
