import { useRef, useEffect, useState, type FC, type FormEvent, type KeyboardEvent } from 'react';
import { Send, Download, Trash2, Sparkles, Loader2 } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import type { Message, DocumentInfo } from '../types';

interface ChatAreaProps {
  messages: Message[];
  onSendMessage: (question: string) => void;
  isLoading: boolean;
  documents: DocumentInfo[];
  hasToken: boolean;
  onClearChat: () => void;
  onOpenSettings: () => void;
  disabled: boolean;
}

export const ChatArea: FC<ChatAreaProps> = ({
  messages,
  onSendMessage,
  isLoading,
  documents,
  hasToken,
  onClearChat,
  onOpenSettings,
  disabled,
}) => {
  const [inputQuestion, setInputQuestion] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const suggestedPrompts = [
    'Summarize the core takeaways from the documents',
    'What key dates, deadlines, or timelines are specified?',
    'What methodology, data sources, or evidence are cited?',
    'What primary recommendations or conclusions are reached?',
  ];

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Adjust textarea height automatically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [inputQuestion]);

  const handleSubmit = (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!inputQuestion.trim() || isLoading || disabled) return;
    onSendMessage(inputQuestion.trim());
    setInputQuestion('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExportChat = () => {
    if (messages.length === 0) return;

    let md = `# OpCit Research Notes & Citations\n\n`;
    md += `*Exported on ${new Date().toLocaleString()}*\n\n`;
    md += `## Document Corpus\n\n`;
    documents.forEach((doc) => {
      md += `- **${doc.name}** (${doc.pages} pages, ${doc.chunks} indexed passages)\n`;
    });
    md += `\n---\n\n## Inquiry Transcript\n\n`;

    messages.forEach((msg) => {
      if (msg.role === 'user') {
        md += `### Question:\n${msg.content}\n\n`;
      } else {
        md += `### Answer (via ${msg.model || 'OpCit'}):\n${msg.content}\n\n`;
        if (msg.citations && msg.citations.length > 0) {
          md += `#### Citations:\n`;
          msg.citations.forEach((c) => {
            md += `- **${c.doc}**, Page ${c.page}:\n  > "${c.excerpt}"\n\n`;
          });
        }
      }
      md += `---\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `opcit-inquiry-${new Date().toISOString().slice(0, 10)}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const hasDocs = documents.length > 0;

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] max-w-4xl mx-auto w-full">
      {/* Chat Top Controls Bar */}
      <div className="px-4 sm:px-6 py-2.5 border-b border-[#e7e4db] flex items-center justify-between bg-paper-50/80 backdrop-blur-sm z-10">
        <div className="text-xs text-ink-600 font-medium">
          {hasDocs ? (
            <span>
              Questioning <strong className="text-ink-900">{documents.length}</strong> {documents.length === 1 ? 'document' : 'documents'}
            </span>
          ) : (
            <span>Awaiting document upload</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <>
              <button
                onClick={handleExportChat}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-ink-700 bg-white border border-paper-300 hover:bg-paper-100 rounded-md transition-colors focus-ring shadow-subtle"
                aria-label="Export conversation as Markdown"
                title="Download Markdown summary"
              >
                <Download className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Export notes</span>
              </button>
              <button
                onClick={onClearChat}
                className="p-1 text-ink-400 hover:text-rose-700 hover:bg-rose-50 rounded transition-colors focus-ring"
                aria-label="Clear chat history"
                title="Clear chat"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
        {messages.length === 0 ? (
          /* Empty State */
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-6 max-w-xl mx-auto my-auto">
            <div className="space-y-2">
              <h2 className="text-2xl font-serif font-bold text-ink-950">
                Ask your documents.
              </h2>
              <p className="text-sm text-ink-600 leading-relaxed font-sans">
                Every generated response cites its exact source page with verbatim quotes. All embeddings and search happen in-memory.
              </p>
            </div>

            {/* Step flow cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full text-left">
              <button
                type="button"
                onClick={onOpenSettings}
                className="p-3.5 rounded-lg bg-white border border-paper-300 shadow-subtle space-y-1 text-left hover:border-paper-400 focus-ring transition-all"
              >
                <span className="text-[10px] font-mono font-bold text-scholar px-1.5 py-0.5 rounded bg-paper-100 border border-paper-200">
                  01
                </span>
                <h3 className="text-xs font-semibold text-ink-900 pt-1">Add Token</h3>
                <p className="text-[11px] text-ink-500 leading-normal">
                  Provide your free Hugging Face API token in Settings.
                </p>
              </button>

              <div className="p-3.5 rounded-lg bg-white border border-paper-300 shadow-subtle space-y-1">
                <span className="text-[10px] font-mono font-bold text-scholar px-1.5 py-0.5 rounded bg-paper-100 border border-paper-200">
                  02
                </span>
                <h3 className="text-xs font-semibold text-ink-900 pt-1">Upload PDFs</h3>
                <p className="text-[11px] text-ink-500 leading-normal">
                  Drop up to 10 PDF documents to parse and index into FAISS.
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-white border border-paper-300 shadow-subtle space-y-1">
                <span className="text-[10px] font-mono font-bold text-scholar px-1.5 py-0.5 rounded bg-paper-100 border border-paper-200">
                  03
                </span>
                <h3 className="text-xs font-semibold text-ink-900 pt-1">Cite & Inquire</h3>
                <p className="text-[11px] text-ink-500 leading-normal">
                  Ask targeted questions and inspect verified page excerpts.
                </p>
              </div>
            </div>

            {/* Prompt suggestions if docs already loaded */}
            {hasDocs && (
              <div className="w-full pt-4 space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-ink-700">
                  <Sparkles className="w-3.5 h-3.5 text-scholar" />
                  <span>Suggested inquiries:</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {suggestedPrompts.map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSendMessage(prompt)}
                      className="text-left px-3 py-1.5 rounded-full text-xs font-medium bg-white hover:bg-paper-100 text-ink-800 border border-paper-300 shadow-subtle hover:border-paper-400 transition-all focus-ring"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Message List */
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}

            {/* Thinking / In-Flight Indicator */}
            {isLoading && (
              <div className="py-4 -mx-4 px-4 sm:-mx-6 sm:px-6 bg-white/40 border-y border-[#ede9df]">
                <div className="flex items-start gap-3.5 max-w-3xl">
                  <div className="w-7 h-7 rounded bg-ink-900 text-paper-50 flex items-center justify-center font-serif text-xs font-bold border border-ink-800 shadow-subtle">
                    <span>Op</span>
                  </div>
                  <div className="space-y-1.5 pt-1">
                    <div className="flex items-center gap-2 text-xs font-semibold text-ink-950">
                      <span>OpCit Assistant</span>
                      <span className="text-[11px] text-ink-500 font-normal">
                        Retrieving relevant passages and verifying citations...
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-ink-600">
                      <Loader2 className="w-4 h-4 animate-spin text-scholar" />
                      <span className="italic font-serif">Consulting indexed pages...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Suggested chips row above composer when conversation is active */}
      {messages.length > 0 && hasDocs && !isLoading && (
        <div className="px-4 sm:px-6 py-1.5 overflow-x-auto flex items-center gap-2 bg-paper-50/90 border-t border-paper-200">
          <span className="text-[10px] font-mono uppercase text-ink-500 flex-shrink-0">
            Suggested:
          </span>
          {suggestedPrompts.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => onSendMessage(prompt)}
              className="text-xs whitespace-nowrap px-2.5 py-1 rounded-full bg-white hover:bg-paper-100 text-ink-700 border border-paper-300 shadow-subtle transition-all focus-ring flex-shrink-0"
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* Composer Input */}
      <div className="p-4 sm:p-6 border-t border-[#e7e4db] bg-paper-50">
        <form onSubmit={handleSubmit} className="relative">
          <div className="relative flex items-end rounded-xl border border-paper-300 bg-white shadow-subtle focus-within:border-ink-800 focus-within:ring-1 focus-within:ring-ink-800 transition-all overflow-hidden">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputQuestion}
              onChange={(e) => setInputQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !hasToken
                  ? 'Add your Hugging Face token in Settings first...'
                  : !hasDocs
                  ? 'Upload and process PDFs to ask questions...'
                  : 'Ask a question about your documents (e.g., What are the key takeaways?)...'
              }
              disabled={disabled || isLoading || !hasDocs || !hasToken}
              className="w-full resize-none px-4 py-3 text-sm text-ink-950 placeholder:text-ink-400 bg-transparent focus:outline-none min-h-[48px] max-h-[180px] leading-relaxed"
            />

            <div className="p-2 flex items-center gap-1.5 flex-shrink-0">
              <button
                type="submit"
                disabled={disabled || isLoading || !hasDocs || !hasToken || !inputQuestion.trim()}
                className="p-2 rounded-lg bg-ink-900 text-white hover:bg-ink-950 disabled:opacity-40 disabled:pointer-events-none transition-all focus-ring shadow-subtle"
                aria-label="Send question"
                title="Send question (Enter)"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between mt-2 px-1 text-[11px] text-ink-500 font-mono">
            <span>
              {hasDocs ? 'Grounded exclusively on uploaded PDF content' : 'Awaiting document upload'}
            </span>
            <span className="hidden sm:inline">
              <kbd className="px-1 py-0.5 rounded bg-paper-200 border border-paper-300 text-[10px]">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-paper-200 border border-paper-300 text-[10px]">Shift+Enter</kbd> newline
            </span>
          </div>
        </form>
      </div>
    </div>
  );
};
