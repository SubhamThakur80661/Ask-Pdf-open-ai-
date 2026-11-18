import { useState, type FC } from 'react';
import { BookOpen, ChevronDown, ChevronUp, Copy, Check, Cpu, User } from 'lucide-react';
import type { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: FC<ChatMessageProps> = ({ message }) => {
  const [expandedCitationIndex, setExpandedCitationIndex] = useState<number | null>(null);
  const [isCopied, setIsCopied] = useState(false);

  const toggleCitation = (idx: number) => {
    setExpandedCitationIndex(expandedCitationIndex === idx ? null : idx);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const isAssistant = message.role === 'assistant';

  return (
    <article
      className={`py-4 transition-colors ${
        isAssistant ? 'bg-white/60 -mx-4 px-4 sm:-mx-6 sm:px-6 border-y border-[#ede9df]' : ''
      }`}
      aria-label={`${isAssistant ? 'Assistant response' : 'User question'}`}
    >
      <div className="flex items-start gap-3.5 max-w-3xl">
        {/* Avatar */}
        <div
          className={`w-7 h-7 rounded flex items-center justify-center flex-shrink-0 mt-0.5 text-xs font-semibold shadow-subtle ${
            isAssistant
              ? 'bg-ink-900 text-paper-50 font-serif border border-ink-800'
              : 'bg-paper-200 text-ink-700 border border-paper-300'
          }`}
        >
          {isAssistant ? <span>Op</span> : <User className="w-3.5 h-3.5" />}
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-3 min-w-0">
          {/* Header row (role & timestamp/model) */}
          <div className="flex items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-ink-950">
                {isAssistant ? 'OpCit Assistant' : 'You'}
              </span>
              {isAssistant && message.model && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-paper-100 text-ink-600 border border-paper-200">
                  <Cpu className="w-2.5 h-2.5" />
                  <span>{message.model}</span>
                </span>
              )}
            </div>

            {/* Copy Button */}
            {isAssistant && (
              <button
                onClick={handleCopy}
                className="p-1 text-ink-400 hover:text-ink-800 rounded transition-colors focus-ring"
                aria-label="Copy answer to clipboard"
                title="Copy answer"
              >
                {isCopied ? (
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            )}
          </div>

          {/* Text Body */}
          <div className="text-sm leading-relaxed text-ink-900 font-sans whitespace-pre-wrap selection:bg-amber-100">
            {message.content}
          </div>

          {/* Citation Chips & Excerpts */}
          {isAssistant && message.citations && message.citations.length > 0 && (
            <div className="pt-2 border-t border-paper-200/80 space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-ink-600">
                <BookOpen className="w-3.5 h-3.5 text-scholar" />
                <span>Cited passages ({message.citations.length}):</span>
              </div>

              {/* Chips */}
              <div className="flex flex-wrap gap-1.5">
                {message.citations.map((cite, idx) => {
                  const isExpanded = expandedCitationIndex === idx;
                  return (
                    <button
                      key={`${cite.doc}-${cite.page}-${idx}`}
                      onClick={() => toggleCitation(idx)}
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono border transition-all focus-ring ${
                        isExpanded
                          ? 'bg-scholar text-white border-scholar shadow-sm'
                          : 'bg-white hover:bg-paper-100 text-ink-800 border-paper-300'
                      }`}
                      aria-expanded={isExpanded}
                      aria-label={`Citation: ${cite.doc}, page ${cite.page}. Click to expand verbatim excerpt.`}
                    >
                      <span className="truncate max-w-[160px] sm:max-w-[220px]">
                        {cite.doc}
                      </span>
                      <span className="font-semibold text-scholar-200">
                        p. {cite.page}
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-3 h-3 text-white" />
                      ) : (
                        <ChevronDown className="w-3 h-3 text-ink-400" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Expanded Excerpt Callout */}
              {expandedCitationIndex !== null && message.citations[expandedCitationIndex] && (
                <div className="mt-2 p-3.5 rounded-lg bg-paper-100 border border-paper-300 shadow-subtle space-y-1.5 transition-all">
                  <div className="flex items-center justify-between text-xs font-mono text-ink-700 pb-1 border-b border-paper-200">
                    <span className="font-semibold text-scholar">
                      {message.citations[expandedCitationIndex].doc} — Page{' '}
                      {message.citations[expandedCitationIndex].page}
                    </span>
                    <span className="text-[10px] text-ink-500 uppercase tracking-wider">
                      Verbatim Document Excerpt
                    </span>
                  </div>
                  <blockquote className="text-xs font-serif italic text-ink-900 leading-relaxed border-l-2 border-scholar pl-3 my-1">
                    "{message.citations[expandedCitationIndex].excerpt}"
                  </blockquote>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </article>
  );
};
