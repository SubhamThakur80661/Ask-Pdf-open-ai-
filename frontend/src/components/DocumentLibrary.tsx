import type { FC } from 'react';
import { FileText, Trash2, Layers, BookCheck } from 'lucide-react';
import type { DocumentInfo } from '../types';

interface DocumentLibraryProps {
  documents: DocumentInfo[];
  onDeleteDocument: (docId: string) => void;
  onClearAll: () => void;
  isDeleting: boolean;
  disabled: boolean;
}

export const DocumentLibrary: FC<DocumentLibraryProps> = ({
  documents,
  onDeleteDocument,
  onClearAll,
  isDeleting,
  disabled,
}) => {
  const totalPages = documents.reduce((acc, d) => acc + d.pages, 0);
  const totalChunks = documents.reduce((acc, d) => acc + d.chunks, 0);

  if (documents.length === 0) {
    return (
      <div className="rounded-lg border border-paper-300 bg-white p-6 text-center space-y-2">
        <div className="w-10 h-10 rounded-full bg-paper-100 border border-paper-200 text-ink-500 mx-auto flex items-center justify-center">
          <BookCheck className="w-5 h-5 text-ink-600" />
        </div>
        <h3 className="text-sm font-serif font-semibold text-ink-900">
          No indexed documents
        </h3>
        <p className="text-xs text-ink-500 max-w-xs mx-auto">
          Upload and process PDF files above to begin questioning your texts.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header & Stats */}
      <div className="flex items-center justify-between pb-2 border-b border-paper-200">
        <div>
          <h3 className="text-xs font-serif font-bold uppercase tracking-wider text-ink-900">
            Document Corpus ({documents.length})
          </h3>
          <p className="text-[11px] text-ink-500">
            {totalPages} {totalPages === 1 ? 'page' : 'pages'} · {totalChunks} indexed passages
          </p>
        </div>
        <button
          onClick={onClearAll}
          disabled={disabled || isDeleting}
          className="text-[11px] font-medium text-rose-700 hover:text-rose-900 hover:underline disabled:opacity-50 transition-colors focus-ring rounded"
        >
          Clear corpus
        </button>
      </div>

      {/* Document List */}
      <ul className="space-y-2" aria-label="Indexed documents">
        {documents.map((doc) => (
          <li
            key={doc.doc_id}
            className="flex items-start justify-between gap-3 p-3 rounded-lg bg-white border border-paper-300 shadow-subtle hover:border-paper-400 transition-all"
          >
            <div className="flex items-start gap-2.5 min-w-0">
              <div className="p-1.5 rounded bg-paper-100 text-scholar border border-paper-200 mt-0.5 flex-shrink-0">
                <FileText className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-ink-950 truncate" title={doc.name}>
                  {doc.name}
                </p>
                <div className="flex items-center gap-2 mt-1 text-[11px] text-ink-500 font-mono">
                  <span>{doc.pages} {doc.pages === 1 ? 'page' : 'pages'}</span>
                  <span>·</span>
                  <span className="flex items-center gap-1">
                    <Layers className="w-3 h-3 text-ink-400" />
                    <span>{doc.chunks} chunks</span>
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={() => onDeleteDocument(doc.doc_id)}
              disabled={disabled || isDeleting}
              className="p-1.5 text-ink-400 hover:text-rose-700 hover:bg-rose-50 rounded-md transition-colors disabled:opacity-40 focus-ring"
              aria-label={`Delete ${doc.name} and rebuild index`}
              title="Delete document and re-index"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};
