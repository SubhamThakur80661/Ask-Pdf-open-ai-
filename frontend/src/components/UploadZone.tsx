import { useRef, useState, type FC, type DragEvent, type ChangeEvent } from 'react';
import { UploadCloud, FileText, X, AlertCircle, Loader2 } from 'lucide-react';
import type { UploadStage } from '../types';

interface UploadZoneProps {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
  onRemoveFile: (index: number) => void;
  onProcess: () => void;
  isProcessing: boolean;
  stage: UploadStage;
  hasToken: boolean;
  onOpenSettings: () => void;
  disabled: boolean;
}

export const UploadZone: FC<UploadZoneProps> = ({
  onFilesSelected,
  selectedFiles,
  onRemoveFile,
  onProcess,
  isProcessing,
  stage,
  hasToken,
  onOpenSettings,
  disabled,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    if (!disabled && !isProcessing) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const validateAndAddFiles = (incomingFiles: FileList | File[]) => {
    setValidationError(null);
    const newFiles: File[] = [];
    const maxFiles = 10;
    const maxSizeBytes = 25 * 1024 * 1024; // 25 MB

    const totalCount = selectedFiles.length + incomingFiles.length;
    if (totalCount > maxFiles) {
      setValidationError(`Maximum ${maxFiles} PDFs allowed. Please remove excess files.`);
      return;
    }

    for (let i = 0; i < incomingFiles.length; i++) {
      const file = incomingFiles[i];
      if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
        setValidationError(`'${file.name}' is not a PDF. Only PDF files are supported.`);
        return;
      }
      if (file.size > maxSizeBytes) {
        setValidationError(`'${file.name}' exceeds the 25 MB file limit (${formatFileSize(file.size)}).`);
        return;
      }
      newFiles.push(file);
    }

    if (newFiles.length > 0) {
      onFilesSelected(newFiles);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled || isProcessing) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndAddFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndAddFiles(e.target.files);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const getStageDescription = (s: UploadStage): string => {
    switch (s) {
      case 'reading':
        return 'Reading uploaded PDF streams...';
      case 'extracting':
        return 'Extracting digital text and page numbers...';
      case 'embedding':
        return 'Calling Hugging Face embedding endpoint...';
      case 'indexing':
        return 'Building in-memory FAISS vector index...';
      default:
        return 'Processing documents...';
    }
  };

  return (
    <div className="space-y-4">
      {/* Drop Target */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && !isProcessing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-ink-900 bg-white/80 backdrop-blur-md scale-[0.99] shadow-lg'
            : 'border-paper-300 hover:border-ink-600 bg-white/70 backdrop-blur-md hover:bg-white/85 hover:shadow-card'
        } ${disabled || isProcessing ? 'opacity-60 cursor-not-allowed pointer-events-none' : ''}`}
        role="button"
        tabIndex={disabled || isProcessing ? -1 : 0}
        aria-label="Upload PDF documents"
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled && !isProcessing) {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled || isProcessing}
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="p-2.5 rounded-full bg-paper-100 text-ink-700 border border-paper-200 animate-float">
            <UploadCloud className="w-5 h-5 text-ink-700" />
          </div>
          <div>
            <p className="text-sm font-medium text-ink-950">
              <span className="text-scholar font-semibold underline decoration-paper-300 underline-offset-2">
                Click to browse
              </span>{' '}
              or drag & drop PDFs
            </p>
            <p className="text-xs text-ink-500 mt-0.5">
              PDF only, up to 10 files, 25 MB each.
            </p>
          </div>
        </div>
      </div>

      {/* Validation Alert */}
      {validationError && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800">
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1 leading-normal">{validationError}</div>
          <button
            onClick={() => setValidationError(null)}
            className="text-rose-500 hover:text-rose-800"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Selected Files Queue (before processing) */}
      {selectedFiles.length > 0 && (
        <div className="space-y-2 bg-white/75 backdrop-blur-md p-3 rounded-lg border border-white/70 shadow-subtle animate-rise">
          <div className="flex items-center justify-between text-xs text-ink-600 font-medium pb-1 border-b border-paper-200">
            <span>Staged files ({selectedFiles.length})</span>
            <span>
              Total:{' '}
              {formatFileSize(selectedFiles.reduce((acc, f) => acc + f.size, 0))}
            </span>
          </div>

          <ul className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
            {selectedFiles.map((file, idx) => (
              <li
                key={`${file.name}-${idx}`}
                className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md bg-paper-100 text-xs text-ink-800 border border-paper-200 animate-rise"
                style={{ animationDelay: `${idx * 60}ms` }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="w-3.5 h-3.5 text-ink-500 flex-shrink-0" />
                  <span className="truncate font-medium" title={file.name}>
                    {file.name}
                  </span>
                  <span className="text-ink-400 font-mono text-[11px] flex-shrink-0">
                    ({formatFileSize(file.size)})
                  </span>
                </div>
                {!isProcessing && (
                  <button
                    onClick={() => onRemoveFile(idx)}
                    className="p-1 text-ink-400 hover:text-rose-700 hover:bg-rose-50 rounded transition-colors focus-ring"
                    aria-label={`Remove ${file.name} from queue`}
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </li>
            ))}
          </ul>

          {/* Token Requirement Warning if missing */}
          {!hasToken && (
            <div className="flex items-center justify-between p-2.5 rounded-md bg-amber-50 border border-amber-200 text-xs text-amber-900">
              <span className="flex items-center gap-1.5">
                <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
                <span>Token required before processing</span>
              </span>
              <button
                onClick={onOpenSettings}
                className="text-xs font-semibold text-scholar underline hover:text-scholar-hover focus-ring rounded"
              >
                Add token
              </button>
            </div>
          )}

          {/* Action Button */}
          <div className="pt-1">
            <button
              onClick={onProcess}
              disabled={isProcessing || !hasToken || disabled}
              className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-xs font-semibold text-white bg-ink-900 hover:bg-ink-950 disabled:opacity-50 disabled:pointer-events-none transition-all shadow-subtle focus-ring"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>{getStageDescription(stage)}</span>
                </>
              ) : (
                <span>Process {selectedFiles.length} {selectedFiles.length === 1 ? 'document' : 'documents'}</span>
              )}
            </button>
            {isProcessing && (
              <div className="mt-2 h-1 rounded-full bg-paper-200 overflow-hidden" aria-hidden="true">
                <div className="h-full w-2/5 rounded-full bg-gradient-to-r from-scholar/40 via-scholar to-scholar/40 animate-shimmer" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
