import type { FC } from 'react';
import { AlertCircle, KeyRound, RefreshCw, X, ServerOff } from 'lucide-react';
import type { BackendStatus } from '../types';

interface StatusBannerProps {
  backendStatus: BackendStatus;
  isRetrying: boolean;
  onRetry: () => void;
  hasToken: boolean;
  onOpenSettings: () => void;
  errorMessage: string | null;
  onDismissError: () => void;
}

export const StatusBanner: FC<StatusBannerProps> = ({
  backendStatus,
  isRetrying,
  onRetry,
  hasToken,
  onOpenSettings,
  errorMessage,
  onDismissError,
}) => {
  if (errorMessage) {
    return (
      <div className="bg-rose-50 border-b border-rose-200 px-4 py-2.5 text-xs text-rose-900">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span className="font-medium truncate">{errorMessage}</span>
          </div>
          <button
            onClick={onDismissError}
            className="p-1 hover:bg-rose-100 rounded text-rose-700 transition-colors focus-ring"
            aria-label="Dismiss error banner"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  }

  if (backendStatus === 'unconfigured') {
    return (
      <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-xs text-amber-900">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <ServerOff className="w-4 h-4 text-amber-600 flex-shrink-0" />
            <span>
              <strong>Backend not configured yet.</strong> Set <code className="font-mono bg-amber-100 px-1 py-0.5 rounded">VITE_API_URL</code> to connect to your FastAPI instance.
            </span>
          </div>
          <button
            onClick={onRetry}
            disabled={isRetrying}
            className="inline-flex items-center gap-1 font-semibold text-amber-800 underline hover:text-amber-950 focus-ring rounded"
          >
            <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  if (backendStatus === 'offline') {
    return (
      <div className="bg-rose-50 border-b border-rose-200 px-4 py-2 text-xs text-rose-900">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <ServerOff className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>
              <strong>Backend service offline.</strong> Could not reach the API at configured endpoint. Please start the backend or verify connection.
            </span>
          </div>
          <button
            onClick={onRetry}
            disabled={isRetrying}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-white border border-rose-300 font-semibold text-rose-800 hover:bg-rose-100 transition-colors focus-ring shadow-subtle"
          >
            <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
            <span>Retry connection</span>
          </button>
        </div>
      </div>
    );
  }

  if (!hasToken && backendStatus === 'online') {
    return (
      <div className="bg-amber-50/90 border-b border-amber-200/80 px-4 py-2 text-xs text-amber-900">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <KeyRound className="w-4 h-4 text-amber-600 flex-shrink-0" />
            <span>
              Add your free Hugging Face token in Settings to embed and query your documents.
            </span>
          </div>
          <button
            onClick={onOpenSettings}
            className="font-semibold text-scholar underline hover:text-scholar-hover focus-ring rounded"
          >
            Open Settings
          </button>
        </div>
      </div>
    );
  }

  return null;
};
