import type { FC } from 'react';
import { KeyRound, RefreshCw, CheckCircle2 } from 'lucide-react';
import type { BackendStatus } from '../types';

interface HeaderProps {
  status: BackendStatus;
  isRetrying: boolean;
  onRetry: () => void;
  hasToken: boolean;
  onOpenSettings: () => void;
}

export const Header: FC<HeaderProps> = ({
  status,
  isRetrying,
  onRetry,
  hasToken,
  onOpenSettings,
}) => {
  return (
    <header className="border-b border-[#e7e4db] bg-paper-50 sticky top-0 z-30 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-md bg-ink-900 text-paper-50 flex items-center justify-center font-serif font-bold text-lg shadow-sm border border-ink-800">
            <span>Op</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-serif text-2xl font-bold tracking-tight text-ink-950">
                OpCit
              </span>
              <span className="hidden sm:inline-block text-xs font-mono px-1.5 py-0.5 rounded bg-paper-200 text-ink-700 border border-paper-300">
                op. cit.
              </span>
            </div>
            <p className="text-xs text-ink-600 hidden md:block">
              Ask your documents. Every answer cites its page.
            </p>
          </div>
        </div>

        {/* Status and Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Backend Status Pill */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
              status === 'online'
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                : status === 'checking'
                ? 'bg-amber-50 text-amber-800 border-amber-200'
                : status === 'offline'
                ? 'bg-rose-50 text-rose-800 border-rose-200'
                : 'bg-zinc-100 text-zinc-700 border-zinc-200'
            }`}
            role="status"
            aria-live="polite"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                status === 'online'
                  ? 'bg-emerald-500'
                  : status === 'checking'
                  ? 'bg-amber-500 animate-pulse'
                  : status === 'offline'
                  ? 'bg-rose-500'
                  : 'bg-zinc-400'
              }`}
            />
            <span className="capitalize">
              {status === 'online'
                ? 'Online'
                : status === 'checking'
                ? 'Checking...'
                : status === 'offline'
                ? 'Offline'
                : 'Not configured'}
            </span>
            {status === 'offline' && (
              <button
                onClick={onRetry}
                disabled={isRetrying}
                className="ml-1 p-0.5 hover:bg-rose-100 rounded text-rose-700 transition-colors focus-ring"
                aria-label="Retry connection to backend"
                title="Retry connection"
              >
                <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
              </button>
            )}
          </div>

          {/* Token Status / Settings Trigger */}
          <button
            onClick={onOpenSettings}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-all focus-ring ${
              hasToken
                ? 'bg-white text-ink-800 border-paper-300 hover:bg-paper-100 shadow-subtle'
                : 'bg-amber-50 text-amber-900 border-amber-300 hover:bg-amber-100 animate-pulse'
            }`}
            aria-label="Open Hugging Face Token Settings"
          >
            <KeyRound className="w-3.5 h-3.5 text-ink-600" />
            <span className="hidden sm:inline">
              {hasToken ? 'Token configured' : 'Add token'}
            </span>
            {hasToken ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            ) : (
              <span className="w-2 h-2 rounded-full bg-amber-500" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
