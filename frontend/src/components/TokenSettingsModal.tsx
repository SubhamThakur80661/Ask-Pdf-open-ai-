import { useState, useEffect, type FC } from 'react';
import { X, Eye, EyeOff, ExternalLink, ShieldCheck, Check, AlertCircle, RefreshCw, Trash2 } from 'lucide-react';
import { validateToken } from '../api';

interface TokenSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  token: string;
  onSaveToken: (token: string) => void;
  onClearToken: () => void;
}

export const TokenSettingsModal: FC<TokenSettingsModalProps> = ({
  isOpen,
  onClose,
  token,
  onSaveToken,
  onClearToken,
}) => {
  const [inputVal, setInputVal] = useState(token);
  const [showToken, setShowToken] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<'idle' | 'valid' | 'invalid'>('idle');

  useEffect(() => {
    setInputVal(token);
    setValidationResult('idle');
  }, [token, isOpen]);

  // Handle Escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleValidate = async () => {
    if (!inputVal.trim()) {
      setValidationResult('invalid');
      return;
    }
    setIsValidating(true);
    setValidationResult('idle');
    try {
      const isValid = await validateToken(inputVal.trim());
      setValidationResult(isValid ? 'valid' : 'invalid');
    } catch {
      setValidationResult('invalid');
    } finally {
      setIsValidating(false);
    }
  };

  const handleSave = () => {
    onSaveToken(inputVal.trim());
    onClose();
  };

  const handleClear = () => {
    setInputVal('');
    setValidationResult('idle');
    onClearToken();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink-950/40 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="token-modal-title"
    >
      <div
        className="w-full max-w-lg bg-paper-50 rounded-xl border border-[#dedad0] shadow-modal overflow-hidden transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-5 border-b border-[#e7e4db] flex items-center justify-between bg-white">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-paper-100 text-ink-800 border border-paper-200">
              <ShieldCheck className="w-5 h-5 text-ink-700" />
            </div>
            <div>
              <h2 id="token-modal-title" className="text-lg font-serif font-semibold text-ink-950">
                Hugging Face Token
              </h2>
              <p className="text-xs text-ink-600">Client-side authentication</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-ink-500 hover:text-ink-900 hover:bg-paper-200 rounded-md transition-colors focus-ring"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 space-y-5">
          {/* Plain language explanation */}
          <div className="p-4 rounded-lg bg-paper-100/70 border border-paper-200 text-xs leading-relaxed text-ink-700 space-y-2">
            <p className="font-medium text-ink-900">
              Your token stays strictly in your browser.
            </p>
            <p>
              It is sent directly with each request in an <code className="text-[11px] bg-paper-200 px-1 py-0.5 rounded font-mono text-ink-800">X-HF-Token</code> header so the app can invoke Hugging Face embedding and inference models on your behalf. It is never stored on the server and never logged.
            </p>
            <div className="pt-1">
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-scholar font-medium hover:underline focus-ring rounded"
              >
                <span>Get a free token on Hugging Face</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {/* Token Input */}
          <div className="space-y-1.5">
            <label htmlFor="hf-token-input" className="block text-xs font-medium text-ink-800 uppercase tracking-wider">
              Access Token (Read Permission)
            </label>
            <div className="relative">
              <input
                id="hf-token-input"
                type={showToken ? 'text' : 'password'}
                value={inputVal}
                onChange={(e) => {
                  setInputVal(e.target.value);
                  setValidationResult('idle');
                }}
                placeholder="hf_..."
                className="w-full px-3.5 py-2.5 pr-20 rounded-md border border-paper-300 bg-white text-ink-900 font-mono text-sm placeholder:text-ink-400 focus-ring shadow-subtle"
                autoComplete="off"
                spellCheck="false"
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setShowToken(!showToken)}
                  className="p-1.5 text-ink-500 hover:text-ink-800 rounded transition-colors focus-ring"
                  aria-label={showToken ? 'Hide token' : 'Show token'}
                  title={showToken ? 'Hide token' : 'Show token'}
                >
                  {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>

          {/* Validation Feedback */}
          {validationResult === 'valid' && (
            <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-50 border border-emerald-200 text-xs text-emerald-800">
              <Check className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <span>Token is valid. Connected to Hugging Face successfully.</span>
            </div>
          )}

          {validationResult === 'invalid' && (
            <div className="flex items-center gap-2 p-3 rounded-md bg-rose-50 border border-rose-200 text-xs text-rose-800">
              <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
              <span>Token is invalid or unauthorized. Please verify token on Hugging Face.</span>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-[#e7e4db] bg-paper-100 flex items-center justify-between">
          <div>
            {token && (
              <button
                type="button"
                onClick={handleClear}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-rose-700 hover:bg-rose-50 border border-transparent hover:border-rose-200 transition-colors focus-ring"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear token</span>
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleValidate}
              disabled={isValidating || !inputVal.trim()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-ink-800 bg-white border border-paper-300 hover:bg-paper-50 disabled:opacity-50 transition-colors focus-ring shadow-subtle"
            >
              {isValidating ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Validating...</span>
                </>
              ) : (
                <span>Validate</span>
              )}
            </button>
            <button
              type="button"
              onClick={handleSave}
              className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-medium text-white bg-ink-900 hover:bg-ink-950 transition-colors focus-ring shadow-subtle"
            >
              <span>Save token</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
