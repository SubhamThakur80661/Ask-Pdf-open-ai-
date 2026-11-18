import { useState, type FC, type MouseEvent } from 'react';
import { FileText, Quote, Sparkles } from 'lucide-react';

/**
 * Floating 3D document stack (pure CSS 3D + React mouse parallax, no 3D
 * library): three layered pages in a perspective scene that tilt toward
 * the cursor, with glass citation chips floating around them.
 */
export const FloatingDocs: FC = () => {
  const [tilt, setTilt] = useState({ rx: -6, ry: 8 });

  const handleMove = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ rx: -6 - py * 14, ry: 8 + px * 16 });
  };

  const resetTilt = () => setTilt({ rx: -6, ry: 8 });

  return (
    <div
      className="perspective-1200 relative h-48 w-full max-w-sm mx-auto select-none"
      onMouseMove={handleMove}
      onMouseLeave={resetTilt}
      aria-hidden="true"
    >
      <div
        className="preserve-3d absolute inset-0 transition-transform duration-300 ease-out"
        style={{ transform: `rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg)` }}
      >
        {/* Back page */}
        <div
          className="absolute left-1/2 top-1/2 w-32 h-40 rounded-lg bg-white border border-paper-300 shadow-card"
          style={{ transform: 'translate(-50%, -50%) translateZ(-48px) translateX(-26px) rotate(-9deg)' }}
        >
          <div className="p-3 space-y-2">
            <div className="h-2 w-3/4 rounded bg-paper-200" />
            <div className="h-1.5 w-full rounded bg-paper-200/80" />
            <div className="h-1.5 w-11/12 rounded bg-paper-200/80" />
            <div className="h-1.5 w-2/3 rounded bg-paper-200/80" />
          </div>
        </div>
        {/* Middle page */}
        <div
          className="absolute left-1/2 top-1/2 w-32 h-40 rounded-lg bg-white border border-paper-300 shadow-card"
          style={{ transform: 'translate(-50%, -50%) translateZ(-20px) translateX(14px) rotate(6deg)' }}
        >
          <div className="p-3 space-y-2">
            <div className="h-2 w-2/3 rounded bg-paper-200" />
            <div className="h-1.5 w-full rounded bg-paper-200/80" />
            <div className="h-1.5 w-10/12 rounded bg-paper-200/80" />
            <div className="h-1.5 w-3/4 rounded bg-paper-200/80" />
          </div>
        </div>
        {/* Front page (glass) */}
        <div
          className="absolute left-1/2 top-1/2 w-32 h-40 rounded-lg glass-strong"
          style={{ transform: 'translate(-50%, -50%) translateZ(24px) rotate(-2deg)' }}
        >
          <div className="p-3 space-y-2">
            <div className="flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-scholar" />
              <div className="h-2 w-16 rounded bg-scholar/80" />
            </div>
            <div className="h-1.5 w-full rounded bg-paper-200" />
            <div className="h-1.5 w-11/12 rounded bg-paper-200" />
            <div className="h-1.5 w-full rounded bg-amber-200/80" />
            <div className="h-1.5 w-2/3 rounded bg-paper-200" />
          </div>
        </div>
      </div>

      {/* Floating glass chips */}
      <div className="absolute top-2 right-2 sm:right-6 animate-float">
        <div className="glass flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-mono font-semibold text-ink-800">
          <Quote className="w-3 h-3 text-scholar" />
          <span>p. 42 &middot; cited</span>
        </div>
      </div>
      <div className="absolute bottom-3 left-1 sm:left-4 animate-float-slow">
        <div className="glass flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-mono font-semibold text-ink-800">
          <Sparkles className="w-3 h-3 text-amber-600" />
          <span>grounded answer</span>
        </div>
      </div>
    </div>
  );
};
