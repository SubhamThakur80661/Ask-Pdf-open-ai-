import type { FC } from 'react';

/**
 * Ambient animated background: soft drifting gradient orbs + a faint dot
 * grid over the paper base. Pure CSS animation, pointer-events-none, and
 * very low opacity so text stays readable over it.
 */
export const AmbientBackground: FC = () => {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {/* Faint paper dot texture */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(21,24,33,0.045)_1px,transparent_1px)] bg-[length:26px_26px]" />
      {/* Drifting orbs */}
      <div className="absolute -top-40 -left-32 w-[38rem] h-[38rem] rounded-full bg-scholar/10 blur-3xl animate-aurora" />
      <div
        className="absolute top-1/3 -right-44 w-[30rem] h-[30rem] rounded-full bg-amber-400/10 blur-3xl animate-aurora"
        style={{ animationDelay: '-9s', animationDuration: '32s' }}
      />
      <div
        className="absolute -bottom-48 left-1/4 w-[34rem] h-[34rem] rounded-full bg-sky-400/10 blur-3xl animate-aurora"
        style={{ animationDelay: '-17s', animationDuration: '38s' }}
      />
    </div>
  );
};
