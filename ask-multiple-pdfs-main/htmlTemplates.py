css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --signal: #3DF5A8;
  --signal-dim: rgba(61, 245, 168, 0.55);
  --sky: #38BDF8;
  --teal: #2DD4BF;
  --amber: #F5B83D;
  --panel: #14181D;
  --line: rgba(216, 221, 227, 0.12);
  --text: #D8DDE3;
  --muted: #8A919C;
  --bright: #F2F5F3;
  /* glassmorphism layer: frosted surfaces over the dark ink.
     Fills are deliberately translucent (8–25%): the blur only reads as
     glass when colour shows through it. The two near-opaque dark fills
     (55%) from the first glass pass are what made panels look solid.
     Root cause of the invisible glass (proven by screenshot 2026-11-18):
     backdrop-filter only blurs what is BEHIND a panel, and behind these
     panels was near-black plus faint edge glows — so the blur showed
     nothing. The fix is colour placed directly behind each panel: big,
     vivid aurora blobs (25–40% alpha) centred on every glass surface,
     via per-panel ::before layers plus a much stronger page background.
     Fills stay as they are — lowering them further would not fix it and
     would hurt readability. */
  --glass-fill: rgba(255, 255, 255, 0.10);
  --glass-fill-strong: rgba(13, 16, 19, 0.25);
  --glass-border: rgba(255, 255, 255, 0.18);
  --glass-shadow: 0 16px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.10);
}

/* ---------- motion ---------- */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
  from { opacity: 0; transform: translateX(-10px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes gradPan {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes pulseGreen {
  0%, 100% { box-shadow: 0 0 0 0 rgba(61, 245, 168, 0.45); }
  50% { box-shadow: 0 0 0 6px rgba(61, 245, 168, 0); }
}
@keyframes pulseAmber {
  0%, 100% { box-shadow: 0 0 0 0 rgba(245, 184, 61, 0.45); }
  50% { box-shadow: 0 0 0 6px rgba(245, 184, 61, 0); }
}

/* ---------- page frame ---------- */
.block-container {
  padding-top: 1.6rem;
  max-width: 72rem;
}
/* Ambient aurora behind the glass: backdrop-filter only refracts what is
   painted behind a panel, so these washes are now LARGE, VIVID and placed
   across the CENTRE of the viewport — not just at the edges/corners — with
   24–34% alpha, over a very dark green-tinted base instead of pure
   near-black. background-attachment: fixed anchors the washes to the
   viewport so scrolled content (long chat answers) always has colour
   behind it; if a browser ignores fixed attachment the gradients still
   render, positioned over the page. The strong left-edge green/teal blobs
   sit directly behind the sidebar. Signal green + sky blue/teal only —
   same palette, no rainbow. */
div[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(52rem 34rem at 50% -8%, rgba(61, 245, 168, 0.34), transparent 65%),
    radial-gradient(44rem 28rem at 92% 10%, rgba(56, 189, 248, 0.30), transparent 65%),
    radial-gradient(46rem 30rem at 4% 28%, rgba(61, 245, 168, 0.32), transparent 65%),
    radial-gradient(48rem 30rem at 50% 52%, rgba(45, 212, 191, 0.24), transparent 65%),
    radial-gradient(44rem 28rem at 96% 72%, rgba(56, 189, 248, 0.24), transparent 65%),
    radial-gradient(48rem 30rem at 6% 96%, rgba(61, 245, 168, 0.28), transparent 65%),
    linear-gradient(180deg, #0B120E 0%, #0A0F0C 55%, #0A100D 100%);
  background-attachment: fixed;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* animated underline links */
a {
  color: var(--signal);
  text-decoration: none;
  background-image: linear-gradient(var(--signal), var(--signal));
  background-size: 0% 1px;
  background-repeat: no-repeat;
  background-position: left 100%;
  transition: background-size 0.3s ease;
}
a:hover { background-size: 100% 1px; }

/* ---------- hero ---------- */
.hero {
  position: relative;
  padding: 1.4rem 1.5rem 1.3rem;
  border: 1px solid var(--glass-border);
  border-radius: 20px;
  background: var(--glass-fill);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  backdrop-filter: blur(18px) saturate(140%);
  box-shadow: var(--glass-shadow);
}
/* Vivid aurora blob directly behind the hero: larger than the panel,
   centred on it, so the hero's own blur refracts real colour. The ::before
   sits at z-index -1 inside the hero's backdrop-filter stacking context,
   i.e. behind the translucent fill but in front of the page — and it
   carries its own blur, so the colour reads as glass even where the
   backdrop-filter sampling is weak. */
.hero::before {
  content: "";
  position: absolute;
  inset: -70px -48px;
  z-index: -1;
  border-radius: 50%;
  background:
    radial-gradient(closest-side at 28% 42%, rgba(61, 245, 168, 0.40), transparent 72%),
    radial-gradient(closest-side at 74% 58%, rgba(56, 189, 248, 0.32), transparent 72%);
  filter: blur(16px);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  animation: fadeUp 0.7s ease both;
}
.hero-eyebrow .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--signal);
  animation: pulseGreen 2s infinite;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: clamp(2.6rem, 5.8vw, 4.4rem);
  line-height: 0.98;
  letter-spacing: -0.03em;
  margin: 0.9rem 0 0.9rem;
  color: var(--bright);
  animation: fadeUp 0.7s 0.08s ease both;
}
.hero-title .grad {
  background: linear-gradient(90deg, #3DF5A8, #7DD3FC, #3DF5A8);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: gradPan 6s ease infinite;
}
.hero-sub {
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--muted);
  max-width: 34rem;
  margin: 0;
  animation: fadeUp 0.7s 0.16s ease both;
}
.status-pill {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-top: 1.2rem;
  padding: 0.45rem 0.95rem;
  border: 1px solid var(--glass-border);
  border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.06em;
  color: var(--text);
  background: rgba(255, 255, 255, 0.10);
  -webkit-backdrop-filter: blur(12px) saturate(140%);
  backdrop-filter: blur(12px) saturate(140%);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  animation: fadeUp 0.7s 0.24s ease both;
}
.status-pill::before {
  content: "";
  position: absolute;
  inset: -20px -16px;
  z-index: -1;
  border-radius: 50%;
  background:
    radial-gradient(closest-side at 35% 50%, rgba(61, 245, 168, 0.38), transparent 72%),
    radial-gradient(closest-side at 72% 50%, rgba(45, 212, 191, 0.30), transparent 72%);
  filter: blur(10px);
  pointer-events: none;
}
.status-pill .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--signal);
  animation: pulseGreen 2s infinite;
}
.status-pill .dot.amber {
  background: var(--amber);
  animation: pulseAmber 2s infinite;
}

/* ---------- step cards ---------- */
.step-card {
  position: relative;
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  padding: 1.1rem 1.15rem;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.06));
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  backdrop-filter: blur(16px) saturate(140%);
  box-shadow: var(--glass-shadow);
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
  animation: fadeUp 0.7s 0.3s ease both;
}
/* One aurora blob per card, centred on the card and larger than it. */
.step-card::before {
  content: "";
  position: absolute;
  inset: -46px -32px;
  z-index: -1;
  border-radius: 50%;
  background:
    radial-gradient(closest-side at 30% 32%, rgba(61, 245, 168, 0.36), transparent 72%),
    radial-gradient(closest-side at 72% 70%, rgba(45, 212, 191, 0.28), transparent 72%);
  filter: blur(16px);
  pointer-events: none;
}
.step-card:hover {
  transform: translateY(-4px);
  border-color: var(--signal-dim);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.08));
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(61, 245, 168, 0.15);
}
.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.2em;
  color: var(--signal);
}
.step-card h3 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.05rem;
  font-weight: 600;
  margin: 0.55rem 0 0.3rem;
  color: var(--bright);
}
.step-card p {
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--muted);
  margin: 0;
}

/* ---------- buttons ---------- */
div[data-testid="stButton"] > button {
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: 0.01em;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(61, 245, 168, 0.18);
  border-color: var(--signal);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); }
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(90deg, #3DF5A8, #2BD98F) !important;
  color: #06281A !important;
  border: none !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  box-shadow: 0 10px 28px rgba(61, 245, 168, 0.35);
}

/* ---------- inputs ---------- */
div[data-testid="stTextInput"] {
  margin-top: 0.6rem;
  position: relative;
  isolation: isolate;
}
/* Inputs can't take ::before, so the blob lives on the wrapper behind the
   translucent input, which blurs it. */
div[data-testid="stTextInput"]::before {
  content: "";
  position: absolute;
  inset: -22px -16px;
  z-index: -1;
  border-radius: 50%;
  background: radial-gradient(closest-side at 50% 50%, rgba(56, 189, 248, 0.30), transparent 72%);
  filter: blur(14px);
  pointer-events: none;
}
div[data-testid="stTextInput"] input {
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.08);
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
  border: 1px solid var(--glass-border);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--signal) !important;
  box-shadow: 0 0 0 3px rgba(61, 245, 168, 0.15) !important;
}
div[data-testid="stFileUploader"] > section {
  position: relative;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.08);
  -webkit-backdrop-filter: blur(12px) saturate(140%);
  backdrop-filter: blur(12px) saturate(140%);
  border: 1px dashed rgba(255, 255, 255, 0.22);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
  transition: border-color 0.2s ease;
}
div[data-testid="stFileUploader"] > section::before {
  content: "";
  position: absolute;
  inset: -30px -20px;
  z-index: -1;
  border-radius: 50%;
  background:
    radial-gradient(closest-side at 35% 45%, rgba(61, 245, 168, 0.34), transparent 72%),
    radial-gradient(closest-side at 70% 60%, rgba(45, 212, 191, 0.26), transparent 72%);
  filter: blur(14px);
  pointer-events: none;
}
div[data-testid="stFileUploader"] > section:hover {
  border-color: rgba(61, 245, 168, 0.5);
}

/* ---------- sidebar ---------- */
/* Frosted sidebar: translucent fill + blur over the app background. This is
   the most fragile glass surface — Streamlit paints the sidebar as a
   Streamlit-managed panel, so a ::before blob cannot be safely anchored to
   it (its positioning is not guaranteed, and a mis-anchored blob would wash
   the whole page). Instead the colour is baked into the sidebar's own
   background: vivid green/teal/sky gradient washes UNDER a 20% dark-ink
   veil, so colour shows through the glass by construction, on top of the
   strong left-edge blobs in the page background that the backdrop-filter
   refracts. Deeper sidebar layers are forced transparent so no
   Streamlit-managed opaque layer hides the glass. If a Streamlit layer
   still stays opaque, the sidebar will read as a dark panel with visible
   green/teal washes rather than true frosted glass — a screenshot check is
   the only way to confirm. */
section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(13, 16, 19, 0.20), rgba(13, 16, 19, 0.20)),
    radial-gradient(24rem 20rem at 18% 10%, rgba(61, 245, 168, 0.34), transparent 70%),
    radial-gradient(22rem 18rem at 82% 52%, rgba(45, 212, 191, 0.26), transparent 70%),
    radial-gradient(24rem 20rem at 28% 94%, rgba(56, 189, 248, 0.24), transparent 70%),
    rgba(13, 16, 19, 0.25) !important;
  -webkit-backdrop-filter: blur(20px) saturate(140%);
  backdrop-filter: blur(20px) saturate(140%);
  border-right: 1px solid var(--glass-border);
  box-shadow: 8px 0 32px rgba(0, 0, 0, 0.25);
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
  background-color: transparent !important;
}
section[data-testid="stSidebar"] input {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--glass-border);
}
.side-head {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text);
  padding-bottom: 0.55rem;
  margin: 0.4rem 0 1rem;
  border-bottom: 1px solid var(--line);
  position: relative;
}
.side-head::after {
  content: "";
  position: absolute;
  left: 0;
  bottom: -1px;
  width: 2.4rem;
  height: 2px;
  background: linear-gradient(90deg, var(--signal), transparent);
}
.side-note {
  font-size: 0.85rem;
  line-height: 1.55;
  color: var(--muted);
  margin: 0.2rem 0 1rem;
}
.side-foot {
  font-size: 0.8rem;
  line-height: 1.55;
  color: var(--muted);
  border-top: 1px solid var(--line);
  padding-top: 0.9rem;
  margin-top: 1.4rem;
}

/* ---------- chat ---------- */
.chat-message {
  position: relative;
  padding: 1.1rem 1.2rem;
  border-radius: 14px;
  margin-bottom: 0.9rem;
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  border: 1px solid var(--glass-border);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  backdrop-filter: blur(14px) saturate(140%);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  animation: slideIn 0.35s ease both;
}
/* One aurora blob per bubble, centred on the bubble and larger than it, so
   each chat message refracts its own colour instead of flat black. */
.chat-message::before {
  content: "";
  position: absolute;
  inset: -34px -26px;
  z-index: -1;
  border-radius: 50%;
  filter: blur(16px);
  pointer-events: none;
}
.chat-message.user::before {
  background: radial-gradient(closest-side at 50% 50%, rgba(61, 245, 168, 0.34), transparent 72%);
}
.chat-message.bot::before {
  background:
    radial-gradient(closest-side at 30% 40%, rgba(56, 189, 248, 0.30), transparent 72%),
    radial-gradient(closest-side at 72% 62%, rgba(45, 212, 191, 0.26), transparent 72%);
}
.chat-message.user {
  background: rgba(61, 245, 168, 0.12);
  border-color: rgba(61, 245, 168, 0.25);
}
.chat-message.bot {
  /* Strongest glass fill in the app (25% dark ink, down from 55%): long
     answer text must stay readable over the brighter ambient colour, and
     the dark fill + 14px blur keeps #E6EAE9 text on a calm surface while
     still letting the glow show through at the edges. */
  background: var(--glass-fill-strong);
}
.chat-message .avatar {
  flex: 0 0 44px;
}
.chat-message .avatar img {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  flex: 1;
  color: #E6EAE9;
  line-height: 1.65;
  font-size: 0.98rem;
  padding: 0;
  word-break: break-word;
}

/* ---------- glass fallbacks ---------- */
/* No backdrop-filter support: fall back to stronger, near-opaque fills so
   text stays readable instead of floating on a faint wash. */
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
  .hero, .step-card { background: rgba(20, 24, 29, 0.88); }
  .status-pill { background: rgba(20, 24, 29, 0.88); }
  .chat-message.bot { background: #14181D; }
  .chat-message.user { background: rgba(61, 245, 168, 0.14); }
  section[data-testid="stSidebar"] { background: #101418 !important; }
  div[data-testid="stTextInput"] input,
  div[data-testid="stFileUploader"] > section { background: rgba(20, 24, 29, 0.88); }
}

/* ---------- reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
  }
  .hero-title .grad { animation: none; }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
