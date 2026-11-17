css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --signal: #3DF5A8;
  --signal-dim: rgba(61, 245, 168, 0.55);
  --amber: #F5B83D;
  --panel: #14181D;
  --line: rgba(216, 221, 227, 0.12);
  --text: #D8DDE3;
  --muted: #8A919C;
  --bright: #F2F5F3;
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
.hero { padding: 0.6rem 0 0.2rem; }
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
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-top: 1.2rem;
  padding: 0.45rem 0.95rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.06em;
  color: var(--text);
  background: rgba(255, 255, 255, 0.03);
  animation: fadeUp 0.7s 0.24s ease both;
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
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1.1rem 1.15rem;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.005));
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
  animation: fadeUp 0.7s 0.3s ease both;
}
.step-card:hover {
  transform: translateY(-4px);
  border-color: var(--signal-dim);
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
div[data-testid="stTextInput"] { margin-top: 0.6rem; }
div[data-testid="stTextInput"] input {
  border-radius: 10px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--signal) !important;
  box-shadow: 0 0 0 3px rgba(61, 245, 168, 0.15) !important;
}
div[data-testid="stFileUploader"] > section {
  border-radius: 12px;
  transition: border-color 0.2s ease;
}
div[data-testid="stFileUploader"] > section:hover {
  border-color: rgba(61, 245, 168, 0.5);
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
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
  padding: 1.1rem 1.2rem;
  border-radius: 14px;
  margin-bottom: 0.9rem;
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  border: 1px solid var(--line);
  animation: slideIn 0.35s ease both;
}
.chat-message.user {
  background: rgba(61, 245, 168, 0.06);
  border-color: rgba(61, 245, 168, 0.25);
}
.chat-message.bot {
  background: var(--panel);
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
