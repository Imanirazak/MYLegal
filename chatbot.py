import os, io, re, textwrap, hashlib, base64, json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ---------------- Setup ----------------
load_dotenv()

# Defaults so this file is runnable without missing vars
MODEL_DEFAULT = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
TEMPERATURE_DEFAULT = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
SYSTEM_DEFAULT = (
   "You are AI Legal Assistant for the Rakyat, a helpful and friendly assistant. "
    "Your goal is to help individuals and small businesses understand legal documents in simple, everyday language. "
    "Summarize agreements, highlight important clauses, and explain potential risks clearly. "
    "If CONTEXT (document text or extracted clauses) is provided, use it strictly to generate explanations. "
    "Match the user's language (Bahasa Melayu or English). "
    "Always provide practical, easy-to-understand insights, but do not act as a lawyer or give official legal advice. "
    "If unsure or if something is outside the CONTEXT, say you don't know and suggest which Malaysian Act, regulation, "
    "or authority (e.g., Employment Act, Tenancy Tribunal, LHDN) the user should consult. "
    "End with a gentle reminder that this is for educational purposes, not professional legal advice."
    "If CONTEXT is provided, answer strictly using it and cite the sources. "
    "If unsure or not in CONTEXT, say you don't know and suggest which Act/section to consult. "
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)

# Hide and collapse sidebar; also hide menu/toolbar if desired
st.set_page_config(
    page_title="MYLegal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Black • Turquoise • White theme (plus: hide sidebar/menu/toolbar)
def apply_mylegal_theme():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
      :root{
        --bg:#0B0F0F;            /* black */
        --panel:rgba(18,22,23,.75);
        --card:rgba(20,25,26,.56);
        --aqua:#10E7D1;          /* turquoise */
        --aqua2:#20F3FF;
        --txt:#FFFFFF;           /* white */
        --muted:#93a3a5;
        --border:rgba(16,231,209,.22);
      }

      /* App background */
      .stApp, .stApp header{
        background:
          radial-gradient(900px 420px at 15% -10%, rgba(16,231,209,.10), transparent 60%),
          radial-gradient(900px 420px at 110% 0%, rgba(32,243,255,.08), transparent 60%),
          linear-gradient(180deg,#0B0F0F 0%, #0A0E0E 60%, #090D0D 100%);
        color:var(--txt);
        font-family:"Outfit", ui-sans-serif, system-ui;
      }
      .block-container{max-width:1100px; padding-top:1rem;}

      /* Hide sidebar completely */
      section[data-testid="stSidebar"]{ display:none !important; }
      div[data-testid="stSidebar"]{ display:none !important; }

      /* (Optional) Hide the top-right menu and toolbar */
      #MainMenu { visibility:hidden; }
      [data-testid="stToolbar"] { display:none !important; }
      button[kind="header"] { display:none !important; } /* hamburger */

      /* Centered hero */
      .hero {
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:.35rem; margin:.2rem 0 1.1rem 0; text-align:center;
      }
      .hero-logo {
        width:72px; height:auto; border-radius:10px;
        filter: drop-shadow(0 10px 22px rgba(16,231,209,.18));
      }
      .hero h1{
        font-size:42px; margin:.1rem 0 0 0;
        background:linear-gradient(90deg,var(--txt),var(--aqua2));
        -webkit-background-clip:text; background-clip:text; color:transparent;
      }
      .hero p{ color:var(--muted); margin:0; }

      /* Headings */
      h1,h2,h3{ color:var(--txt)!important; }
      .brand h1{
        font-size:42px; margin:0;
        background:linear-gradient(90deg,var(--txt),var(--aqua2));
        -webkit-background-clip:text; background-clip:text; color:transparent;
        text-shadow:0 0 10px rgba(16,231,209,.2);
      }
      .brand p{ color:var(--muted); margin:.15rem 0 0 0; }

      /* Uploader */
      [data-testid="stFileUploaderDropzone"]{
        background:rgba(255,255,255,.03); border:2px dashed var(--border); border-radius:16px;
      }
      [data-testid="stFileUploaderDropzone"]:hover{
        border-color:var(--aqua); background:rgba(16,231,209,.06);
      }

      /* Chat bubbles */
      [data-testid="stChatMessage"]{
        background:var(--card); border:1px solid rgba(255,255,255,.08);
        border-radius:18px; padding:16px; box-shadow:0 14px 30px rgba(0,0,0,.35);
      }

      /* Inputs / textarea / slider */
      .stTextArea textarea, .stTextInput>div>div>input{
        background:rgba(255,255,255,.03)!important; border:1px solid rgba(255,255,255,.1)!important; color:var(--txt)!important;
      }
      input[type="range"]{ accent-color:var(--aqua); }

      /* Buttons */
      .stButton>button{
        background:linear-gradient(135deg,var(--aqua),var(--aqua2));
        color:#001313; border:0; border-radius:12px; padding:.6rem 1rem; font-weight:700;
        box-shadow:0 10px 28px rgba(16,231,209,.25); transition:all .15s;
      }
      .stButton>button:hover{ transform:translateY(-1px); box-shadow:0 14px 34px rgba(32,243,255,.35); }

      /* Sources card */
      .src-card{ background:rgba(20,25,26,.65); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }
    </style>
    """, unsafe_allow_html=True)

def logo_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def pin_logo_top_left(path: str):
    b64 = logo_b64(path)
    if not b64:
        return
    st.markdown(dedent(f"""
    <style>
      .mylegal-top-left {{
        position: fixed;
        top: 60px; left: 50px;
        width: 150px; height: auto;
        border-radius: 10px;
        filter: drop-shadow(0 12px 24px rgba(16,231,209,.22));
        z-index: 2000;
      }}
      @media (max-width: 640px) {{
        .mylegal-top-left {{ width: 42px; top: 10px; left: 12px; }}
      }}
    </style>
    <a href="#" title="MYLegal">
      <img class="mylegal-top-left" src="data:image/png;base64,{b64}" alt="MYLegal logo" />
    </a>
    """), unsafe_allow_html=True)

def hero_header(logo_path="D:/sdk/AWS/mylegal_logo.png"):
    b64 = logo_b64(logo_path)
    logo_tag = f'<img class="hero-logo" src="data:image/png;base64,{b64}" alt="MYLegal logo" />' if b64 else ""
    html = f"""
<div class="hero">
  {logo_tag}
  <h1>MYLegal</h1>
  <p>Welcome to Malaysia’s AI legal assistant. MYLegal is bilingual (Bahasa Melayu/English).</p>
  <p>Upload PDFs/TXT and ask employment-law questions. It reads your documents, finds the right clauses, and answers with explanations and inline citations. It can also summarize uploads, suggest Acts/sections to check, and fall back to general knowledge when needed. <em>Educational information only, not legal advice.</em></p>
</div>
"""
    st.markdown(dedent(html), unsafe_allow_html=True)

apply_mylegal_theme()
pin_logo_top_left("D:/sdk/AWS/mylegal_logo.jpg")
hero_header()

# ---------------- Config & Index paths ----------------
INDEX_DIR = Path("index")
EMB_PATH  = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.jsonl"
CONF_PATH = INDEX_DIR / "config.json"

@st.cache_resource(show_spinner=False)
def load_disk_store():
    if EMB_PATH.exists() and META_PATH.exists() and CONF_PATH.exists():
        embs = np.load(str(EMB_PATH), mmap_mode="r")  # (N, d), normalized
        meta = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]
        cfg  = json.load(open(CONF_PATH, "r", encoding="utf-8"))
        return {"embs": embs, "meta": meta, "cfg": cfg}
    return None

disk_store = load_disk_store()

# ---------------- Helpers ----------------
def clean_text(s: str) -> str:
    s = s.replace("\u0000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_pdf_bytes(file_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = clean_text(txt)
        if txt:
            out.append((i + 1, txt))
    return out

def chunk_text(t: str, size=900, overlap=150) -> List[str]:
    chunks, start = [], 0
    while start < len(t):
        end = min(start + size, len(t))
        chunk = t[start:end]
        chunks.append(chunk)
        if end == len(t): break
        start = max(0, end - overlap)
    return chunks

def shorten(s: str, n=220):
    s = s.replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

def build_prompt(user_q: str, sources_for_prompt: List[Dict], system_msg: str) -> List[Dict]:
    lines = []
    for s in sources_for_prompt:
        lines.append(f"- {s['name']} p.{s['page']}: {s['text']}")
    context = "\n".join(lines) if lines else "(no context available)"

    user_block = f"""
QUESTION:
{user_q}

CONTEXT (verbatim excerpts with citations):
{context}

Instructions:
- Answer ONLY using CONTEXT above; if not available in CONTEXT, say you don't know.
- Keep citations inline, e.g., ({{file}} p.{{page}}).
- Reply in the user's language (Malay or English).
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_block},
    ]

# ---------------- Session State ----------------
if "corpus" not in st.session_state:
    st.session_state.corpus = []     # list of dicts: {name, page, text, emb}
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
if "seen_files" not in st.session_state:
    st.session_state.seen_files = set()
if "seen_chunks" not in st.session_state:
    st.session_state.seen_chunks = set()
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Upload & Index (in-memory) ----------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT) to be summarize. They’re kept in memory for this session.",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    new_chunks = 0
    new_files_count = 0
    summary_texts = []  # one entry per NEW file

    for f in uploaded_files:
        file_bytes = f.getvalue()
        file_id = f"{f.name}:{len(file_bytes)}:{hashlib.sha1(file_bytes).hexdigest()[:12]}"

        if file_id in st.session_state.seen_files:
            continue
        st.session_state.seen_files.add(file_id)
        new_files_count += 1

        if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
            pages = read_pdf_bytes(file_bytes)
            doc_text_parts = []
            for page_no, txt in pages:
                if not txt:
                    continue
                doc_text_parts.append(txt)

                for ch in chunk_text(txt):
                    chunk_key = hashlib.sha1(
                        (f.name + f":{page_no}:" + ch[:120]).encode("utf-8")
                    ).hexdigest()
                    if chunk_key in st.session_state.seen_chunks:
                        continue
                    st.session_state.seen_chunks.add(chunk_key)

                    st.session_state.corpus.append(
                        {"name": f.name, "page": page_no, "text": ch}
                    )
                    new_chunks += 1

            if doc_text_parts:
                summary_texts.append("\n".join(doc_text_parts))

        else:  # TXT
            txt = clean_text(file_bytes.decode("utf-8", errors="ignore"))
            if txt:
                summary_texts.append(txt)
                for ch in chunk_text(txt):
                    chunk_key = hashlib.sha1(
                        (f.name + ":txt:" + ch[:120]).encode("utf-8")
                    ).hexdigest()
                    if chunk_key in st.session_state.seen_chunks:
                        continue
                    st.session_state.seen_chunks.add(chunk_key)

                    st.session_state.corpus.append(
                        {"name": f.name, "page": 1, "text": ch}
                    )
                    new_chunks += 1

    if new_chunks:
        with st.spinner(f"Embedding {new_chunks} new chunk(s)…"):
            texts = [c["text"] for c in st.session_state.corpus if "emb" not in c]
            if texts:
                embs = st.session_state.embedder.encode(texts, normalize_embeddings=True)
                idx = 0
                for c in st.session_state.corpus:
                    if "emb" not in c:
                        c["emb"] = embs[idx]
                        idx += 1
        st.success(f"Added {new_chunks} chunks from {new_files_count} new file(s).")

        # 🔎 Auto-summarize (once per NEW file)
        if summary_texts:
            with st.spinner("Summarizing uploaded documents…"):
                try:
                    summary_prompt = [
                        {"role": "system",
                         "content": "You are an AI legal assistant. Summarize the following documents in simple everyday language for a Malaysian audience. Highlight key clauses, obligations, and risks."},
                        {"role": "user",
                         "content": "\n\n".join(summary_texts)[:6000]}
                    ]
                    resp = client.chat.completions.create(
                        model=MODEL_DEFAULT,
                        temperature=0.3,
                        messages=summary_prompt,
                        max_tokens=700,
                    )
                    summary = resp.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"⚠ Could not summarize documents: {e}"

            with st.chat_message("assistant"):
                st.markdown("📑 **Document Summary**\n\n" + summary)
            st.session_state.history.append({"q": "Document uploaded", "a": summary, "ctx": []})

# ---------------- Retrieval ----------------
def retrieve(query: str, top_k: int = 6) -> List[Dict]:
    pieces = []

    # 1) Uploaded docs (session memory)
    if st.session_state.corpus:
        q_emb = st.session_state.embedder.encode([query], normalize_embeddings=True)[0].reshape(1, -1)
        mat = np.stack([c["emb"] for c in st.session_state.corpus])
        sims = cosine_similarity(q_emb, mat)[0]
        for i in np.argsort(-sims)[:top_k]:
            c = st.session_state.corpus[i]
            pieces.append({"name": c["name"], "page": c["page"], "text": c["text"], "score": float(sims[i])})

    # 2) Prebuilt disk index
    if disk_store is not None:
        q_vec = st.session_state.embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
        sims = disk_store["embs"] @ q_vec  # cosine (all normalized)
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, k-1)[:k]
        top = top[np.argsort(-sims[top])]
        for i in top:
            m = disk_store["meta"][int(i)]
            pieces.append({"name": m["file"], "page": m["page"], "text": m["text"], "score": float(sims[int(i)])})

    if not pieces:
        return []
    pieces.sort(key=lambda x: -x["score"])
    return pieces[:top_k]

# ---------------- Chat History ----------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])

# ---------------- Chat Input ----------------
user_q = st.chat_input("Type your question… (e.g., 'Berapa lamakah cuti bersalin di bawah Akta Kerja 1955?')")
if user_q:
    with st.chat_message("user"):
        st.markdown(user_q)

    hits = retrieve(user_q, top_k=6)
    msgs = build_prompt(user_q, hits, SYSTEM_DEFAULT)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            resp = client.chat.completions.create(
                model=MODEL_DEFAULT,
                temperature=TEMPERATURE_DEFAULT,
                messages=msgs,
                max_tokens=900,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Sorry, model error: `{e}`"
        placeholder.markdown(answer)

    st.session_state.history.append({"q": user_q, "a": answer, "ctx": hits})

    # ✅ Sources
    if hits:
        st.markdown("### Sources")
        st.markdown('<div class="src-card">', unsafe_allow_html=True)
        for h in hits[:3]:
            st.markdown(f"• **{h['name']}** p.{h['page']} — “{shorten(h['text'])}”")
        st.markdown("</div>", unsafe_allow_html=True)
