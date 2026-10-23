"""
Enso Group HR Policy Assistant
A strictly grounded enterprise RAG system for corporate HR policies.
"""

import importlib
import logging
import os
import time

import streamlit as st

import src.config
import src.rag_chain
importlib.reload(src.config)
importlib.reload(src.rag_chain)

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    RETRIEVER_TOP_K,
)
from src.pdf_preview import render_page_image
from src.rag_chain import ask
from src.retriever import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enso Group HR Policy Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimalist Corporate Theme (No Emojis, Clean Typography) ───────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Unwanted Streamlit Chrome */
    #MainMenu { visibility: hidden !important; height: 0 !important; }
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Layout Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 900px !important;
    }

    /* Top Corporate Header */
    .corporate-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #1e293b;
    }
    .corporate-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .corporate-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin: 0.15rem 0 0 0;
    }
    .corporate-badge {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        margin-bottom: 2rem;
    }
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .empty-state-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Answer Containers */
    .answer-box-found {
        background: #090d16;
        border: 1px solid #1e293b;
        border-left: 3px solid #10b981;
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        margin: 0.5rem 0 1rem 0;
    }
    .answer-box-notfound {
        background: #090d16;
        border: 1px solid #1e293b;
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        margin: 0.5rem 0 1rem 0;
    }
    .answer-label-found {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #10b981;
        margin-bottom: 0.5rem;
    }
    .answer-label-notfound {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #f59e0b;
        margin-bottom: 0.5rem;
    }
    .answer-body {
        font-size: 0.95rem;
        line-height: 1.65;
        color: #e2e8f0;
        margin-bottom: 0.85rem;
    }
    .citation-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
        padding-top: 0.65rem;
        border-top: 1px solid #1e293b;
    }
    .citation-link {
        font-size: 0.78rem;
        color: #38bdf8;
        font-weight: 500;
    }
    .latency-pill {
        font-size: 0.72rem;
        color: #64748b;
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Evidence Excerpts */
    .excerpt-box {
        background: #050811;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .excerpt-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .excerpt-text {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1.25rem 0 !important;
        border-color: #1e293b !important;
    }
    .sidebar-heading {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.65rem;
    }
    .sidebar-brand-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0.15rem 0 1rem 0;
    }

    /* Action Buttons */
    .stButton > button {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 0.45rem 0.75rem !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background: #1e293b !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }

    /* Chat Input */
    [data-testid="stChatInput"] {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_evidence" not in st.session_state:
    st.session_state.show_evidence = True
if "top_k" not in st.session_state:
    st.session_state.top_k = RETRIEVER_TOP_K

# ── Cached Vector Store ─────────────────────────────────────────────
@st.cache_resource
def load_store():
    try:
        store = get_vector_store()
        count = store._collection.count()
        return store, count
    except FileNotFoundError:
        return None, 0

store, doc_count = load_store()

# ── Minimalist Corporate Header ─────────────────────────────────────
st.markdown("""
<div class="corporate-header">
    <div>
        <h1 class="corporate-title">Enso Group HR Policy Assistant</h1>
        <p class="corporate-subtitle">Corporate Policy & Compliance Knowledge Base (79 Pages)</p>
    </div>
    <div class="corporate-badge">
        Strict Grounding Active
    </div>
</div>
""", unsafe_allow_html=True)

# ── Answer Renderer ─────────────────────────────────────────────────
def render_answer(msg: dict, msg_idx: int) -> None:
    """Renders a clean corporate answer card without emojis."""
    result = msg.get("result", {})
    found = result.get("found", False)
    answer = result.get("answer", msg.get("content", ""))
    latency = result.get("latency_s", 0)
    citations = result.get("citations", [])
    chunks = result.get("retrieved_chunks", [])

    if answer.startswith("Error:"):
        st.error(answer)
        return

    if found:
        # Separate main answer from reference line
        parts = answer.split("\n\nReference: ")
        main_answer = parts[0]
        ref_text = parts[1] if len(parts) > 1 else ""

        citation_html = ""
        if citations:
            refs = [f"{c.get('section', 'Section')} > {c.get('policy', 'Policy')}, Page {c.get('page', '?')}" for c in citations]
            citation_html = f'<div class="citation-link">Reference: {"; ".join(refs)}</div>'
        elif ref_text:
            citation_html = f'<div class="citation-link">Reference: {ref_text}</div>'

        st.markdown(f"""
        <div class="answer-box-found">
            <div class="answer-label-found">Verified Policy Directive</div>
            <div class="answer-body">{main_answer}</div>
            <div class="citation-strip">
                {citation_html}
                <div class="latency-pill">{latency}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="answer-box-notfound">
            <div class="answer-label-notfound">Policy Not Found</div>
            <div class="answer-body"><strong>Not found</strong></div>
            <div class="citation-strip">
                <div class="citation-link" style="color:#94a3b8;">Out-of-document queries are prohibited by compliance protocol.</div>
                <div class="latency-pill">{latency}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Retrieved Evidence Accordion
    if st.session_state.show_evidence and chunks:
        with st.expander(f"Retrieved Policy Passages ({len(chunks)})", expanded=False):
            for i, (doc, score) in enumerate(chunks):
                meta = doc.metadata
                match_pct = int(score * 100)
                st.markdown(f"""
                <div class="excerpt-box">
                    <div class="excerpt-header">
                        <span>Passage {i+1} · Page {meta.get("page", "?")} · {meta.get("section", "?")}</span>
                        <span>Relevance: {match_pct}%</span>
                    </div>
                    <div class="excerpt-text">{doc.page_content}</div>
                </div>
                """, unsafe_allow_html=True)

    # PyMuPDF Source Page Preview
    if found and citations:
        cited_pages = sorted(set(c["page"] for c in citations))
        for p in cited_pages:
            with st.expander(f"View Manual Page {p}", expanded=False):
                try:
                    img_bytes = render_page_image(p, dpi=160)
                    st.image(img_bytes, caption=f"Enso Group HR Manual — Page {p}", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render page {p}: {e}")

# ── Sidebar Navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-name">Enso Group HR</div>
    <div class="sidebar-brand-sub">Policy & Compliance Assistant</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">Preset Questions</div>', unsafe_allow_html=True)
    exam_questions = [
        ("Privilege Leave (PL) entitlement", "How many days of Privilege Leave (PL) does a permanent employee get per year?"),
        ("Probation period (up to Senior Manager)", "What is the probation period for employees up to Senior Manager?"),
        ("Health insurance premium amount", "What is the company's health-insurance premium amount?"),
        ("Current GST rate in India", "What is the current GST rate in India?"),
        ("Maternity leave entitlement", "How many days of maternity leave are given?"),
    ]

    for label, query in exam_questions:
        if st.button(label, key=f"btn_{hash(query)}", use_container_width=True):
            st.session_state.pending_question = query

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">Settings</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider("Context passages", 3, 10, RETRIEVER_TOP_K)
    st.session_state.show_evidence = st.toggle("Show retrieved passages", value=True)

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Content Area ───────────────────────────────────────────────
if store is None:
    st.error("The vector index has not been built yet. Run `python -m src.ingest --rebuild` to construct the index.")
    st.stop()

# Handle Pending Question
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Enter policy query...")
question = pending or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})

# Show clean empty state ONLY if there are no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-title">How can I assist with company policy?</div>
        <div class="empty-state-desc">
            Ask any question regarding official Enso Group policies. Answers are extracted directly from the verified 79-page manual with page citations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg, idx)

# Process New User Query
if question:
    with st.chat_message("assistant"):
        with st.spinner("Searching HR manual..."):
            try:
                result = ask(
                    question,
                    store=store,
                    top_k=st.session_state.top_k,
                )
            except Exception as e:
                result = {
                    "answer": f"Error: {e}",
                    "found": False,
                    "citations": [],
                    "retrieved_chunks": [],
                    "latency_s": 0,
                }

        msg = {"role": "assistant", "content": result["answer"], "result": result}
        render_answer(msg, len(st.session_state.messages))
        st.session_state.messages.append(msg)
