"""
Enso Group HR Policy Assistant - Enterprise Edition
A production-ready, strictly grounded RAG system for corporate HR governance.
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

# ── Enterprise Styling & Modern Web Theme ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Unwanted Streamlit Chrome */
    #MainMenu { visibility: hidden !important; height: 0 !important; }
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1050px !important;
    }

    /* Enterprise Navigation Header */
    .enterprise-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.5rem;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .brand-logo-pill {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }
    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }
    .nav-status-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Welcome Hero Card (Empty State) */
    .hero-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 2.2rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-description {
        font-size: 0.95rem;
        color: #94a3b8;
        max-width: 650px;
        margin: 0 auto 1.6rem auto;
        line-height: 1.6;
    }
    .pillars-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-top: 1rem;
    }
    .pillar-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 10px;
        padding: 0.9rem 0.8rem;
        text-align: left;
        transition: all 0.2s ease;
    }
    .pillar-tag {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #3b82f6;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }
    .pillar-title {
        font-weight: 700;
        font-size: 0.82rem;
        color: #e2e8f0;
        margin-bottom: 0.15rem;
    }
    .pillar-sub {
        font-size: 0.72rem;
        color: #64748b;
        line-height: 1.35;
    }

    /* Answer Cards */
    .enterprise-answer-found {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .enterprise-answer-notfound {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .badge-enterprise-found {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .badge-enterprise-notfound {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24 !important;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .answer-text {
        color: #f1f5f9 !important;
        font-size: 0.98rem;
        line-height: 1.65;
        font-weight: 400;
        margin: 0 0 0.85rem 0;
    }
    .citation-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(51, 65, 85, 0.6);
        margin-top: 0.75rem;
    }
    .citation-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(30, 58, 138, 0.35);
        border: 1px solid rgba(96, 165, 250, 0.3);
        color: #93c5fd;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
    }
    .citation-tag-amber {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(120, 53, 15, 0.25);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fde68a;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
    }
    .metadata-pill {
        color: #64748b;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Evidence Drawer */
    .evidence-card {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .evidence-meta {
        display: flex;
        justify-content: space-between;
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.75rem;
        margin-bottom: 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .evidence-body {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1rem 0 !important;
        border-color: #1e293b !important;
    }
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.65rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.25rem;
    }
    .sidebar-stat-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 0.85rem;
        color: #e2e8f0;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Buttons in sidebar */
    .stButton > button {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        border-radius: 10px !important;
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

# ── Enterprise Top Navigation Bar ───────────────────────────────────
st.markdown("""
<div class="enterprise-nav">
    <div class="brand-group">
        <div class="brand-logo-pill">ENSO</div>
        <div>
            <h1 class="brand-title">HR Policy Assistant</h1>
            <p class="brand-subtitle">Enterprise Governance & Compliance Knowledge Hub</p>
        </div>
    </div>
    <div class="nav-status-badge">
        <div class="status-dot"></div>
        <span>Strict Grounding Active (79 Pages)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Answer Renderer ─────────────────────────────────────────────────
def render_answer(msg: dict, msg_idx: int) -> None:
    """Renders a polished corporate answer card with citations, evidence, and PDF view."""
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

        citation_chips = ""
        if citations:
            for c in citations:
                citation_chips += f'<div class="citation-tag">{c.get("section", "Section")} &gt; {c.get("policy", "Policy")} (Page {c.get("page", "?")})</div>'
        elif ref_text:
            citation_chips = f'<div class="citation-tag">{ref_text}</div>'

        st.markdown(f"""
        <div class="enterprise-answer-found">
            <div class="badge-enterprise-found">
                <span>●</span> VERIFIED POLICY CLAUSE
            </div>
            <div class="answer-text">{main_answer}</div>
            <div class="citation-bar">
                {citation_chips}
                <div class="metadata-pill" style="margin-left: auto;">{latency}s latency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="enterprise-answer-notfound">
            <div class="badge-enterprise-notfound">
                <span>●</span> POLICY CLAUSE NOT SPECIFIED
            </div>
            <div class="answer-text"><strong>Not found</strong></div>
            <div class="citation-bar">
                <div class="citation-tag-amber">Out-of-Document Restriction Enforced</div>
                <div class="metadata-pill" style="margin-left: auto;">{latency}s verification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Retrieved Evidence Drawer
    if st.session_state.show_evidence and chunks:
        with st.expander(f"Audit Trail: Inspect {len(chunks)} Retrieved Policy Passages", expanded=False):
            for i, (doc, score) in enumerate(chunks):
                meta = doc.metadata
                match_pct = int(score * 100)
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="evidence-meta">
                        <span>Chunk #{i+1} · Page {meta.get("page", "?")} · {meta.get("section", "?")}</span>
                        <span>Relevance: {match_pct}%</span>
                    </div>
                    <div class="evidence-body">{doc.page_content}</div>
                </div>
                """, unsafe_allow_html=True)

    # PyMuPDF Source Page Preview
    if found and citations:
        cited_pages = sorted(set(c["page"] for c in citations))
        for p in cited_pages:
            with st.expander(f"View Official HR Manual Page {p}", expanded=False):
                try:
                    img_bytes = render_page_image(p, dpi=160)
                    st.image(img_bytes, caption=f"Enso Group HR Policy Manual — Page {p} (Official Document)", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render page {p}: {e}")

# ── Sidebar Navigation & Diagnostics ────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="background:#2563eb; color:#fff; font-weight:800; font-size:0.75rem; padding:0.2rem 0.5rem; border-radius:4px;">ENSO</span>
        <span style="font-weight:700; color:#f8fafc; font-size:0.95rem;">Group HR Portal</span>
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-bottom:1rem;">Governance & Employee Handbooks</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Verified MidSem Exam Queries</div>', unsafe_allow_html=True)
    exam_questions = [
        ("Privilege Leave (PL) Allowance", "How many days of Privilege Leave (PL) does a permanent employee get per year?"),
        ("Probation Period (Up to Sr. Mgr)", "What is the probation period for employees up to Senior Manager?"),
        ("Health Insurance Premium", "What is the company's health-insurance premium amount?"),
        ("Current GST Rate in India", "What is the current GST rate in India?"),
        ("Maternity Leave Entitlement", "How many days of maternity leave are given?"),
    ]

    for label, query in exam_questions:
        if st.button(label, key=f"btn_{hash(query)}", use_container_width=True):
            st.session_state.pending_question = query

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Governance Settings</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider("Retrieval Context Chunks", 3, 10, RETRIEVER_TOP_K)
    st.session_state.show_evidence = st.toggle("Show Audit Trail / Evidence", value=True)

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Content Area ───────────────────────────────────────────────
if store is None:
    st.error("The vector index has not been built yet. Run `python -m src.ingest --rebuild` to construct the ChromaDB index.")
    st.stop()

# If empty chat history, show Enterprise Welcome Hero
if not st.session_state.messages:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Enso Group Policy Intelligence</div>
        <div class="hero-description">
            Ask any question regarding official company policies. All answers are strictly grounded in the 79-page HR Policy Manual with exact page and section citations. Outside hallucination is strictly disabled.
        </div>
        <div class="pillars-grid">
            <div class="pillar-item">
                <div class="pillar-tag">SECTION 1</div>
                <div class="pillar-title">Leave & Absence</div>
                <div class="pillar-sub">PL, CL, SL, Maternity & Bereavement rules</div>
            </div>
            <div class="pillar-item">
                <div class="pillar-tag">SECTION 2</div>
                <div class="pillar-title">Probation & Roles</div>
                <div class="pillar-sub">3-month appraisal, confirmation criteria</div>
            </div>
            <div class="pillar-item">
                <div class="pillar-tag">SECTION 3</div>
                <div class="pillar-title">Health & Wellness</div>
                <div class="pillar-sub">Workplace safety standards & protocols</div>
            </div>
            <div class="pillar-item">
                <div class="pillar-tag">SECTION 4</div>
                <div class="pillar-title">Code of Conduct</div>
                <div class="pillar-sub">Grievance procedures & compliance guidelines</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"**Question:** {msg['content']}")
        else:
            render_answer(msg, idx)

# Check for button click from sidebar
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Enter employee query (e.g. 'How many days of Privilege Leave does a permanent employee get?')...")
question = pending or user_input

if question:
    # Append and render user query
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**Question:** {question}")

    # Process and render assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified HR Manual clauses..."):
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
