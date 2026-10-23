"""
Streamlit UI for the Enso Group HR Policy Assistant.
Run: streamlit run app.py
"""

import logging
import time

import streamlit as st

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

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enso Group HR Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
    .main-header p { margin: 0.4rem 0 0 0; font-size: 0.95rem; opacity: 0.9; font-weight: 300; }

    .answer-card {
        background: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid #22c55e;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin: 0.8rem 0;
    }
    .answer-card-notfound {
        background: #fefce8; border: 1px solid #fde68a; border-left: 4px solid #f59e0b;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin: 0.8rem 0;
    }
    .badge-found {
        display: inline-block; background: #22c55e; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; margin-bottom: 0.6rem;
    }
    .badge-notfound {
        display: inline-block; background: #f59e0b; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; margin-bottom: 0.6rem;
    }
    .ref-chip {
        display: inline-block; background: #e0f2fe; border: 1px solid #7dd3fc;
        color: #0369a1; padding: 0.3rem 0.8rem; border-radius: 6px;
        font-size: 0.85rem; font-weight: 500; margin-top: 0.5rem;
    }
    .latency { color: #9ca3af; font-size: 0.8rem; margin-top: 0.4rem; }
    .evidence-chunk {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0; font-size: 0.85rem;
    }
    .evidence-header {
        color: #64748b; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.4rem;
    }
    .stButton > button { text-align: left !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_evidence" not in st.session_state:
    st.session_state.show_evidence = True
if "top_k" not in st.session_state:
    st.session_state.top_k = RETRIEVER_TOP_K


# ── Load vector store (cached) ──────────────────────────────────────
@st.cache_resource
def load_store():
    try:
        store = get_vector_store()
        count = store._collection.count()
        return store, count
    except FileNotFoundError:
        return None, 0


# ── Render answer helper ────────────────────────────────────────────
def render_answer(msg: dict, msg_idx: int) -> None:
    """Render an assistant answer with card, badges, evidence, and PDF preview."""
    result = msg.get("result", {})
    found = result.get("found", False)
    answer = result.get("answer", msg.get("content", ""))
    latency = result.get("latency_s", 0)
    citations = result.get("citations", [])
    chunks = result.get("retrieved_chunks", [])

    if found:
        parts = answer.split("\n\nReference: ")
        main_answer = parts[0]
        ref_text = parts[1] if len(parts) > 1 else ""

        st.markdown('<div class="badge-found">✅ Answered from HR Manual</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{main_answer}</div>', unsafe_allow_html=True)
        if ref_text:
            st.markdown(f'<div class="ref-chip">📄 {ref_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-notfound">⚠️ Not found in HR Manual</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-card-notfound">Not found</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="latency">⏱️ Response time: {latency}s</div>', unsafe_allow_html=True)

    # Evidence panel
    if st.session_state.show_evidence and chunks:
        with st.expander(f"📚 Retrieved Evidence ({len(chunks)} chunks)", expanded=False):
            for i, (doc, score) in enumerate(chunks):
                meta = doc.metadata
                text_preview = doc.page_content[:500].replace("\n", " ")
                st.markdown(
                    f'<div class="evidence-chunk">'
                    f'<div class="evidence-header">Chunk {i+1} | Page {meta.get("page", "?")} | '
                    f'{meta.get("section", "?")} > {meta.get("policy_title", "?")} | '
                    f'Score: {score:.3f}</div>'
                    f'{text_preview}...'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # PDF page preview
    if found and citations:
        cited_pages = sorted(set(c["page"] for c in citations))
        for p in cited_pages:
            with st.expander(f"📄 View Source: Page {p}", expanded=False):
                try:
                    img_bytes = render_page_image(p, dpi=150)
                    st.image(img_bytes, caption=f"HR Manual - Page {p}", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render page {p}: {e}")


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 HR Policy Assistant")
    st.markdown(
        "Ask questions about Enso Group's HR policies. "
        "All answers are grounded in the official HR Manual."
    )
    st.markdown("---")

    st.markdown("#### 🎯 Try These Questions")
    exam_questions = [
        "How many days of Privilege Leave (PL) does a permanent employee get per year?",
        "What is the probation period for employees up to Senior Manager?",
        "What is the company's health-insurance premium amount?",
        "What is the current GST rate in India?",
        "How many days of maternity leave are given?",
    ]
    for q in exam_questions:
        if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    st.session_state.top_k = st.slider("Retrieval top-k", 3, 12, RETRIEVER_TOP_K)
    st.session_state.show_evidence = st.toggle("Show retrieved evidence", value=True)

    st.markdown("---")
    store, doc_count = load_store()
    if store:
        st.markdown("#### 📊 Index Status")
        st.success(f"✅ {doc_count} chunks indexed")
        st.caption(f"Model: `{GEMINI_MODEL}`")
        st.caption(f"Embeddings: `{EMBEDDING_MODEL.split('/')[-1]}`")
    else:
        st.error("❌ Index not built")
        st.info("Run: `python -m src.ingest --rebuild`")
        if st.button("🔨 Build Index Now"):
            with st.spinner("Building index... This may take a minute."):
                from src.ingest import run_ingestion
                run_ingestion(rebuild=True)
                st.cache_resource.clear()
                st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main content ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📋 Enso Group HR Policy Assistant</h1>
    <p>Ask anything about company policy. Answers are grounded in the HR Manual.</p>
</div>
""", unsafe_allow_html=True)

if store is None:
    st.warning(
        "⚠️ The vector index has not been built yet. "
        "Run `python -m src.ingest --rebuild` or use the button in the sidebar."
    )
    st.stop()

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg, idx)

# Handle pending question from sidebar
pending = st.session_state.pop("pending_question", None)

# Chat input
user_input = st.chat_input("Ask about HR policy...")
question = pending or user_input

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching the HR Manual..."):
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
