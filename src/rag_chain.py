"""
RAG chain: grounded Q&A with structured output.

Architecture:
  1. Retrieve top-k relevant chunks from ChromaDB.
  2. Build a strict grounded prompt forbidding outside knowledge.
  3. Call Gemini with structured output (Pydantic model).
  4. Two-layer "Not found" guard:
     (a) If no chunks pass the relevance filter → Not found.
     (b) The LLM itself sets found=false when context lacks the answer.
  5. Format the final response with citations or exactly "Not found".
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_TEMPERATURE,
    MIN_RELEVANCE_SCORE,
    RETRIEVER_TOP_K,
)
from src.retriever import retrieve, retrieve_with_store

logger = logging.getLogger(__name__)


# ── Structured output schema ────────────────────────────────────────
class Citation(BaseModel):
    """A citation pointing to a specific page and section of the HR Manual."""
    page: int = Field(description="Page number in the HR Manual (1-indexed)")
    section: str = Field(description="Section or Annexure label, e.g. 'Annexure III'")
    policy: str = Field(description="Policy title, e.g. '[G] Leave Policy'")


class RAGAnswer(BaseModel):
    """Structured answer from the RAG chain."""
    found: bool = Field(
        description=(
            "True if the answer was found in the provided context. "
            "False if the context does not explicitly contain the answer."
        )
    )
    answer: str = Field(
        description=(
            "The concise answer (1-2 sentences) if found=True. "
            "Include key conditions (e.g. 'per calendar year', 'no carry-forward'). "
            "If found=False, this must be exactly 'Not found'."
        )
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations for the pages used. Empty list if found=False.",
    )


# ── Prompt template ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an HR Policy Assistant for Enso Group. You answer questions ONLY using the provided context from the Enso Group HR Policy Manual.

STRICT RULES:
1. Answer ONLY from the context below. Do NOT use any outside knowledge.
2. If the context does NOT explicitly contain the answer, you MUST set found=false and answer="Not found".
3. Do NOT infer, guess, or estimate numbers or facts that are not explicitly written in the context.
4. Do NOT answer questions about topics outside the HR Manual (e.g., GST rates, stock prices, general knowledge).
5. If the context discusses a related topic but does NOT contain the specific information asked about, set found=false.
6. When found=true, give a concise answer (1-2 sentences) with key conditions. Cite only the pages from the context chunks that support your answer.
7. When found=false, set answer to exactly "Not found" and citations to an empty list.

CONTEXT FROM HR MANUAL:
{context}
"""

USER_PROMPT = """Question: {question}

Based ONLY on the context above, provide your answer. Remember: if the specific information is not in the context, found must be false and answer must be "Not found"."""


class _MockStructuredLLM:
    """Offline mock LLM for testing when GOOGLE_API_KEY is not configured."""
    def invoke(self, messages: list) -> RAGAnswer:
        user_content = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                user_content = m.get("content", "")
            elif hasattr(m, "content"):
                user_content += str(m.content)

        import re
        q_lower = user_content.lower()
        if "probation" in q_lower:
            return RAGAnswer(
                found=True,
                answer="The probation period for employees up to Senior Manager is 3 months from the date of joining. Employees above Senior Manager are confirmed from joining.",
                citations=[Citation(page=42, section="Annexure I", policy="[B] Probation Policy")],
            )
        elif "privilege leave" in q_lower or re.search(r"\bpl\b", q_lower):
            return RAGAnswer(
                found=True,
                answer="Permanent employees receive 12 days of Privilege Leave (PL) per calendar year (1 Jan to 31 Dec). Unavailed PL cannot be accumulated or carried forward.",
                citations=[Citation(page=60, section="Annexure III", policy="[G] Leave Policy")],
            )
        elif "maternity" in q_lower:
            return RAGAnswer(
                found=True,
                answer="Eligible female employees receive 12 weeks (90 days) of paid maternity leave, of which at most 6 weeks (45 days) may precede the expected delivery date.",
                citations=[Citation(page=62, section="Annexure III", policy="[G] Leave Policy")],
            )
        elif "casual leave" in q_lower or re.search(r"\bcl\b", q_lower):
            return RAGAnswer(
                found=True,
                answer="Employees are entitled to 6 days of Casual Leave per calendar year.",
                citations=[Citation(page=60, section="Annexure III", policy="[G] Leave Policy")],
            )
        elif "sick leave" in q_lower or re.search(r"\bsl\b", q_lower):
            return RAGAnswer(
                found=True,
                answer="Employees are entitled to 6 days of Sick Leave per calendar year.",
                citations=[Citation(page=61, section="Annexure III", policy="[G] Leave Policy")],
            )
        elif "stress leave" in q_lower:
            return RAGAnswer(
                found=True,
                answer="Employees are entitled to 6 days of Stress Leave per calendar year.",
                citations=[Citation(page=61, section="Annexure III", policy="[G] Leave Policy")],
            )
        elif "leave year" in q_lower:
            return RAGAnswer(
                found=True,
                answer="The leave year is calculated based on the calendar year from 1st January to 31st December.",
                citations=[Citation(page=60, section="Annexure III", policy="[G] Leave Policy")],
            )
        else:
            return RAGAnswer(found=False, answer="Not found", citations=[])


class _MockLLM:
    def with_structured_output(self, schema):
        return _MockStructuredLLM()


def _get_llm() -> Any:
    """Initialize the Gemini LLM (or mock for offline tests)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    if model in ("gemini-2.0-flash", "gemini-2.0-flash-exp"):
        model = "gemini-3.5-flash"
    temp = float(os.getenv("LLM_TEMPERATURE", str(LLM_TEMPERATURE)))

    if not api_key:
        if os.getenv("MOCK_LLM_FOR_TESTS") == "1":
            return _MockLLM()
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file. "
            "See .env.example for the format."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temp,
        convert_system_message_to_human=False,
    )


def _format_context(results: List[Tuple[Document, float]]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        header = (
            f"[Chunk {i+1} | Page {meta.get('page', '?')} | "
            f"{meta.get('section', '?')} > {meta.get('policy_title', '?')} | "
            f"Relevance: {score:.2f}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_final_response(rag_answer: RAGAnswer) -> str:
    """Format the RAGAnswer into a human-readable response string."""
    if not rag_answer.found:
        return "Not found"

    response = rag_answer.answer

    # Add references
    if rag_answer.citations:
        refs = []
        for cite in rag_answer.citations:
            refs.append(f"{cite.section} → {cite.policy}, Page {cite.page}")
        ref_str = "; ".join(refs)
        response += f"\n\nReference: {ref_str}"

    return response


def ask(
    question: str,
    store: Optional[Chroma] = None,
    top_k: int = RETRIEVER_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → prompt → LLM → structured answer.

    Args:
        question: The user's question.
        store: Pre-loaded ChromaDB store (optional, loaded fresh if None).
        top_k: Number of chunks to retrieve.
        min_score: Minimum relevance score threshold.

    Returns:
        Dict with keys:
          - "answer": formatted answer string
          - "found": bool
          - "citations": list of citation dicts
          - "raw_answer": RAGAnswer object
          - "retrieved_chunks": list of (doc, score) tuples
          - "latency_s": response time in seconds
    """
    start_time = time.time()

    # ── Layer 1: Retrieval ──────────────────────────────────────────
    if store is not None:
        results = retrieve_with_store(store, question, top_k=top_k, min_score=min_score)
    else:
        results = retrieve(question, top_k=top_k, min_score=min_score)

    # Layer 1 guard: no relevant chunks at all
    if not results:
        latency = round(time.time() - start_time, 2)
        logger.info("No relevant chunks found (layer 1 guard). Returning 'Not found'.")
        return {
            "answer": "Not found",
            "found": False,
            "citations": [],
            "raw_answer": RAGAnswer(found=False, answer="Not found", citations=[]),
            "retrieved_chunks": [],
            "latency_s": latency,
        }

    # ── Layer 2: LLM grounded answer ───────────────────────────────
    context_str = _format_context(results)
    llm = _get_llm()

    # Use structured output with Pydantic
    structured_llm = llm.with_structured_output(RAGAnswer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
        {"role": "user", "content": USER_PROMPT.format(question=question)},
    ]

    try:
        rag_answer: RAGAnswer = structured_llm.invoke(messages)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        # Retry once
        try:
            logger.info("Retrying LLM call...")
            rag_answer = structured_llm.invoke(messages)
        except Exception as e2:
            logger.error("LLM retry also failed: %s", e2)
            latency = round(time.time() - start_time, 2)
            return {
                "answer": f"Error: Could not get a response from the LLM. {e2}",
                "found": False,
                "citations": [],
                "raw_answer": None,
                "retrieved_chunks": results,
                "latency_s": latency,
            }

    # Format the response
    formatted = _format_final_response(rag_answer)
    latency = round(time.time() - start_time, 2)

    logger.info(
        "Answer: found=%s, latency=%.2fs, citations=%d",
        rag_answer.found,
        latency,
        len(rag_answer.citations),
    )

    return {
        "answer": formatted,
        "found": rag_answer.found,
        "citations": [c.model_dump() for c in rag_answer.citations],
        "raw_answer": rag_answer,
        "retrieved_chunks": results,
        "latency_s": latency,
    }
