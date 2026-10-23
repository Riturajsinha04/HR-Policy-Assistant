"""
Test suite for the HR Policy Assistant.
Tests the 5 exam questions plus additional in-document and out-of-scope questions.

Usage:
    pytest tests/test_questions.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# If no GOOGLE_API_KEY is configured yet, use mock LLM for offline test runs
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["MOCK_LLM_FOR_TESTS"] = "1"

from src.rag_chain import ask
from src.retriever import retrieve


# ── Helper ──────────────────────────────────────────────────────────
def _ask(question: str) -> dict:
    """Wrapper around ask() for tests."""
    return ask(question)


# ════════════════════════════════════════════════════════════════════
# VECTOR STORE & RETRIEVAL TESTS
# ════════════════════════════════════════════════════════════════════

class TestVectorRetrieval:
    """Tests that vector search correctly indexes and retrieves relevant manual chunks."""

    def test_privilege_leave_retrieval(self):
        """Privilege leave query must retrieve page 60 with high relevance."""
        results = retrieve("How many days of Privilege Leave does a permanent employee get per year?")
        assert len(results) > 0
        pages = [doc.metadata.get("page") for doc, _ in results]
        assert 60 in pages

    def test_probation_policy_retrieval(self):
        """Probation query must retrieve page 42 with high relevance."""
        results = retrieve("What is the probation period for employees up to Senior Manager?")
        assert len(results) > 0
        pages = [doc.metadata.get("page") for doc, _ in results]
        assert 42 in pages

    def test_maternity_leave_retrieval(self):
        """Maternity leave query must retrieve page 62 with high relevance."""
        results = retrieve("How many days of maternity leave are given?")
        assert len(results) > 0
        pages = [doc.metadata.get("page") for doc, _ in results]
        assert 62 in pages


# ════════════════════════════════════════════════════════════════════
# EXAM QUESTIONS (Q1-Q5)
# ════════════════════════════════════════════════════════════════════

class TestExamQuestions:
    """Tests for the 5 mandatory exam questions."""

    def test_q1_privilege_leave(self):
        """Q1: PL entitlement should be 12 days, page 60."""
        result = _ask(
            "How many days of Privilege Leave (PL) does a permanent employee get per year?"
        )
        assert result["found"] is True, f"Expected found=True, got {result}"
        assert "12" in result["answer"], f"Expected '12' in answer: {result['answer']}"
        # Check page reference
        pages = [c["page"] for c in result["citations"]]
        assert 60 in pages, f"Expected page 60 in citations, got {pages}"

    def test_q2_probation_period(self):
        """Q2: Probation period should be 3 months, page 42."""
        result = _ask(
            "What is the probation period for employees up to Senior Manager?"
        )
        assert result["found"] is True, f"Expected found=True, got {result}"
        assert "3" in result["answer"], f"Expected '3' in answer: {result['answer']}"
        assert "month" in result["answer"].lower(), f"Expected 'month' in answer: {result['answer']}"
        pages = [c["page"] for c in result["citations"]]
        assert 42 in pages, f"Expected page 42 in citations, got {pages}"

    def test_q3_health_insurance_not_found(self):
        """Q3: Health insurance premium should be 'Not found'."""
        result = _ask(
            "What is the company's health-insurance premium amount?"
        )
        assert result["found"] is False, f"Expected found=False, got {result}"
        assert "Not found" in result["answer"], f"Expected 'Not found' in answer: {result['answer']}"

    def test_q4_gst_rate_not_found(self):
        """Q4: GST rate is outside the document, should be 'Not found'."""
        result = _ask("What is the current GST rate in India?")
        assert result["found"] is False, f"Expected found=False, got {result}"
        assert "Not found" in result["answer"], f"Expected 'Not found' in answer: {result['answer']}"

    def test_q5_maternity_leave(self):
        """Q5: Maternity leave should mention 12 weeks or 90 days, page 62."""
        result = _ask("How many days of maternity leave are given?")
        assert result["found"] is True, f"Expected found=True, got {result}"
        answer_lower = result["answer"].lower()
        assert (
            "12 week" in answer_lower or "90" in answer_lower
        ), f"Expected '12 week' or '90' in answer: {result['answer']}"
        pages = [c["page"] for c in result["citations"]]
        assert 62 in pages, f"Expected page 62 in citations, got {pages}"


# ════════════════════════════════════════════════════════════════════
# ADDITIONAL IN-DOCUMENT QUESTIONS
# ════════════════════════════════════════════════════════════════════

class TestInDocumentQuestions:
    """Extra questions that should be answerable from the HR Manual."""

    def test_casual_leave_days(self):
        """CL should be 6 days."""
        result = _ask("How many days of Casual Leave is an employee entitled to?")
        assert result["found"] is True
        assert "6" in result["answer"]

    def test_sick_leave_days(self):
        """SL should be 6 days."""
        result = _ask("How many days of Sick Leave does an employee get?")
        assert result["found"] is True
        assert "6" in result["answer"]

    def test_leave_year(self):
        """Leave year should be calendar year (Jan-Dec)."""
        result = _ask("What is the leave year period as per the HR policy?")
        assert result["found"] is True
        answer_lower = result["answer"].lower()
        assert "calendar year" in answer_lower or ("january" in answer_lower and "december" in answer_lower) or ("jan" in answer_lower and "dec" in answer_lower)

    def test_stress_leave_days(self):
        """Stress Leave should be 6 days."""
        result = _ask("How many days of Stress Leave are given to employees?")
        assert result["found"] is True
        assert "6" in result["answer"]


# ════════════════════════════════════════════════════════════════════
# OUT-OF-SCOPE QUESTIONS (should all return "Not found")
# ════════════════════════════════════════════════════════════════════

class TestOutOfScopeQuestions:
    """Questions about topics not in the HR Manual should return 'Not found'."""

    def test_ceo_of_google(self):
        """General knowledge question should be Not found."""
        result = _ask("Who is the CEO of Google?")
        assert result["found"] is False
        assert "Not found" in result["answer"]

    def test_salary_of_cto(self):
        """Salary specifics not in the document should be Not found."""
        result = _ask("What is the salary of the CTO?")
        assert result["found"] is False
        assert "Not found" in result["answer"]

    def test_stock_price(self):
        """Stock market question should be Not found."""
        result = _ask("What is Enso Group's current stock price?")
        assert result["found"] is False
        assert "Not found" in result["answer"]
