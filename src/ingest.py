"""
Ingestion pipeline: PDF → clean text → section-mapped chunks → ChromaDB.

Stages:
  1. Extract per-page text with PyMuPDF (blocks in reading order).
  2. Clean: remove footers, normalise bullets/hyphens/whitespace.
  3. Attempt table extraction (page.find_tables) for leave-policy pages.
  4. Chunk with RecursiveCharacterTextSplitter per page (metadata preserved).
  5. Add whole-page chunks for table-heavy pages so table data is never split.
  6. Persist to ChromaDB with idempotent rebuild support.

Usage:
  python -m src.ingest            # build if collection is empty
  python -m src.ingest --rebuild  # delete and rebuild from scratch
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATH,
    TABLE_PAGES,
)
from src.section_map import get_metadata_for_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Regex patterns for cleaning ──────────────────────────────────────
_FOOTER_RE = re.compile(r"HR\s*Manual\s*-\s*\d+", re.IGNORECASE)
_BULLET_CHARS = re.compile(r"[Ԃԃ¤▶•■►◆◇○●→‣⁃⊳⊲▸▹\u25B6\u25BA\u25CF\u2022]")
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(raw: str) -> str:
    """Clean extracted PDF text: fix bullets, hyphens, whitespace, footers."""
    text = raw

    # Remove footer lines  ("HR Manual - 60")
    text = _FOOTER_RE.sub("", text)

    # Map odd bullet glyphs to a simple dash
    text = _BULLET_CHARS.sub("-", text)

    # Fix hyphenated line breaks ("en-\ncashed" → "encashed")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # Collapse multiple spaces / excessive newlines
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def _extract_table_text(page: fitz.Page) -> str:
    """
    Try to extract tables via page.find_tables() and convert to readable
    pipe-separated rows.  Returns empty string if no tables found or
    extraction is too noisy.
    """
    try:
        tables = page.find_tables()
        if not tables.tables:
            return ""
        lines: List[str] = []
        for table in tables.tables:
            rows = table.extract()
            for row in rows:
                # Clean each cell
                cells = []
                for cell in row:
                    if cell is None:
                        cells.append("")
                    else:
                        c = str(cell)
                        # Remove excessive whitespace in cell
                        c = re.sub(r"\s+", " ", c).strip()
                        # Remove soft-hyphen artifacts
                        c = c.replace("\xad", "")
                        cells.append(c)
                # Skip rows that are mostly empty or are just the footer
                non_empty = [c for c in cells if c]
                if non_empty and not _FOOTER_RE.match(" ".join(non_empty)):
                    lines.append(" | ".join(cells))
            lines.append("")  # blank between tables
        result = "\n".join(lines).strip()
        # Only return if the table text is substantial enough
        if len(result) > 50:
            return result
    except Exception as e:
        logger.debug("Table extraction failed on page: %s", e)
    return ""


def _build_clean_page_text(page: fitz.Page, page_num: int) -> str:
    """
    Build clean page text.  For pages that contain important tables,
    also append readable table-row text.
    """
    # Get main text (blocks sorted in reading order)
    raw = page.get_text("text", sort=True)
    clean = _clean_text(raw)

    # For table-heavy pages, also try structured table extraction
    if (page_num - 1) in TABLE_PAGES:  # TABLE_PAGES is 0-indexed
        table_text = _extract_table_text(page)
        if table_text:
            table_clean = _clean_text(table_text)
            clean += "\n\n[TABLE DATA]\n" + table_clean

    return clean


def extract_pages() -> List[Document]:
    """
    Extract and clean every page from the PDF, returning a list of
    LangChain Documents with page-level metadata.
    """
    if not PDF_PATH.exists():
        logger.error("PDF not found at %s", PDF_PATH)
        sys.exit(1)

    doc = fitz.open(str(PDF_PATH))
    logger.info("Opened PDF: %d pages", len(doc))

    documents: List[Document] = []
    for idx in range(len(doc)):
        page = doc[idx]
        page_num = idx + 1  # 1-indexed, matches footer "HR Manual - N"
        text = _build_clean_page_text(page, page_num)

        if not text.strip():
            logger.debug("Skipping empty page %d", page_num)
            continue

        meta = get_metadata_for_page(page_num)
        documents.append(Document(page_content=text, metadata=meta))

    doc.close()
    logger.info("Extracted %d non-empty pages", len(documents))
    return documents


def chunk_documents(pages: List[Document]) -> List[Document]:
    """
    Split page-level documents into smaller chunks while preserving
    metadata.  Also adds whole-page chunks for table-heavy pages so
    table answers are never split apart.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Document] = []
    chunk_id = 0

    for page_doc in pages:
        page_num = page_doc.metadata["page"]
        section = page_doc.metadata["section"]
        policy = page_doc.metadata["policy_title"]

        # Header line prepended to each chunk for retrieval quality
        header = f"[{section} > {policy} | Page {page_num}]"

        # Split page into sub-chunks
        sub_chunks = splitter.split_text(page_doc.page_content)

        for sc in sub_chunks:
            chunk_text = f"{header}\n{sc}"
            meta = {
                **page_doc.metadata,
                "chunk_id": f"p{page_num}_c{chunk_id}",
                "is_whole_page": False,
            }
            all_chunks.append(Document(page_content=chunk_text, metadata=meta))
            chunk_id += 1

        # Whole-page chunk for table pages (leave policy pages 60-62, etc.)
        if (page_num - 1) in TABLE_PAGES and len(page_doc.page_content) > 100:
            whole_text = f"{header}\n{page_doc.page_content}"
            meta = {
                **page_doc.metadata,
                "chunk_id": f"p{page_num}_whole",
                "is_whole_page": True,
            }
            all_chunks.append(Document(page_content=whole_text, metadata=meta))
            chunk_id += 1

    logger.info("Created %d chunks (including whole-page chunks)", len(all_chunks))
    return all_chunks


def build_vector_store(chunks: List[Document], rebuild: bool = False) -> Chroma:
    """
    Embed chunks and persist to ChromaDB.
    If rebuild=True, deletes existing collection first.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if rebuild:
        # Remove existing data
        import shutil
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info("Removed existing ChromaDB at %s", CHROMA_DIR)

    logger.info("Building ChromaDB vector store (%d chunks)...", len(chunks))
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    count = vector_store._collection.count()
    logger.info("ChromaDB collection '%s' now has %d documents", COLLECTION_NAME, count)
    return vector_store


def run_ingestion(rebuild: bool = False) -> None:
    """Full ingestion pipeline: extract → chunk → embed → persist."""
    # Check if already indexed (skip if not rebuilding)
    if not rebuild:
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists():
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                existing = Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings,
                )
                count = existing._collection.count()
                if count > 0:
                    logger.info(
                        "Collection '%s' already has %d documents. "
                        "Use --rebuild to re-index.",
                        COLLECTION_NAME,
                        count,
                    )
                    return
            except Exception:
                pass  # collection doesn't exist yet, proceed

    pages = extract_pages()
    chunks = chunk_documents(pages)
    build_vector_store(chunks, rebuild=rebuild)
    logger.info("Ingestion complete!")


# ── CLI entry point ──────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HR Manual PDF into ChromaDB")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    args = parser.parse_args()
    run_ingestion(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
