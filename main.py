import os
import sqlite3
import json
import logging
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.chroma.base import _to_chroma_filter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PDFReader
from ingestion.document_structure import PaperMetadata, PaperRecord, create_passage_chunks, extract_sections_from_pages, parse_page_number
import chromadb

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
ZOTERO_BASE       = Path(os.getenv("ZOTERO_PATH", str(Path.home() / "Zotero")))
ZOTERO_DB         = ZOTERO_BASE / "zotero.sqlite"
ZOTERO_STORE      = ZOTERO_BASE / "storage"
CHROMA_PATH       = Path(__file__).parent / "chroma_db"
OPENROUTER_KEY    = os.getenv("OPENROUTER_API_KEY", "")

# ── LlamaIndex global settings ────────────────────────────────────────────────
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    ollama_additional_kwargs={"num_ctx": 8192},
)
Settings.llm = OpenAILike(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=OPENROUTER_KEY,
    api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    is_chat_model=True,
    context_window=131072,
)
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)

# ── Chroma client (persistent) ────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# ── In-memory chat history per collection ────────────────────────────────────
chat_histories: dict[str, list[dict]] = {}

CORE_RETRIEVAL_SECTION_HINTS = {
    "Abstract": {"abstract"},
    "Introduction": {"introduction", "overview", "motivation"},
    "Background": {"background"},
    "Related Work": {"related work", "prior work", "literature review"},
    "Methods": {"method", "methods", "methodology", "approach", "implementation", "experimental setup"},
    "Results": {"result", "results", "evaluation", "experiment", "experiments", "findings"},
    "Discussion": {"discussion", "analysis", "interpretation"},
    "Conclusion": {"conclusion", "conclusions", "future work", "summary"},
}

END_MATTER_LABELS = {
    "references",
    "acknowledgements",
    "acknowledgments",
    "author contributions",
    "competing interests",
    "data availability",
}

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i", "in",
    "into", "is", "it", "me", "of", "on", "or", "our", "paper", "papers", "show", "summarize",
    "tell", "that", "the", "their", "them", "these", "this", "to", "using", "what", "which", "with",
    "would", "you",
}

COMPARATIVE_QUERY_MARKERS = {
    "across", "compare", "comparison", "contrast", "contradiction", "differences", "different", "similarities", "versus", "vs",
}


@dataclass
class RetrievalPlan:
    query_text: str
    section_labels: list[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    comparative: bool = False
    dense_top_k: int = 14
    target_papers: int = 3
    per_paper_limit: int = 2
    max_evidence_chunks: int = 6


@dataclass
class EvidenceChunk:
    node: NodeWithScore
    dense_score: float
    rerank_score: float
    paper_id: str
    title: str
    year: Optional[int]
    section_label: Optional[str]
    section_heading: Optional[str]
    source_id: str = ""


@dataclass
class RetrievalResult:
    plan: RetrievalPlan
    dense_candidates: list[NodeWithScore]
    filtered_candidates: list[NodeWithScore]
    reranked_candidates: list[EvidenceChunk]
    selected_chunks: list[EvidenceChunk]



class SafeChromaVectorStore(ChromaVectorStore):
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify metadata filters via both query and kwargs. "
                    "Use kwargs only for chroma specific items that are not supported via the generic query interface."
                )
            where = _to_chroma_filter(query.filters)
        else:
            where = kwargs.pop("where", None)

        if not query.query_embedding:
            return self._get(limit=query.similarity_top_k, where=where, **kwargs)

        return self._query(
            query_embeddings=query.query_embedding,
            n_results=query.similarity_top_k,
            where=where,
            **kwargs,
        )


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokenize_query_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower())
        if token not in QUERY_STOPWORDS
    }


def infer_section_filters(query_text: str) -> list[str]:
    lowered = query_text.lower()
    section_labels: list[str] = []
    for label, hints in CORE_RETRIEVAL_SECTION_HINTS.items():
        if any(hint in lowered for hint in hints):
            section_labels.append(label)
    return section_labels


def infer_year_filters(query_text: str) -> tuple[Optional[int], Optional[int]]:
    lowered = query_text.lower()

    between_match = re.search(r"between\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})", lowered)
    if between_match:
        return int(between_match.group(1)), int(between_match.group(2))

    range_match = re.search(r"from\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})", lowered)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    since_match = re.search(r"(?:since|after)\s+((?:19|20)\d{2})", lowered)
    if since_match:
        return int(since_match.group(1)), None

    before_match = re.search(r"(?:before|until|up to)\s+((?:19|20)\d{2})", lowered)
    if before_match:
        return None, int(before_match.group(1))

    return None, None


def infer_retrieval_plan(query_text: str) -> RetrievalPlan:
    comparative = any(marker in query_text.lower() for marker in COMPARATIVE_QUERY_MARKERS)
    section_labels = infer_section_filters(query_text)
    year_from, year_to = infer_year_filters(query_text)
    return RetrievalPlan(
        query_text=query_text,
        section_labels=section_labels,
        year_from=year_from,
        year_to=year_to,
        comparative=comparative,
        dense_top_k=20 if comparative else 14,
        target_papers=4 if comparative else 3,
        per_paper_limit=2 if comparative else 2,
        max_evidence_chunks=8 if comparative else 6,
    )


def metadata_matches_plan(metadata: dict, plan: RetrievalPlan) -> bool:
    year = metadata.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    if year == -1:
        year = None

    if plan.year_from is not None and (year is None or year < plan.year_from):
        return False
    if plan.year_to is not None and (year is None or year > plan.year_to):
        return False

    if plan.section_labels:
        section_label = normalize_text(metadata.get("section_label")).lower()
        section_heading = normalize_text(metadata.get("section_heading")).lower()
        section_path = normalize_text(metadata.get("section_path")).lower()
        if not any(
            label.lower() == section_label
            or label.lower() in section_heading
            or label.lower() in section_path
            for label in plan.section_labels
        ):
            return False

    return True


def get_collection_index(collection_id: int) -> Optional[VectorStoreIndex]:
    chroma_name = collection_chroma_name(collection_id)
    try:
        chroma_collection = chroma_client.get_collection(chroma_name)
    except Exception:
        return None

    vector_store = SafeChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def lexical_overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = tokenize_query_terms(text)
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def rerank_evidence_candidates(query_text: str, plan: RetrievalPlan, candidates: list[NodeWithScore]) -> list[EvidenceChunk]:
    query_terms = tokenize_query_terms(query_text)
    reranked: list[EvidenceChunk] = []

    for candidate in candidates:
        metadata = candidate.node.metadata or {}
        title = normalize_text(metadata.get("title"))
        section_label = normalize_text(metadata.get("section_label")) or None
        section_heading = normalize_text(metadata.get("section_heading")) or None
        section_path = normalize_text(metadata.get("section_path"))
        paper_id = normalize_text(metadata.get("paper_node_id") or metadata.get("parent_paper_id") or title or candidate.node.node_id)
        dense_score = float(candidate.score or 0.0)

        overlap = lexical_overlap_score(query_terms, f"{title} {section_heading or ''} {section_path} {candidate.text[:800]}")
        title_overlap = lexical_overlap_score(query_terms, title)
        section_overlap = lexical_overlap_score(query_terms, f"{section_label or ''} {section_heading or ''}")

        score = dense_score
        score += overlap * 0.35
        score += title_overlap * 0.15
        score += section_overlap * 0.1

        if plan.section_labels and section_label and section_label in plan.section_labels:
            score += 0.12
        if section_label and section_label.lower() in END_MATTER_LABELS:
            score -= 0.2

        reranked.append(
            EvidenceChunk(
                node=candidate,
                dense_score=dense_score,
                rerank_score=score,
                paper_id=paper_id,
                title=title,
                year=metadata.get("year") if metadata.get("year") != -1 else None,
                section_label=section_label,
                section_heading=section_heading,
            )
        )

    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return reranked


def select_evidence_chunks(plan: RetrievalPlan, reranked_candidates: list[EvidenceChunk]) -> list[EvidenceChunk]:
    grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for candidate in reranked_candidates:
        grouped[candidate.paper_id].append(candidate)

    ranked_papers = sorted(
        grouped.items(),
        key=lambda item: max(candidate.rerank_score for candidate in item[1]),
        reverse=True,
    )

    selected: list[EvidenceChunk] = []
    for _, paper_candidates in ranked_papers[: plan.target_papers]:
        selected.extend(paper_candidates[: plan.per_paper_limit])

    selected.sort(key=lambda item: item.rerank_score, reverse=True)
    selected = selected[: plan.max_evidence_chunks]

    for index, candidate in enumerate(selected, start=1):
        candidate.source_id = f"S{index}"

    return selected


def run_retrieval_pipeline(index: VectorStoreIndex, query_text: str) -> RetrievalResult:
    plan = infer_retrieval_plan(query_text)
    retriever = index.as_retriever(similarity_top_k=plan.dense_top_k)
    dense_candidates = list(retriever.retrieve(query_text))

    filtered_candidates = [candidate for candidate in dense_candidates if metadata_matches_plan(candidate.node.metadata or {}, plan)]
    if not filtered_candidates:
        filtered_candidates = dense_candidates

    reranked_candidates = rerank_evidence_candidates(query_text, plan, filtered_candidates)
    selected_chunks = select_evidence_chunks(plan, reranked_candidates)
    return RetrievalResult(
        plan=plan,
        dense_candidates=dense_candidates,
        filtered_candidates=filtered_candidates,
        reranked_candidates=reranked_candidates,
        selected_chunks=selected_chunks,
    )


def build_answer_prompt(question: str, history_text: str, retrieval: RetrievalResult) -> str:
    evidence_lines = []
    for chunk in retrieval.selected_chunks:
        metadata = chunk.node.node.metadata or {}
        page_number = metadata.get("page_number")
        page_text = f"p.{page_number}" if page_number not in (None, -1) else "page unknown"
        heading = chunk.section_heading or chunk.section_label or "Unlabeled section"
        evidence_lines.append(
            f"[{chunk.source_id}] {chunk.title} ({chunk.year or 'year unknown'}) | {heading} | {page_text}\n"
            f"Excerpt: {chunk.node.text[:700].strip()}"
        )

    plan_notes = []
    if retrieval.plan.section_labels:
        plan_notes.append(f"section focus={', '.join(retrieval.plan.section_labels)}")
    if retrieval.plan.year_from or retrieval.plan.year_to:
        plan_notes.append(
            f"year range={retrieval.plan.year_from or 'any'}..{retrieval.plan.year_to or 'any'}"
        )

    history_block = f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    retrieval_block = "\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."
    plan_block = f"Retrieval notes: {'; '.join(plan_notes)}\n\n" if plan_notes else ""

    return (
        "You are answering questions over a Zotero paper collection. Use only the provided evidence. "
        "Synthesize across papers when useful, prefer concrete claims, and say when the evidence is insufficient. "
        "Cite evidence inline using source ids like [S1] or [S2].\n\n"
        f"{history_block}"
        f"{plan_block}"
        f"Question: {question}\n\n"
        f"Evidence:\n{retrieval_block}\n\n"
        "Answer:"
    )


def synthesize_answer(question: str, history_text: str, retrieval: RetrievalResult) -> str:
    prompt = build_answer_prompt(question, history_text, retrieval)
    response = Settings.llm.complete(prompt)
    return getattr(response, "text", str(response)).strip()


def format_sources_from_evidence(chunks: list[EvidenceChunk]) -> list[dict]:
    sources = []
    for chunk in chunks:
        metadata = chunk.node.node.metadata or {}
        page_number = metadata.get("page_number")
        sources.append(
            {
                "source_id": chunk.source_id,
                "file": metadata.get("source_file", "Unknown"),
                "title": metadata.get("title"),
                "year": metadata.get("year") if metadata.get("year") != -1 else None,
                "page_number": page_number if page_number != -1 else None,
                "section_label": metadata.get("section_label") or None,
                "section_heading": metadata.get("section_heading") or None,
                "journal_venue": metadata.get("journal_venue") or None,
                "score": round(chunk.rerank_score, 3),
                "dense_score": round(chunk.dense_score, 3),
                "snippet": chunk.node.text[:240] + "..." if len(chunk.node.text) > 240 else chunk.node.text,
            }
        )
    return sources

# ─────────────────────────────────────────────────────────────────────────────
# Zotero helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_zotero_connection() -> sqlite3.Connection:
    return sqlite3.connect(
        f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True, timeout=5
    )


def extract_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None
def build_passage_nodes(paper_record: PaperRecord, collection_name: str) -> list[TextNode]:
    nodes: list[TextNode] = []
    for section in paper_record.sections:
        section.chunks = create_passage_chunks(section)
        for chunk in section.chunks:
            metadata = build_chunk_metadata(paper_record, collection_name, chunk.page_start)
            metadata.update(
                {
                    "section_label": section.label,
                    "section_heading": section.heading,
                    "section_path": " > ".join(section.path),
                    "section_node_id": section.section_id,
                    "section_order": section.order,
                    "section_level": section.level,
                    "section_page_start": section.page_start if section.page_start is not None else -1,
                    "section_page_end": section.page_end if section.page_end is not None else -1,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "node_kind": "passage",
                }
            )
            nodes.append(
                TextNode(
                    id_=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                )
            )
    return nodes


def format_creator_name(first_name: Optional[str], last_name: Optional[str], field_mode: int) -> str:
    if field_mode == 1:
        return (last_name or first_name or "").strip()
    return " ".join(part for part in [first_name, last_name] if part).strip()


def resolve_attachment_path(attachment_key: str, attachment_path: Optional[str]) -> Optional[Path]:
    if attachment_path:
        normalized = attachment_path.replace("\\", "/")
        if normalized.startswith("storage:"):
            relative_name = normalized.split(":", 1)[1].lstrip("/")
            candidate = ZOTERO_STORE / attachment_key / relative_name
            if candidate.exists():
                return candidate
        else:
            raw_path = attachment_path
            if raw_path.startswith("attachments:"):
                raw_path = raw_path.split(":", 1)[1]
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = ZOTERO_BASE / raw_path
            if candidate.exists():
                return candidate

    folder = ZOTERO_STORE / attachment_key
    if folder.exists():
        pdfs = sorted(folder.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
    return None


def get_item_field_map(cur: sqlite3.Cursor, item_id: int) -> dict[str, str]:
    cur.execute(
        """
        SELECT f.fieldName, v.value
        FROM itemData d
        JOIN fields f ON f.fieldID = d.fieldID
        JOIN itemDataValues v ON v.valueID = d.valueID
        WHERE d.itemID = ?
        """,
        (item_id,),
    )
    return {field_name: value for field_name, value in cur.fetchall() if value}


def get_item_creators(cur: sqlite3.Cursor, item_id: int) -> list[str]:
    cur.execute(
        """
        SELECT c.firstName, c.lastName, c.fieldMode
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        WHERE ic.itemID = ?
        ORDER BY ic.orderIndex
        """,
        (item_id,),
    )
    creators = []
    for first_name, last_name, field_mode in cur.fetchall():
        name = format_creator_name(first_name, last_name, field_mode)
        if name:
            creators.append(name)
    return creators


def get_item_tags(cur: sqlite3.Cursor, item_id: int) -> list[str]:
    cur.execute(
        """
        SELECT t.name
        FROM itemTags it
        JOIN tags t ON t.tagID = it.tagID
        WHERE it.itemID = ?
        ORDER BY t.name
        """,
        (item_id,),
    )
    return [tag for (tag,) in cur.fetchall() if tag]


def get_paper_records_for_collection(collection_id: int) -> list[PaperRecord]:
    conn = get_zotero_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT
            COALESCE(parent.itemID, attachment.itemID) AS paper_item_id,
            COALESCE(parent.key, attachment.key) AS paper_item_key,
            attachment.key AS attachment_key,
            ia.path AS attachment_path
        FROM collectionItems ci
        JOIN itemAttachments ia
            ON (ci.itemID = ia.parentItemID OR ci.itemID = ia.itemID)
        JOIN items attachment ON attachment.itemID = ia.itemID
        LEFT JOIN items parent ON parent.itemID = ia.parentItemID
        WHERE ci.collectionID = ?
          AND ia.contentType = 'application/pdf'
        ORDER BY paper_item_id
        """,
        (collection_id,),
    )

    records: list[PaperRecord] = []
    for paper_item_id, paper_item_key, attachment_key, attachment_path in cur.fetchall():
        file_path = resolve_attachment_path(attachment_key, attachment_path)
        if not file_path:
            logger.warning(
                "Skipping attachment %s for paper %s because the PDF path could not be resolved",
                attachment_key,
                paper_item_key,
            )
            continue

        field_map = get_item_field_map(cur, paper_item_id)
        title = field_map.get("title") or file_path.stem
        journal_venue = (
            field_map.get("publicationTitle")
            or field_map.get("proceedingsTitle")
            or field_map.get("bookTitle")
            or field_map.get("websiteTitle")
            or field_map.get("repository")
            or ""
        )

        metadata = PaperMetadata(
            zotero_item_key=paper_item_key,
            attachment_key=attachment_key,
            title=title,
            authors=get_item_creators(cur, paper_item_id),
            year=extract_year(field_map.get("date")),
            journal_venue=journal_venue,
            abstract=field_map.get("abstractNote", ""),
            tags=get_item_tags(cur, paper_item_id),
            collection_id=collection_id,
            parent_paper_id=paper_item_key,
        )
        records.append(
            PaperRecord(
                paper_id=paper_item_key,
                file_path=file_path,
                metadata=metadata,
            )
        )

    conn.close()
    return records


def build_chunk_metadata(
    paper_record: PaperRecord,
    collection_name: str,
    page_number: Optional[int],
) -> dict:
    return {
        "source_file": paper_record.file_path.name,
        "collection": collection_name,
        "zotero_item_key": paper_record.metadata.zotero_item_key,
        "attachment_key": paper_record.metadata.attachment_key,
        "title": paper_record.metadata.title,
        "authors": json.dumps(paper_record.metadata.authors, ensure_ascii=True),
        "year": paper_record.metadata.year if paper_record.metadata.year is not None else -1,
        "journal_venue": paper_record.metadata.journal_venue,
        "abstract": paper_record.metadata.abstract,
        "tags": json.dumps(paper_record.metadata.tags, ensure_ascii=True),
        "collection_id": paper_record.metadata.collection_id,
        "page_number": page_number if page_number is not None else -1,
        "section_label": paper_record.metadata.section_label,
        "parent_paper_id": paper_record.metadata.parent_paper_id,
        "paper_node_id": paper_record.paper_id,
    }

def get_zotero_collections() -> list[dict]:
    """Return all top-level and nested collections with their names."""
    conn = get_zotero_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT collectionID, collectionName, parentCollectionID
        FROM collections
        WHERE libraryID = 1
        ORDER BY collectionName
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "parent": r[2]} for r in rows]


def get_pdfs_for_collection(collection_id: int) -> list[Path]:
    """Return all PDF file paths for items in a Zotero collection."""
    return [record.file_path for record in get_paper_records_for_collection(collection_id)]


def collection_chroma_name(collection_id: int) -> str:
    return f"zotero_collection_{collection_id}"


def index_collection(collection_id: int, collection_name: str) -> dict:
    """Index all PDFs in a collection into ChromaDB."""
    paper_records = get_paper_records_for_collection(collection_id)
    if not paper_records:
        return {"indexed": 0, "message": "No PDFs found in this collection"}

    chroma_name = collection_chroma_name(collection_id)

    # Delete existing collection to re-index fresh
    try:
        chroma_client.delete_collection(chroma_name)
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(chroma_name)
    vector_store = SafeChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = PDFReader()
    all_nodes = []
    loaded = []
    failed = []

    for paper_record in paper_records:
        try:
            docs = reader.load_data(file=paper_record.file_path)
            page_entries: list[tuple[Optional[int], str]] = []
            for doc in docs:
                page_number = parse_page_number(
                    doc.metadata.get("page_label") or doc.metadata.get("page")
                )
                page_entries.append((page_number, doc.text))
            paper_record.parsed_text = "\n\n".join(text for _, text in page_entries if text)
            paper_record.sections = extract_sections_from_pages(paper_record, page_entries)
            passage_nodes = build_passage_nodes(paper_record, collection_name)
            paper_record.chunks = [node.text for node in passage_nodes]
            all_nodes.extend(passage_nodes)
            loaded.append(
                {
                    "file": paper_record.file_path.name,
                    "title": paper_record.metadata.title,
                    "paper_id": paper_record.paper_id,
                    "sections": len(paper_record.sections),
                    "chunks": len(passage_nodes),
                }
            )
        except Exception as e:
            failed.append(
                {
                    "file": paper_record.file_path.name,
                    "paper_id": paper_record.paper_id,
                    "error": str(e),
                }
            )

    if all_nodes:
        VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True,
        )

    return {
        "indexed": len(loaded),
        "papers": len(paper_records),
        "files": loaded,
        "failed": failed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/collections")
def list_collections():
    try:
        collections = get_zotero_collections()
        # Mark which are already indexed
        existing = {c.name for c in chroma_client.list_collections()}
        for col in collections:
            col["indexed"] = collection_chroma_name(col["id"]) in existing
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IndexRequest(BaseModel):
    collection_id: int
    collection_name: str

@app.post("/api/index")
def index_endpoint(req: IndexRequest):
    try:
        result = index_collection(req.collection_id, req.collection_name)
        return result
    except Exception as e:
        logger.error("index_endpoint error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    collection_id: int
    message: str
    history: Optional[list[dict]] = []

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    index = get_collection_index(req.collection_id)
    if index is None:
        raise HTTPException(
            status_code=400,
            detail="Collection not indexed yet. Please index it first."
        )

    # Build context-aware prompt from history
    history_text = ""
    if req.history:
        for turn in req.history[-6:]:  # last 6 turns for context
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

    try:
        retrieval = run_retrieval_pipeline(index, req.message)
        response_text = synthesize_answer(req.message, history_text, retrieval)
        sources = format_sources_from_evidence(retrieval.selected_chunks)
        return {
            "response": response_text,
            "sources": sources,
            "retrieval": {
                "metadata_filters": {
                    "section_labels": retrieval.plan.section_labels,
                    "year_from": retrieval.plan.year_from,
                    "year_to": retrieval.plan.year_to,
                },
                "comparative": retrieval.plan.comparative,
                "dense_candidates": len(retrieval.dense_candidates),
                "filtered_candidates": len(retrieval.filtered_candidates),
                "evidence_chunks": len(retrieval.selected_chunks),
                "papers_considered": len({chunk.paper_id for chunk in retrieval.selected_chunks}),
            },
        }
    except Exception as e:
        logger.error("chat_endpoint error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "zotero_db_found": ZOTERO_DB.exists()}
