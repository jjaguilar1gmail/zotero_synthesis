import os
import sqlite3
import json
import logging
import re
import traceback
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
from llama_index.core.schema import TextNode
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


def get_query_engine(collection_id: int):
    chroma_name = collection_chroma_name(collection_id)
    try:
        chroma_collection = chroma_client.get_collection(chroma_name)
    except Exception:
        return None
    vector_store = SafeChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index.as_query_engine(similarity_top_k=6, response_mode="tree_summarize")


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
    engine = get_query_engine(req.collection_id)
    if engine is None:
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

    prompt = req.message
    if history_text:
        prompt = (
            f"Conversation so far:\n{history_text}\n"
            f"Now answer this follow-up: {req.message}"
        )

    try:
        response = engine.query(prompt)
        sources = []
        if hasattr(response, "source_nodes"):
            seen = set()
            for node in response.source_nodes:
                fname = node.metadata.get("source_file", "Unknown")
                page_number = node.metadata.get("page_number")
                dedupe_key = (fname, page_number)
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    sources.append({
                        "file": fname,
                        "title": node.metadata.get("title"),
                        "year": node.metadata.get("year") if node.metadata.get("year") != -1 else None,
                        "page_number": page_number if page_number != -1 else None,
                        "section_label": node.metadata.get("section_label") or None,
                        "section_heading": node.metadata.get("section_heading") or None,
                        "journal_venue": node.metadata.get("journal_venue") or None,
                        "score": round(node.score, 3) if node.score else None,
                        "snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    })
        return {
            "response": str(response),
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "zotero_db_found": ZOTERO_DB.exists()}
