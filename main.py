import os
import sqlite3
import json
import logging
import traceback
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PDFReader
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

# ─────────────────────────────────────────────────────────────────────────────
# Zotero helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_zotero_collections() -> list[dict]:
    """Return all top-level and nested collections with their names."""
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True, timeout=5)
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
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True, timeout=5)
    cur = conn.cursor()
    # PDF attachments whose parent item is in the collection
    cur.execute("""
        SELECT i.key
        FROM collectionItems ci
        JOIN itemAttachments ia ON ci.itemID = ia.parentItemID
        JOIN items i ON ia.itemID = i.itemID
        WHERE ci.collectionID = ?
          AND ia.contentType = 'application/pdf'
    """, (collection_id,))
    keys = [r[0] for r in cur.fetchall()]

    # PDF attachments that are themselves directly in the collection
    cur.execute("""
        SELECT i.key
        FROM collectionItems ci
        JOIN items i ON ci.itemID = i.itemID
        JOIN itemAttachments ia ON ia.itemID = i.itemID
        WHERE ci.collectionID = ?
          AND ia.contentType = 'application/pdf'
    """, (collection_id,))
    keys += [r[0] for r in cur.fetchall()]
    conn.close()

    paths = []
    for key in set(keys):
        folder = ZOTERO_STORE / key
        if folder.exists():
            for pdf in folder.glob("*.pdf"):
                paths.append(pdf)
    return paths


def collection_chroma_name(collection_id: int) -> str:
    return f"zotero_collection_{collection_id}"


def index_collection(collection_id: int, collection_name: str) -> dict:
    """Index all PDFs in a collection into ChromaDB."""
    pdfs = get_pdfs_for_collection(collection_id)
    if not pdfs:
        return {"indexed": 0, "message": "No PDFs found in this collection"}

    chroma_name = collection_chroma_name(collection_id)

    # Delete existing collection to re-index fresh
    try:
        chroma_client.delete_collection(chroma_name)
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(chroma_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = PDFReader()
    all_docs = []
    loaded = []
    failed = []

    for pdf_path in pdfs:
        try:
            docs = reader.load_data(file=pdf_path)
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["collection"] = collection_name
            all_docs.extend(docs)
            loaded.append(pdf_path.name)
        except Exception as e:
            failed.append({"file": pdf_path.name, "error": str(e)})

    if all_docs:
        VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_context,
            show_progress=True,
        )

    return {
        "indexed": len(loaded),
        "files": loaded,
        "failed": failed,
    }


def get_query_engine(collection_id: int):
    chroma_name = collection_chroma_name(collection_id)
    try:
        chroma_collection = chroma_client.get_collection(chroma_name)
    except Exception:
        return None
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
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
                if fname not in seen:
                    seen.add(fname)
                    sources.append({
                        "file": fname,
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
