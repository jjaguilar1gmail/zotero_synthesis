# Zotero Synthesis

Chat with your Zotero collections using **Llama 3.3 70B** (via OpenRouter) and local embeddings.

## Stack

| Component | Role |
|---|---|
| **LlamaIndex** | RAG orchestration |
| **meta-llama/llama-3.3-70b-instruct** via OpenRouter | Answers and synthesis |
| **nomic-embed-text via Ollama** | Local embeddings (free, private) |
| **ChromaDB** | Persistent vector store (indexed once, reused) |
| **FastAPI** | Local backend server |

---

## Project Structure

```
zotero_synthesis/
├── main.py           # FastAPI backend
├── index.html        # Frontend UI (open directly in browser)
├── requirements.txt
├── .env              # Your API key (create from .env.example, not committed)
└── .env.example      # Template for .env
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Zotero desktop app with your library synced locally
- [OpenRouter](https://openrouter.ai) account and API key

### 2. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running (`ollama serve` or via the desktop app).

### 3. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks script execution, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your API key

Copy the example env file and fill in your OpenRouter key:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 6. (Optional) Set your Zotero path

By default the app looks at `~/Zotero`. If yours is elsewhere, add it to `.env`:

```
ZOTERO_PATH=C:\Users\you\Documents\Zotero
```

### 7. Start the backend

From the `zotero_synthesis` directory (with your venv active):

```bash
uvicorn main:app --reload --port 8000
```

### 8. Open the UI

Open `index.html` directly in your browser — no separate frontend server needed.

---

## Usage

1. **Pick a collection** from the sidebar
2. **Click "Index Collection"** — reads all PDFs in that Zotero folder, detects academic sections, creates section-aware passage chunks, attaches Zotero metadata to every chunk, and stores embeddings locally in ChromaDB. Only needs to run once per collection (or after adding new papers)
3. **Chat** — ask anything across all papers in the collection. Sources with relevance scores appear under each response

### Example questions

- *"What are the core contributions across these papers?"*
- *"What gaps exist that I could address in a proposal?"*
- *"Which papers address [specific technique] and what do they conclude?"*
- *"Summarize the main methodologies used"*
- *"What datasets are commonly used and what are their limitations?"*

---

## Notes

- Chat history is kept in-memory per session for follow-up questions
- Embeddings are persisted in `chroma_db/` — re-indexing overwrites the existing index for that collection
- Indexed chunks now carry Zotero metadata including title, authors, year, venue, abstract, tags, paper key, attachment key, collection id, and page number
- Ingestion now chunks papers by detected sections such as Abstract, Introduction, Methods, Results, Discussion, and Conclusion before creating passage-level retrieval chunks
- The Zotero database is opened read-only, but avoid having Zotero running during a large indexing job to prevent lock conflicts
- PDFs stored in Zotero's linked-file mode may need path adjustments in `main.py`
