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
3. **Chat** — ask anything across all papers in the collection. Retrieval now does a lightweight multi-stage pass: query-aware metadata filtering, dense retrieval, paper grouping, heuristic reranking, and evidence synthesis. Answers are now returned with a grounded contract: summary text, claim-level source citations, and confidence labels. Sources with relevance scores appear under each response

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
- Retrieval is now paper-aware rather than raw top-k only: it retrieves a larger dense candidate set, applies query-driven metadata constraints when available, groups evidence by paper, reranks candidates, and then synthesizes from the selected evidence set
- Answer synthesis now uses a grounding contract instead of freeform prose only: each claim is tied to source ids from the selected evidence set, includes a confidence label, and falls back to a deterministic evidence-only grounded response if the LLM output fails validation
- The Zotero database is opened read-only, but avoid having Zotero running during a large indexing job to prevent lock conflicts
- PDFs stored in Zotero's linked-file mode may need path adjustments in `main.py`

## Development

Run parser tests:

```bash
pytest -q
```

Inspect how the section parser splits a committed fixture:

```bash
python tools/debug_document_structure.py --fixture tests/fixtures/document_structure/numbered_subsections.json
```

Inspect a real PDF and optionally write a JSON report for diffing:

```bash
python tools/debug_document_structure.py --file path/to/paper.pdf --title "Paper Title" --json-out debug_output/paper-report.json
```

Write a static HTML review report for a single paper or fixture:

```bash
python tools/debug_document_structure.py --fixture tests/fixtures/document_structure/numbered_subsections.json --html-out debug_output/fixture-report.html
```

Batch-review a local PDF corpus and generate per-paper reports plus a summary:

```bash
python tools/review_document_structure_batch.py --input-dir path/to/pdf/corpus --recursive
```

That batch review now also writes an HTML index and one HTML report per paper.

Run a retrieval evaluation question set against an indexed Zotero collection:

```bash
python tools/evaluate_retrieval.py --collection-id 123 --questions-file docs/architecture/retrieval_eval_question_set.example.json --output-dir debug_output/retrieval_eval
```

Add `--with-answer` if you also want grounded synthesized answers captured alongside the retrieval evidence for each evaluation query. The evaluation output now also records whether the generated answer satisfied the claim-citation contract.

Generate a structurally diverse recommendation shortlist directly from your Zotero library:

```bash
python tools/select_document_structure_candidates.py --max-per-collection 3 --recommendation-count 10
```

See [docs/architecture/document_structure_review_workflow.md](docs/architecture/document_structure_review_workflow.md) for the gold-fixture versus local-review workflow.
