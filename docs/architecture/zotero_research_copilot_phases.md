# Zotero Research Copilot -- Architecture Roadmap

This document describes the phased evolution of the current Zotero RAG
application into a full **research copilot system** capable of
structured paper understanding, cross‑paper synthesis, and agentic
research workflows.

The goal is **not to rewrite the system**, but to evolve it
incrementally while preserving the working MVP.

------------------------------------------------------------------------

# Phase 0 --- Current MVP (Baseline)

The current system is a clean personal RAG interface over Zotero
collections.

## Key Characteristics

-   FastAPI backend
-   LlamaIndex indexing pipeline
-   Chroma persistent vector store
-   Ollama local embeddings (`nomic-embed-text`)
-   Llama 3.3 70B via OpenRouter for synthesis
-   Per‑collection indexes
-   Simple chat UI
-   Source snippets returned with responses

## Strengths

-   Very clean architecture
-   Easy indexing workflow
-   Local embeddings (cheap + private)
-   Persistent vector DB
-   Collection‑scoped retrieval

## Limitations

-   Minimal metadata stored
-   Papers treated as flat PDF text
-   Single‑stage retrieval
-   No section awareness
-   No query routing
-   No LangGraph orchestration yet

------------------------------------------------------------------------

# Phase 1 --- Metadata‑Rich Ingestion

Upgrade ingestion so the system understands papers as structured objects
rather than raw PDFs.

## Goals

Introduce a **paper‑aware data model**.

## Changes

Add metadata fields to every chunk:

-   zotero_item_key
-   attachment_key
-   title
-   authors
-   year
-   journal / venue
-   abstract
-   tags
-   collection_id
-   page_number
-   section_label
-   parent_paper_id

## Architecture Change

Introduce a **PaperRecord** model:

PaperRecord - metadata - parsed text - chunks - optional generated
summary

## Benefits

-   Enables filtering by year, author, tags
-   Enables paper‑level grouping
-   Enables section‑aware retrieval

------------------------------------------------------------------------

# Phase 2 --- Improved Chunking & Document Structure

Replace naive sentence chunking with structure‑aware chunking.

## Goals

Preserve academic document structure.

## New Chunk Hierarchy

Paper Node → Section Node → Passage Chunk

### Paper Node

High level metadata and optional summary.

### Section Node

Represents: - Abstract - Introduction - Methods - Results - Discussion -
Conclusion

### Passage Chunk

Small retrieval unit (\~400--800 tokens).

## Benefits

-   Section‑aware retrieval
-   Better synthesis
-   Reduced hallucination risk

------------------------------------------------------------------------

# Phase 3 --- Retrieval Pipeline Upgrade

Move from **single‑stage RAG** to **multi‑stage retrieval**.

## Current Retrieval

Query → Top‑K chunks → LLM summarize

## New Retrieval Pipeline

1.  Metadata filtering
2.  Dense retrieval
3.  Candidate expansion
4.  Paper grouping
5.  Reranking
6.  Evidence synthesis

## Why This Matters

Academic questions often require: - comparing multiple papers -
extracting evidence - identifying contradictions

Flat top‑k retrieval performs poorly for those tasks.

------------------------------------------------------------------------

# Phase 4 --- Query Classification

Introduce lightweight routing before retrieval.

## Query Types

1.  Lookup
2.  Single‑paper explanation
3.  Cross‑paper comparison
4.  Topic synthesis

## Example

User question:

"Compare the datasets used in these papers"

The system should:

-   retrieve relevant papers
-   group evidence by paper
-   generate a structured comparison

## Implementation

Start with a simple classifier using the LLM.

Later this becomes part of LangGraph.

------------------------------------------------------------------------

# Phase 5 --- Source Grounding & Citation Contracts

Improve answer reliability.

## Current

Sources appear under the answer.

## Upgrade

Require the model to generate:

-   Claims tied to sources
-   Explicit citations
-   Evidence excerpts
-   Confidence signals

## Output Format Example

Answer Evidence Paper references Confidence

This significantly improves research usability.

------------------------------------------------------------------------

# Phase 6 --- Research Workflow UI

Extend the interface beyond chat.

## New Views

### Paper Inspector

Click source → view full chunk context.

### Compare View

Side‑by‑side comparison of papers.

### Literature Matrix

Rows = papers\
Columns = methods, datasets, limitations.

### Evidence Explorer

Browse chunks retrieved for a question.

------------------------------------------------------------------------

# Phase 7 --- LangGraph Workflow Engine

Introduce agentic orchestration once retrieval and metadata are strong.

## Graph Nodes

QueryClassifier RetrievalPlanner RetrieveEvidence GroupByPaper
RerankEvidence SynthesizeAnswer VerifyCitations FormatOutput

## Why LangGraph

-   explicit control over workflow
-   better debugging
-   deterministic reasoning pipeline

------------------------------------------------------------------------

# Phase 8 --- Advanced Research Capabilities

Transform the tool into a full research assistant.

## Capabilities

### Literature Review Generation

Automatically summarize a collection.

### Gap Analysis

Identify missing research areas.

### Contradiction Detection

Find papers that disagree.

### Proposal Ideation

Suggest research directions.

### Reading Order Generation

Recommend what to read first.

------------------------------------------------------------------------

# Phase 9 --- Long‑Term Extensions

Optional future capabilities.

## Knowledge Graph Layer

Extract structured relationships between:

-   methods
-   datasets
-   tasks
-   evaluation metrics

## Cross‑Collection Reasoning

Allow queries across multiple Zotero collections.

## Research Memory

Persist:

-   generated summaries
-   research notes
-   conversation insights

## Collaboration

Shared research libraries for teams.

------------------------------------------------------------------------

# Final System Vision

The end state is not just a chatbot over PDFs.

It becomes a **source‑grounded research copilot** capable of:

-   understanding individual papers
-   synthesizing across literature
-   exposing evidence transparently
-   assisting with research planning
-   accelerating literature reviews
