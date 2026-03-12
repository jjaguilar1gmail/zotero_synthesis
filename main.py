import os
import sqlite3
import json
import logging
import math
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
from llama_index.vector_stores.chroma.base import _to_chroma_filter, legacy_metadata_dict_to_node, metadata_dict_to_node
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PDFReader
from openai import AuthenticationError
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

SECTION_FAMILY_HINTS = {
    "Abstract": {"abstract"},
    "Introduction": {"introduction", "motivation", "overview"},
    "Background": {"background", "preliminaries", "preliminary", "problem formulation", "problem setting"},
    "Related Work": {"related work", "related works", "prior work", "literature review", "survey of"},
    "Methods": {
        "method",
        "methods",
        "methodology",
        "approach",
        "framework",
        "algorithm",
        "algorithm design",
        "implementation",
        "training",
        "workflow",
        "planning",
        "tool integration",
        "system",
    },
    "Results": {
        "results",
        "result",
        "evaluation",
        "evaluations",
        "experiment",
        "experiments",
        "benchmark",
        "benchmarks",
        "ablation",
        "ablation study",
        "analysis",
        "case study",
        "baselines",
        "comparison",
        "quantitative",
        "performance",
    },
    "Discussion": {
        "discussion",
        "limitations",
        "limitation",
        "risk",
        "risks",
        "failure",
        "failures",
        "challenge",
        "challenges",
        "future direction",
        "future directions",
        "future work",
    },
    "Conclusion": {"conclusion", "conclusions", "concluding", "summary"},
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

QUERY_TOPIC_HINTS = {
    "tool_use": {
        "triggers": {"tool", "tools", "acting", "action", "actions", "loop", "loops", "react", "toolformer"},
        "section_labels": ["Methods", "Results", "Introduction"],
        "keywords": {"tool", "tools", "acting", "action", "react", "toolformer", "rs-agent", "workflow"},
    },
    "geospatial": {
        "triggers": {"remote sensing", "geospatial", "geolocalization", "map", "geoagent", "gis"},
        "section_labels": ["Methods", "Results", "Introduction"],
        "keywords": {"remote sensing", "geospatial", "geolocalization", "map", "geoagent", "tree-gpt", "change-agent", "rs-agent"},
    },
    "cost": {
        "triggers": {"cost", "latency", "budget", "routing", "throughput"},
        "section_labels": ["Results", "Discussion", "Conclusion"],
        "keywords": {"cost", "latency", "budget", "routing", "throughput", "efficiency"},
    },
    "risk": {
        "triggers": {"risk", "risks", "limitation", "limitations", "failure", "failures", "safety"},
        "section_labels": ["Discussion", "Conclusion", "Results"],
        "keywords": {"risk", "risks", "limitation", "limitations", "failure", "failures", "safety", "challenge"},
    },
    "reinforcement_learning": {
        "triggers": {"reinforcement", "policy", "optimization", "planning", "reward", "training"},
        "section_labels": ["Methods", "Results"],
        "keywords": {"reinforcement", "policy", "optimization", "planning", "reward", "training", "rl"},
    },
    "structured_generation": {
        "triggers": {"structured generation", "constrained decoding", "grammar", "xgrammar"},
        "section_labels": ["Methods", "Results"],
        "keywords": {"structured generation", "constrained decoding", "grammar", "xgrammar", "tool calling"},
    },
    "survey": {
        "triggers": {"survey", "agenda", "themes", "research agenda", "broader"},
        "section_labels": ["Introduction", "Discussion", "Conclusion"],
        "keywords": {"survey", "agenda", "context engineering", "autonomous gis", "reasoning era", "future directions"},
    },
}

ROUTE_LABELS = {
    "comparison",
    "survey_synthesis",
    "risk_analysis",
    "single_anchor_lookup",
    "topic_synthesis",
}

SURVEY_ROUTE_MARKERS = {"survey", "agenda", "themes", "research agenda", "broader", "overview"}
RISK_ROUTE_MARKERS = {"risk", "risks", "limitation", "limitations", "failure", "failures", "safety", "challenge", "challenges"}
LOOKUP_ROUTE_MARKERS = {"what is", "define", "who is", "when did", "which paper", "find the paper", "where is"}
MULTI_PAPER_MARKERS = {"these papers", "the papers", "agent papers", "papers in this collection", "across the papers"}
COMPARISON_TOPIC_PROFILES = {"tool_use", "geospatial", "cost", "reinforcement_learning", "structured_generation"}


@dataclass
class QueryRoute:
    label: str
    backend: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    query_text: str
    route_label: str = "topic_synthesis"
    route_backend: str = "deterministic"
    route_features: dict[str, Any] = field(default_factory=dict)
    section_labels: list[str] = field(default_factory=list)
    topic_keywords: list[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    comparative: bool = False
    dense_top_k: int = 14
    target_papers: int = 3
    per_paper_limit: int = 2
    max_evidence_chunks: int = 6
    diverse_paper_goal: int = 1


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
    @staticmethod
    def _coerce_result_text(text: Any, node_id: str) -> str:
        if text is None:
            logger.warning("Chroma returned empty document text for node %s; substituting empty string", node_id)
            return ""
        return str(text)

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

    def _query(self, query_embeddings: list[float], n_results: int, where: dict | None, **kwargs: Any) -> VectorStoreQueryResult:
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

        nodes = []
        similarities = []
        ids = []
        for node_id, text, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            safe_text = self._coerce_result_text(text, node_id)
            metadata = metadata or {}
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(safe_text)
            except Exception:
                metadata, node_info, relationships = legacy_metadata_dict_to_node(metadata)
                node = TextNode(
                    text=safe_text,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)
            similarities.append(math.exp(-distance))
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _get(self, limit: Optional[int], where: dict | None, **kwargs: Any) -> VectorStoreQueryResult:
        results = self._collection.get(limit=limit, where=where, **kwargs)

        nodes = []
        ids = []
        raw_ids = results.get("ids") or []
        raw_documents = results.get("documents") or []
        raw_metadatas = results.get("metadatas") or []

        for node_id, text, metadata in zip(raw_ids, raw_documents, raw_metadatas):
            safe_text = self._coerce_result_text(text, node_id)
            metadata = metadata or {}
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(safe_text)
            except Exception:
                metadata, node_info, relationships = legacy_metadata_dict_to_node(metadata)
                node = TextNode(
                    text=safe_text,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, ids=ids)


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sanitize_utf8_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def sanitize_metadata_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_utf8_text(value)
        else:
            sanitized[key] = value
    return sanitized


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

    for hint_profile in QUERY_TOPIC_HINTS.values():
        if any(trigger in lowered for trigger in hint_profile["triggers"]):
            for label in hint_profile["section_labels"]:
                if label not in section_labels:
                    section_labels.append(label)
    return section_labels


def infer_topic_keywords(query_text: str) -> list[str]:
    lowered = query_text.lower()
    keywords: list[str] = []
    for hint_profile in QUERY_TOPIC_HINTS.values():
        if any(trigger in lowered for trigger in hint_profile["triggers"]):
            for keyword in hint_profile["keywords"]:
                if keyword not in keywords:
                    keywords.append(keyword)
    return keywords


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


def classify_query_route(query_text: str, backend: str = "deterministic") -> QueryRoute:
    lowered = query_text.lower()
    topic_profiles = [
        topic_name
        for topic_name, hint_profile in QUERY_TOPIC_HINTS.items()
        if any(trigger in lowered for trigger in hint_profile["triggers"])
    ]
    route_features = {
        "has_comparison_markers": any(marker in lowered for marker in COMPARATIVE_QUERY_MARKERS),
        "has_survey_markers": any(marker in lowered for marker in SURVEY_ROUTE_MARKERS),
        "has_risk_markers": any(marker in lowered for marker in RISK_ROUTE_MARKERS),
        "has_lookup_markers": any(marker in lowered for marker in LOOKUP_ROUTE_MARKERS),
        "has_multi_paper_markers": any(marker in lowered for marker in MULTI_PAPER_MARKERS) or "papers" in lowered,
        "has_how_do_pattern": lowered.startswith("how do ") or lowered.startswith("how are "),
        "has_what_do_these_papers_say": "what do these papers say" in lowered,
        "has_year_filters": bool(re.search(r"(?:19|20)\d{2}", lowered)),
        "token_count": len(tokenize_query_terms(query_text)),
        "topic_profiles": topic_profiles,
    }

    if route_features["has_risk_markers"]:
        label = "risk_analysis"
    elif route_features["has_survey_markers"]:
        label = "survey_synthesis"
    elif route_features["has_lookup_markers"] and not route_features["has_comparison_markers"]:
        label = "single_anchor_lookup"
    elif route_features["has_comparison_markers"]:
        label = "comparison"
    elif route_features["has_multi_paper_markers"] and (
        route_features["has_how_do_pattern"]
        or route_features["has_what_do_these_papers_say"]
        or any(profile in COMPARISON_TOPIC_PROFILES for profile in topic_profiles)
    ):
        label = "comparison"
    else:
        label = "topic_synthesis"

    return QueryRoute(label=label, backend=backend, features=route_features)


def infer_retrieval_plan(query_text: str) -> RetrievalPlan:
    route = classify_query_route(query_text)
    comparative = route.label in {"comparison", "risk_analysis", "survey_synthesis"}
    section_labels = infer_section_filters(query_text)
    topic_keywords = infer_topic_keywords(query_text)
    year_from, year_to = infer_year_filters(query_text)

    dense_top_k = 18
    target_papers = 4
    per_paper_limit = 2
    max_evidence_chunks = 6
    diverse_paper_goal = 1

    if route.label == "comparison":
        dense_top_k = 24
        target_papers = 5
        max_evidence_chunks = 10
        diverse_paper_goal = 3
    elif route.label == "survey_synthesis":
        dense_top_k = 22
        target_papers = 5
        max_evidence_chunks = 8
        diverse_paper_goal = 3
    elif route.label == "risk_analysis":
        dense_top_k = 20
        target_papers = 4
        max_evidence_chunks = 8
        diverse_paper_goal = 2
    elif route.label == "single_anchor_lookup":
        dense_top_k = 12
        target_papers = 2
        per_paper_limit = 3
        max_evidence_chunks = 5

    return RetrievalPlan(
        query_text=query_text,
        route_label=route.label,
        route_backend=route.backend,
        route_features=route.features,
        section_labels=section_labels,
        topic_keywords=topic_keywords,
        year_from=year_from,
        year_to=year_to,
        comparative=comparative,
        dense_top_k=dense_top_k,
        target_papers=target_papers,
        per_paper_limit=per_paper_limit,
        max_evidence_chunks=max_evidence_chunks,
        diverse_paper_goal=diverse_paper_goal,
    )


def infer_effective_section_label(metadata: dict[str, Any]) -> Optional[str]:
    text = " ".join(
        normalize_text(metadata.get(field))
        for field in ("section_label", "section_heading", "section_path")
    ).lower()
    if not text:
        return None

    for label, hints in SECTION_FAMILY_HINTS.items():
        if any(hint in text for hint in hints):
            return label
    if any(hint in text for hint in END_MATTER_LABELS):
        return "EndMatter"
    return None


def is_low_value_section(metadata: dict[str, Any]) -> bool:
    section_label = normalize_text(metadata.get("section_label"))
    section_heading = normalize_text(metadata.get("section_heading"))
    effective_label = infer_effective_section_label(metadata)
    combined = f"{section_label} {section_heading}".lower()

    if effective_label == "EndMatter":
        return True
    if effective_label == "Background" and "prelimin" in combined:
        return True
    if any(token in combined for token in ("references", "appendix", "acknowledg", "author contributions", "data availability")):
        return True
    if len(section_label.split()) >= 6 and effective_label is None:
        return True
    return False


def metadata_matches_plan(metadata: dict, plan: RetrievalPlan) -> bool:
    return _metadata_matches_plan(metadata, plan, enforce_section_filters=True)


def _metadata_matches_plan(metadata: dict, plan: RetrievalPlan, enforce_section_filters: bool) -> bool:
    year = metadata.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    if year == -1:
        year = None

    if plan.year_from is not None and (year is None or year < plan.year_from):
        return False
    if plan.year_to is not None and (year is None or year > plan.year_to):
        return False

    if enforce_section_filters and plan.section_labels:
        section_label = normalize_text(metadata.get("section_label")).lower()
        section_heading = normalize_text(metadata.get("section_heading")).lower()
        section_path = normalize_text(metadata.get("section_path")).lower()
        effective_label = infer_effective_section_label(metadata)
        if not any(
            label.lower() == section_label
            or label.lower() in section_heading
            or label.lower() in section_path
            or label == effective_label
            for label in plan.section_labels
        ):
            return False

    return True


def count_unique_papers(candidates: list[NodeWithScore]) -> int:
    paper_ids = {
        normalize_text(
            (candidate.node.metadata or {}).get("paper_node_id")
            or (candidate.node.metadata or {}).get("parent_paper_id")
            or (candidate.node.metadata or {}).get("title")
            or candidate.node.node_id
        )
        for candidate in candidates
    }
    return len({paper_id for paper_id in paper_ids if paper_id})


def build_dense_query_variants(plan: RetrievalPlan) -> list[str]:
    variants = [plan.query_text.strip()]

    if plan.topic_keywords:
        keyword_query = " ".join(plan.topic_keywords[:6]).strip()
        if keyword_query:
            variants.append(keyword_query)
            variants.append(f"{plan.query_text.strip()} {keyword_query}".strip())

    if plan.route_label == "survey_synthesis":
        survey_query = "survey review agenda overview future directions"
        variants.append(survey_query)
        if plan.topic_keywords:
            variants.append(f"{survey_query} {' '.join(plan.topic_keywords[:4])}".strip())
    elif plan.route_label == "risk_analysis":
        risk_query = "risks limitations failures safety challenges"
        variants.append(risk_query)
    elif plan.route_label == "comparison" and plan.topic_keywords:
        variants.append(f"compare approaches {' '.join(plan.topic_keywords[:5])}".strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = normalize_text(variant)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def merge_dense_candidate_batches(candidate_batches: list[list[NodeWithScore]], limit: int) -> list[NodeWithScore]:
    best_by_node_id: dict[str, NodeWithScore] = {}

    for batch in candidate_batches:
        for candidate in batch:
            node_id = candidate.node.node_id
            existing = best_by_node_id.get(node_id)
            existing_score = float(existing.score or 0.0) if existing is not None else float("-inf")
            candidate_score = float(candidate.score or 0.0)
            if existing is None or candidate_score > existing_score:
                best_by_node_id[node_id] = candidate

    merged = sorted(best_by_node_id.values(), key=lambda item: float(item.score or 0.0), reverse=True)
    return merged[:limit]


def retrieve_dense_candidates(index: VectorStoreIndex, plan: RetrievalPlan) -> list[NodeWithScore]:
    query_variants = build_dense_query_variants(plan)
    retriever = index.as_retriever(similarity_top_k=plan.dense_top_k)
    candidate_batches = [list(retriever.retrieve(query_text)) for query_text in query_variants]
    limit = max(plan.dense_top_k * 2, plan.dense_top_k + 8)
    return merge_dense_candidate_batches(candidate_batches, limit=limit)


def score_paper_seed_relevance(query_text: str, plan: RetrievalPlan, paper_record: PaperRecord) -> float:
    title = normalize_text(paper_record.metadata.title)
    abstract = normalize_text(paper_record.metadata.abstract)
    venue = normalize_text(paper_record.metadata.journal_venue)
    tags = " ".join(paper_record.metadata.tags)
    query_terms = tokenize_query_terms(query_text)
    query_terms.update(tokenize_query_terms(" ".join(plan.topic_keywords)))
    combined_text = f"{title} {abstract} {venue} {tags}"

    score = lexical_overlap_score(query_terms, combined_text)
    score += lexical_overlap_score(query_terms, title) * 0.7
    score += lexical_overlap_score(set(plan.topic_keywords), combined_text) * 0.3 if plan.topic_keywords else 0.0

    lowered_title = title.lower()
    if plan.route_label == "survey_synthesis" and any(marker in lowered_title for marker in ("survey", "agenda", "review", "overview")):
        score += 0.35
    if plan.route_label == "risk_analysis" and any(marker in lowered_title for marker in ("risk", "safety", "failure", "limitations")):
        score += 0.2
    return score


def seed_candidates_from_collection_metadata(collection_id: int, plan: RetrievalPlan, dense_candidates: list[NodeWithScore]) -> list[NodeWithScore]:
    if not plan.comparative or count_unique_papers(dense_candidates) >= plan.diverse_paper_goal:
        return []

    existing_paper_ids = {
        normalize_text(
            (candidate.node.metadata or {}).get("paper_node_id")
            or (candidate.node.metadata or {}).get("parent_paper_id")
        )
        for candidate in dense_candidates
    }
    paper_records = get_paper_records_for_collection(collection_id)
    ranked_records = sorted(
        (
            (score_paper_seed_relevance(plan.query_text, plan, paper_record), paper_record)
            for paper_record in paper_records
            if paper_record.paper_id not in existing_paper_ids
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    top_records = [item for item in ranked_records if item[0] > 0][: max(plan.diverse_paper_goal * 2, 4)]
    if not top_records:
        return []

    chroma_collection = chroma_client.get_collection(collection_chroma_name(collection_id))
    vector_store = SafeChromaVectorStore(chroma_collection=chroma_collection)
    seed_limit = 4 if plan.route_label == "survey_synthesis" else 3
    seeded_candidates: list[NodeWithScore] = []

    for paper_score, paper_record in top_records:
        result = vector_store._get(limit=seed_limit, where={"paper_node_id": paper_record.paper_id})
        for node in result.nodes:
            seeded_candidates.append(NodeWithScore(node=node, score=paper_score))

    return seeded_candidates


def adaptively_filter_candidates(plan: RetrievalPlan, dense_candidates: list[NodeWithScore]) -> list[NodeWithScore]:
    strict_matches = [candidate for candidate in dense_candidates if metadata_matches_plan(candidate.node.metadata or {}, plan)]
    if not strict_matches:
        return dense_candidates

    if not plan.comparative or count_unique_papers(strict_matches) >= plan.diverse_paper_goal:
        return strict_matches

    relaxed_matches = [
        candidate
        for candidate in dense_candidates
        if _metadata_matches_plan(candidate.node.metadata or {}, plan, enforce_section_filters=False)
    ]
    return merge_dense_candidate_batches([strict_matches, relaxed_matches], limit=max(plan.dense_top_k, len(strict_matches)))


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


def normalize_chunk_fingerprint_text(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = re.sub(r"\bfig\.?(?=\s*\d)", "figure", normalized)
    normalized = re.sub(r"\beq\.?(?=\s*\d)", "equation", normalized)
    normalized = re.sub(r"\b(?:figure|table|fig|eq|equation)\s+\d+[a-z]?\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_chunk_fingerprint(chunk: EvidenceChunk) -> str:
    metadata = chunk.node.node.metadata or {}
    heading = normalize_text(chunk.section_heading or chunk.section_label or metadata.get("section_heading") or "")
    text = normalize_chunk_fingerprint_text(chunk.node.text)
    title = normalize_text(chunk.title)
    title_terms = title.lower().split()[:6]
    heading_terms = heading.lower().split()[:8]
    content_terms = text.split()[:40]
    return " | ".join(
        [
            " ".join(title_terms),
            " ".join(heading_terms),
            " ".join(content_terms),
        ]
    ).strip()


def is_near_duplicate_chunk(candidate: EvidenceChunk, selected: list[EvidenceChunk]) -> bool:
    candidate_fp = build_chunk_fingerprint(candidate)
    if not candidate_fp:
        return False

    candidate_terms = set(candidate_fp.split())
    if not candidate_terms:
        return False

    for existing in selected:
        if existing.paper_id != candidate.paper_id:
            continue
        existing_fp = build_chunk_fingerprint(existing)
        if not existing_fp:
            continue
        if candidate_fp == existing_fp:
            return True

        existing_terms = set(existing_fp.split())
        overlap = len(candidate_terms & existing_terms)
        union = len(candidate_terms | existing_terms)
        if union and (overlap / union) >= 0.82:
            return True
    return False


def rerank_evidence_candidates(query_text: str, plan: RetrievalPlan, candidates: list[NodeWithScore]) -> list[EvidenceChunk]:
    query_terms = tokenize_query_terms(query_text)
    query_terms.update(tokenize_query_terms(" ".join(plan.topic_keywords)))
    survey_terms = {"survey", "agenda", "review", "overview", "future", "directions"}
    reranked: list[EvidenceChunk] = []

    for candidate in candidates:
        metadata = candidate.node.metadata or {}
        title = normalize_text(metadata.get("title"))
        section_label = normalize_text(metadata.get("section_label")) or None
        section_heading = normalize_text(metadata.get("section_heading")) or None
        section_path = normalize_text(metadata.get("section_path"))
        effective_section_label = infer_effective_section_label(metadata)
        paper_id = normalize_text(metadata.get("paper_node_id") or metadata.get("parent_paper_id") or title or candidate.node.node_id)
        dense_score = float(candidate.score or 0.0)

        overlap = lexical_overlap_score(query_terms, f"{title} {section_heading or ''} {section_path} {candidate.text[:800]}")
        title_overlap = lexical_overlap_score(query_terms, title)
        section_overlap = lexical_overlap_score(query_terms, f"{section_label or ''} {section_heading or ''}")
        topic_overlap = lexical_overlap_score(set(plan.topic_keywords), f"{title} {section_heading or ''} {candidate.text[:600]}") if plan.topic_keywords else 0.0

        score = dense_score
        score += overlap * 0.45
        score += title_overlap * 0.25
        score += section_overlap * 0.15
        score += topic_overlap * 0.2

        if plan.section_labels and effective_section_label in plan.section_labels:
            score += 0.25
        elif plan.section_labels and effective_section_label is not None:
            score -= 0.08

        if effective_section_label in {"Results", "Methods", "Discussion", "Conclusion"}:
            score += 0.08
        if effective_section_label == "Introduction" and plan.comparative:
            score -= 0.03
        if is_low_value_section(metadata):
            score -= 0.45

        if title_overlap > 0.0 and plan.topic_keywords:
            score += 0.08

        if plan.route_label == "survey_synthesis":
            survey_overlap = lexical_overlap_score(survey_terms, f"{title} {section_heading or ''}")
            score += survey_overlap * 0.35
            if effective_section_label in {"Introduction", "Discussion", "Conclusion"}:
                score += 0.12
            if any(marker in f"{title} {section_heading or ''}".lower() for marker in ("survey", "agenda", "review", "future direction", "future work", "overview")):
                score += 0.18

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
    selected_papers = ranked_papers[: plan.target_papers]

    for _, paper_candidates in selected_papers:
        if paper_candidates:
            if not is_near_duplicate_chunk(paper_candidates[0], selected):
                selected.append(paper_candidates[0])

    round_index = 1
    while len(selected) < plan.max_evidence_chunks:
        added = False
        for _, paper_candidates in selected_papers:
            if round_index < len(paper_candidates) and round_index < plan.per_paper_limit + 1:
                candidate = paper_candidates[round_index]
                if is_near_duplicate_chunk(candidate, selected):
                    continue
                selected.append(candidate)
                added = True
                if len(selected) >= plan.max_evidence_chunks:
                    break
        if not added:
            break
        round_index += 1

    selected.sort(key=lambda item: item.rerank_score, reverse=True)
    selected = selected[: plan.max_evidence_chunks]

    for index, candidate in enumerate(selected, start=1):
        candidate.source_id = f"S{index}"

    return selected


def run_retrieval_pipeline(index: VectorStoreIndex, query_text: str, collection_id: Optional[int] = None) -> RetrievalResult:
    plan = infer_retrieval_plan(query_text)
    dense_candidates = retrieve_dense_candidates(index, plan)
    if collection_id is not None:
        seeded_candidates = seed_candidates_from_collection_metadata(collection_id, plan, dense_candidates)
        if seeded_candidates:
            dense_candidates = merge_dense_candidate_batches(
                [dense_candidates, seeded_candidates],
                limit=max(len(dense_candidates) + len(seeded_candidates), plan.dense_top_k * 2),
            )
    filtered_candidates = adaptively_filter_candidates(plan, dense_candidates)

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
    if retrieval.plan.topic_keywords:
        plan_notes.append(f"topic hints={', '.join(retrieval.plan.topic_keywords[:8])}")
    if retrieval.plan.route_label:
        plan_notes.append(f"route={retrieval.plan.route_label}")
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


def build_retrieval_only_response(retrieval: RetrievalResult) -> str:
    if not retrieval.selected_chunks:
        return "Retrieval completed, but no supporting evidence chunks were selected."

    lines = ["Retrieval completed, but answer synthesis is unavailable. Top evidence:"]
    for chunk in retrieval.selected_chunks[:4]:
        heading = chunk.section_heading or chunk.section_label or "Unlabeled section"
        excerpt = normalize_text(chunk.node.text)[:220]
        lines.append(f"- [{chunk.source_id}] {chunk.title} | {heading} | {excerpt}")
    return "\n".join(lines)


def dedupe_paper_records(records: list[PaperRecord]) -> list[PaperRecord]:
    seen: set[tuple[str, str]] = set()
    deduped: list[PaperRecord] = []
    for record in records:
        key = (record.metadata.attachment_key, str(record.file_path).lower())
        if key in seen:
            logger.info("Skipping duplicate paper record for attachment %s at %s", record.metadata.attachment_key, record.file_path)
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def dedupe_nodes_by_id(nodes: list[TextNode]) -> tuple[list[TextNode], int]:
    deduped: list[TextNode] = []
    seen_ids: set[str] = set()
    skipped = 0
    for node in nodes:
        if node.node_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(node.node_id)
        deduped.append(node)
    return deduped, skipped


def format_sources_from_evidence(chunks: list[EvidenceChunk]) -> list[dict]:
    sources = []
    for chunk in chunks:
        metadata = chunk.node.node.metadata or {}
        page_number = metadata.get("page_number")
        effective_section_label = infer_effective_section_label(metadata)
        sources.append(
            {
                "source_id": chunk.source_id,
                "file": metadata.get("source_file", "Unknown"),
                "title": metadata.get("title"),
                "year": metadata.get("year") if metadata.get("year") != -1 else None,
                "page_number": page_number if page_number != -1 else None,
                "section_label": effective_section_label or metadata.get("section_label") or None,
                "raw_section_label": metadata.get("section_label") or None,
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
                    text=sanitize_utf8_text(chunk.text),
                    metadata=sanitize_metadata_dict(metadata),
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
    return dedupe_paper_records(records)


def build_chunk_metadata(
    paper_record: PaperRecord,
    collection_name: str,
    page_number: Optional[int],
) -> dict:
    return sanitize_metadata_dict({
        "source_file": paper_record.file_path.name,
        "collection": collection_name,
        "zotero_item_key": paper_record.metadata.zotero_item_key,
        "attachment_key": paper_record.metadata.attachment_key,
        "title": sanitize_utf8_text(paper_record.metadata.title),
        "authors": json.dumps(paper_record.metadata.authors, ensure_ascii=True),
        "year": paper_record.metadata.year if paper_record.metadata.year is not None else -1,
        "journal_venue": sanitize_utf8_text(paper_record.metadata.journal_venue),
        "abstract": sanitize_utf8_text(paper_record.metadata.abstract),
        "tags": json.dumps(paper_record.metadata.tags, ensure_ascii=True),
        "collection_id": paper_record.metadata.collection_id,
        "page_number": page_number if page_number is not None else -1,
        "section_label": paper_record.metadata.section_label,
        "parent_paper_id": paper_record.metadata.parent_paper_id,
        "paper_node_id": paper_record.paper_id,
    })

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
                page_entries.append((page_number, sanitize_utf8_text(doc.text)))
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

    deduped_nodes, duplicate_node_count = dedupe_nodes_by_id(all_nodes)

    if deduped_nodes:
        VectorStoreIndex(
            nodes=deduped_nodes,
            storage_context=storage_context,
            show_progress=True,
        )

    return {
        "indexed": len(loaded),
        "papers": len(paper_records),
        "chunks": len(deduped_nodes),
        "duplicate_nodes_skipped": duplicate_node_count,
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
        retrieval = run_retrieval_pipeline(index, req.message, collection_id=req.collection_id)
        llm_warning = None
        try:
            response_text = synthesize_answer(req.message, history_text, retrieval)
        except AuthenticationError:
            llm_warning = "LLM synthesis unavailable because the configured OpenRouter credentials were rejected. Returning retrieval evidence only."
            response_text = build_retrieval_only_response(retrieval)
        sources = format_sources_from_evidence(retrieval.selected_chunks)
        return {
            "response": response_text,
            "sources": sources,
            "warning": llm_warning,
            "retrieval": {
                "route": {
                    "label": retrieval.plan.route_label,
                    "backend": retrieval.plan.route_backend,
                    "features": retrieval.plan.route_features,
                },
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
