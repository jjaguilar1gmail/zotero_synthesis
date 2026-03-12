from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from pathlib import Path

from main import (
    adaptively_filter_candidates,
    build_dense_query_variants,
    SafeChromaVectorStore,
    build_retrieval_only_response,
    classify_query_route,
    count_unique_papers,
    dedupe_nodes_by_id,
    infer_effective_section_label,
    infer_retrieval_plan,
    is_low_value_section,
    merge_dense_candidate_batches,
    metadata_matches_plan,
    rerank_evidence_candidates,
    sanitize_utf8_text,
    score_paper_seed_relevance,
    select_evidence_chunks,
)
from ingestion.document_structure import PaperMetadata, PaperRecord


def make_candidate(*, text: str, score: float, paper_id: str, title: str, year: int, section_label: str, section_heading: str) -> NodeWithScore:
    node = TextNode(
        text=text,
        metadata={
            "paper_node_id": paper_id,
            "title": title,
            "year": year,
            "section_label": section_label,
            "section_heading": section_heading,
            "section_path": f"{section_label} > {section_heading}",
            "source_file": f"{paper_id}.pdf",
            "page_number": 3,
        },
    )
    return NodeWithScore(node=node, score=score)


def test_infer_retrieval_plan_extracts_years_sections_and_comparative_mode():
    plan = infer_retrieval_plan("Compare the methods used in papers between 2020 and 2024")

    assert plan.comparative is True
    assert plan.route_label == "comparison"
    assert plan.year_from == 2020
    assert plan.year_to == 2024
    assert "Methods" in plan.section_labels
    assert plan.dense_top_k >= 20
    assert plan.diverse_paper_goal == 3


def test_classify_query_route_distinguishes_risk_survey_and_lookup_queries():
    assert classify_query_route("What risks and limitations do these agent papers discuss?").label == "risk_analysis"
    assert classify_query_route("What broader survey themes appear across these papers?").label == "survey_synthesis"
    assert classify_query_route("Which paper introduces Toolformer?").label == "single_anchor_lookup"


def test_classify_query_route_promotes_multi_paper_tool_and_cost_queries_to_comparison():
    assert classify_query_route("How do the agent papers integrate tool use, action, or acting loops?").label == "comparison"
    assert classify_query_route("What do these papers say about cost, latency, routing, or budget tradeoffs for agentic systems?").label == "comparison"


def test_build_dense_query_variants_adds_topic_and_route_expansions_without_duplicates():
    survey_plan = infer_retrieval_plan("What broader research agenda or survey-level themes appear across these papers?")

    variants = build_dense_query_variants(survey_plan)

    assert variants[0] == survey_plan.query_text
    assert any("survey review agenda overview future directions" in variant.lower() for variant in variants)
    assert len({variant.lower() for variant in variants}) == len(variants)


def test_metadata_matches_plan_respects_year_and_section_constraints():
    plan = infer_retrieval_plan("Summarize the results since 2022")

    assert metadata_matches_plan(
        {
            "year": 2023,
            "section_label": "Results",
            "section_heading": "Main Results",
            "section_path": "Results > Main Results",
        },
        plan,
    )
    assert not metadata_matches_plan(
        {
            "year": 2021,
            "section_label": "Results",
            "section_heading": "Main Results",
            "section_path": "Results > Main Results",
        },
        plan,
    )
    assert not metadata_matches_plan(
        {
            "year": 2023,
            "section_label": "PRELIMINARIES",
            "section_heading": "Problem Formulation",
            "section_path": "PRELIMINARIES > Problem Formulation",
        },
        plan,
    )


def test_adaptive_filter_backfills_relaxed_candidates_when_comparison_is_paper_starved():
    plan = infer_retrieval_plan("Compare the reinforcement learning or policy optimization approaches used for agent training and planning.")
    candidates = [
        make_candidate(
            text="Reinforcement learning improves agent planning with reward shaping.",
            score=0.9,
            paper_id="paper-a",
            title="Paper A",
            year=2025,
            section_label="Methods",
            section_heading="Policy Optimization",
        ),
        make_candidate(
            text="Training uses reward models and policy gradients.",
            score=0.7,
            paper_id="paper-b",
            title="Paper B",
            year=2024,
            section_label="Introduction",
            section_heading="Overview",
        ),
        make_candidate(
            text="Another agent training setup compares RL trajectories.",
            score=0.6,
            paper_id="paper-c",
            title="Paper C",
            year=2023,
            section_label="Background",
            section_heading="Problem Setting",
        ),
    ]

    filtered = adaptively_filter_candidates(plan, candidates)

    assert count_unique_papers(filtered) >= 3
    assert metadata_matches_plan(filtered[0].node.metadata or {}, plan)


def test_merge_dense_candidate_batches_keeps_best_duplicate_node_score():
    duplicate_low = make_candidate(
        text="Tool use candidate.",
        score=0.2,
        paper_id="paper-a",
        title="Paper A",
        year=2025,
        section_label="Methods",
        section_heading="Tool Use",
    )
    duplicate_low.node.id_ = "dup-node"
    duplicate_high = make_candidate(
        text="Tool use candidate.",
        score=0.8,
        paper_id="paper-a",
        title="Paper A",
        year=2025,
        section_label="Methods",
        section_heading="Tool Use",
    )
    duplicate_high.node.id_ = "dup-node"
    unique = make_candidate(
        text="Unique candidate.",
        score=0.5,
        paper_id="paper-b",
        title="Paper B",
        year=2024,
        section_label="Results",
        section_heading="Results",
    )

    merged = merge_dense_candidate_batches([[duplicate_low, unique], [duplicate_high]], limit=10)

    assert len(merged) == 2
    assert float(merged[0].score or 0.0) == 0.8


def test_score_paper_seed_relevance_prefers_survey_like_titles_for_survey_queries():
    plan = infer_retrieval_plan("What broader research agenda or survey-level themes appear across the survey and agenda papers in this collection?")
    survey_record = PaperRecord(
        paper_id="paper-survey",
        file_path=Path("paper-survey.pdf"),
        metadata=PaperMetadata(
            zotero_item_key="paper-survey",
            attachment_key="att-survey",
            title="A Survey of Context Engineering for Large Language Models",
            authors=[],
            year=2025,
            journal_venue="arXiv",
            abstract="Survey of context engineering systems, open challenges, and future directions.",
            tags=["survey"],
            collection_id=3,
            parent_paper_id="paper-survey",
        ),
    )
    unrelated_record = PaperRecord(
        paper_id="paper-unrelated",
        file_path=Path("paper-unrelated.pdf"),
        metadata=PaperMetadata(
            zotero_item_key="paper-unrelated",
            attachment_key="att-unrelated",
            title="Efficient Tool Calling for Agents",
            authors=[],
            year=2025,
            journal_venue="arXiv",
            abstract="Methods for tool use and agent control.",
            tags=["agents"],
            collection_id=3,
            parent_paper_id="paper-unrelated",
        ),
    )

    assert score_paper_seed_relevance(plan.query_text, plan, survey_record) > score_paper_seed_relevance(plan.query_text, plan, unrelated_record)


def test_infer_effective_section_label_maps_evaluation_and_problem_sections():
    assert infer_effective_section_label(
        {
            "section_label": "End-to-End LLM Engine Evaluation",
            "section_heading": "Mask Generation Efficiency",
            "section_path": "Evaluation > Mask Generation Efficiency",
        }
    ) == "Results"
    assert infer_effective_section_label(
        {
            "section_label": "PRELIMINARIES",
            "section_heading": "Problem Formulation",
            "section_path": "PRELIMINARIES > Problem Formulation",
        }
    ) == "Background"


def test_is_low_value_section_flags_references_and_preliminaries():
    assert is_low_value_section({"section_label": "REFERENCES", "section_heading": "REFERENCES"})
    assert is_low_value_section({"section_label": "PRELIMINARIES", "section_heading": "PRELIMINARIES"})
    assert not is_low_value_section({"section_label": "Results", "section_heading": "Benchmark Results"})


def test_reranking_and_selection_preserve_cross_paper_evidence():
    plan = infer_retrieval_plan("Compare results for gait analysis")
    candidates = [
        make_candidate(
            text="This results section reports strong gait analysis accuracy improvements.",
            score=0.82,
            paper_id="paper-a",
            title="Paper A",
            year=2024,
            section_label="Results",
            section_heading="Results",
        ),
        make_candidate(
            text="These findings compare gait analysis benchmarks across datasets.",
            score=0.79,
            paper_id="paper-b",
            title="Paper B",
            year=2025,
            section_label="Results",
            section_heading="Benchmark Results",
        ),
        make_candidate(
            text="Background context for gait analysis tasks.",
            score=0.76,
            paper_id="paper-a",
            title="Paper A",
            year=2024,
            section_label="Background",
            section_heading="Background",
        ),
    ]

    reranked = rerank_evidence_candidates("Compare results for gait analysis", plan, candidates)
    selected = select_evidence_chunks(plan, reranked)

    assert len(selected) >= 2
    assert {chunk.paper_id for chunk in selected} >= {"paper-a", "paper-b"}
    assert selected[0].source_id == "S1"


class FakeChromaCollection:
    def query(self, **kwargs):
        return {
            "ids": [["node-1"]],
            "documents": [[None]],
            "metadatas": [[{"title": "Paper A", "section_label": "Results"}]],
            "distances": [[0.2]],
        }

    def get(self, **kwargs):
        return {
            "ids": ["node-1"],
            "documents": [None],
            "metadatas": [{"title": "Paper A", "section_label": "Results"}],
        }


def test_safe_chroma_vector_store_handles_missing_document_text():
    store = SafeChromaVectorStore(chroma_collection=FakeChromaCollection())

    query_result = store.query(VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=1))
    get_result = store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=1))

    assert query_result.nodes[0].text == ""
    assert get_result.nodes[0].text == ""


def test_sanitize_utf8_text_removes_surrogate_codepoints():
    broken = "prefix\ud83dsuffix"

    assert sanitize_utf8_text(broken) == "prefixsuffix"


def test_dedupe_nodes_by_id_skips_duplicate_node_ids():
    node_a = TextNode(id_="dup", text="first")
    node_b = TextNode(id_="dup", text="second")
    node_c = TextNode(id_="unique", text="third")

    deduped, skipped = dedupe_nodes_by_id([node_a, node_b, node_c])

    assert [node.node_id for node in deduped] == ["dup", "unique"]
    assert skipped == 1


def test_build_retrieval_only_response_lists_top_evidence():
    plan = infer_retrieval_plan("Compare results for gait analysis")
    candidates = [
        make_candidate(
            text="This results section reports strong gait analysis accuracy improvements.",
            score=0.82,
            paper_id="paper-a",
            title="Paper A",
            year=2024,
            section_label="Results",
            section_heading="Results",
        ),
        make_candidate(
            text="These findings compare gait analysis benchmarks across datasets.",
            score=0.79,
            paper_id="paper-b",
            title="Paper B",
            year=2025,
            section_label="Results",
            section_heading="Benchmark Results",
        ),
    ]
    reranked = rerank_evidence_candidates("Compare results for gait analysis", plan, candidates)
    selected = select_evidence_chunks(plan, reranked)

    from main import RetrievalResult

    response = build_retrieval_only_response(
        RetrievalResult(
            plan=plan,
            dense_candidates=candidates,
            filtered_candidates=candidates,
            reranked_candidates=reranked,
            selected_chunks=selected,
        )
    )

    assert "Top evidence" in response
    assert "[S1]" in response
