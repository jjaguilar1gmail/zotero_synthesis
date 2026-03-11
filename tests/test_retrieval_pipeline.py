from llama_index.core.schema import NodeWithScore, TextNode

from main import (
    infer_retrieval_plan,
    metadata_matches_plan,
    rerank_evidence_candidates,
    select_evidence_chunks,
)


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
    assert plan.year_from == 2020
    assert plan.year_to == 2024
    assert "Methods" in plan.section_labels
    assert plan.dense_top_k == 20


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
            "section_label": "Introduction",
            "section_heading": "Intro",
            "section_path": "Introduction",
        },
        plan,
    )


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
