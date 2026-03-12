from tools.evaluate_retrieval import evaluate_expectations, evaluate_route_expectation


def test_evaluate_expectations_passes_when_sources_meet_constraints():
    sources = [
        {
            "title": "Paper A",
            "section_label": "Methods",
            "section_heading": "Dataset and Methods",
            "snippet": "This benchmark dataset improves retrieval quality.",
        },
        {
            "title": "Paper B",
            "section_label": "Results",
            "section_heading": "Results",
            "snippet": "Comparison across datasets shows stronger accuracy.",
        },
    ]
    retrieval_summary = {"papers_considered": 2, "evidence_chunks": 4}

    result = evaluate_expectations(
        {
            "min_papers": 2,
            "min_evidence_chunks": 3,
            "section_labels_any": ["Methods", "Results"],
            "keywords_any": ["dataset", "accuracy"],
        },
        sources,
        retrieval_summary,
    )

    assert result["status"] == "pass"
    assert all(check["passed"] for check in result["checks"])


def test_evaluate_expectations_fails_when_sources_do_not_meet_constraints():
    sources = [
        {
            "title": "Paper A",
            "section_label": "Introduction",
            "section_heading": "Introduction",
            "snippet": "General background only.",
        }
    ]
    retrieval_summary = {"papers_considered": 1, "evidence_chunks": 1}

    result = evaluate_expectations(
        {
            "min_papers": 2,
            "min_evidence_chunks": 2,
            "section_labels_any": ["Methods"],
            "keywords_any": ["benchmark"],
        },
        sources,
        retrieval_summary,
    )

    assert result["status"] == "fail"
    assert any(not check["passed"] for check in result["checks"])


def test_build_retrieval_summary_carries_route_metadata():
    from main import RetrievalPlan, RetrievalResult
    from tools.evaluate_retrieval import build_retrieval_summary

    summary = build_retrieval_summary(
        RetrievalResult(
            plan=RetrievalPlan(
                query_text="Compare the methods used across papers",
                route_label="comparison",
                route_backend="deterministic",
                route_features={"has_comparison_markers": True},
            ),
            dense_candidates=[],
            filtered_candidates=[],
            reranked_candidates=[],
            selected_chunks=[],
        )
    )

    assert summary["route"]["label"] == "comparison"
    assert summary["route"]["backend"] == "deterministic"


def test_evaluate_route_expectation_reports_match_and_mismatch():
    retrieval_summary = {"route": {"label": "comparison"}}

    assert evaluate_route_expectation("comparison", retrieval_summary)["status"] == "pass"
    assert evaluate_route_expectation("survey_synthesis", retrieval_summary)["status"] == "fail"
