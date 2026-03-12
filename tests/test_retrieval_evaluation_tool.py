from tools.evaluate_retrieval import (
    build_markdown_report,
    evaluate_expectations,
    evaluate_grounding_contract,
    evaluate_route_expectation,
    grounded_answer_uses_deterministic_fallback,
)


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


def test_evaluate_expectations_supports_max_papers_for_single_paper_routes():
    sources = [
        {
            "title": "Paper A",
            "section_label": "Methods",
            "section_heading": "Methods",
            "snippet": "Core algorithm description.",
        }
    ]
    retrieval_summary = {"papers_considered": 1, "evidence_chunks": 3}

    result = evaluate_expectations(
        {
            "min_papers": 1,
            "max_papers": 1,
            "min_evidence_chunks": 2,
            "keywords_any": ["algorithm"],
        },
        sources,
        retrieval_summary,
    )

    assert result["status"] == "pass"
    assert any(check["name"] == "max_papers" and check["passed"] for check in result["checks"])


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


def test_evaluate_grounding_contract_checks_claim_citations():
    grounding = {
        "answer_summary": "Summary",
        "claims": [
            {"claim_text": "This benchmark dataset improves retrieval quality.", "source_ids": ["S1"], "confidence": "medium", "note": None}
        ],
    }
    retrieval_summary = {
        "sources": [
            {"source_id": "S1", "title": "Paper A", "section_heading": "Dataset and Methods", "snippet": "This benchmark dataset improves retrieval quality."},
            {"source_id": "S2", "title": "Paper B", "section_heading": "Results", "snippet": "Comparison across datasets shows stronger accuracy."},
        ]
    }

    result = evaluate_grounding_contract(grounding, retrieval_summary)

    assert result["status"] == "pass"
    assert all(check["passed"] for check in result["checks"])


def test_evaluate_grounding_contract_fails_on_unsupported_overconfident_claims():
    grounding = {
        "answer_summary": "Summary",
        "claims": [
            {"claim_text": "This paper studies marine biology taxonomies.", "source_ids": ["S1"], "confidence": "high", "note": None}
        ],
    }
    retrieval_summary = {
        "sources": [
            {"source_id": "S1", "title": "Paper A", "section_heading": "Results", "snippet": "The algorithm uses reward shaping for planning."},
        ]
    }

    result = evaluate_grounding_contract(grounding, retrieval_summary)

    assert result["status"] == "fail"
    assert any(check["name"] == "claims_have_support_overlap" and not check["passed"] for check in result["checks"])


def test_evaluate_grounding_contract_prefers_support_text_over_display_snippet():
    grounding = {
        "answer_summary": "Summary",
        "claims": [
            {
                "claim_text": "Adaptive model routing reduces computational and memory demands for agentic systems.",
                "source_ids": ["S1"],
                "confidence": "medium",
                "note": None,
            }
        ],
    }
    retrieval_summary = {
        "sources": [
            {
                "source_id": "S1",
                "title": "Paper A",
                "section_heading": "Discussion",
                "snippet": "Short display snippet without the routing language.",
                "support_text": "Paper A Discussion Future directions discuss adaptive model routing that reduces computational and memory demands for agentic systems.",
            }
        ]
    }

    result = evaluate_grounding_contract(grounding, retrieval_summary)

    assert result["status"] == "pass"
    assert all(check["passed"] for check in result["checks"])


def test_grounded_answer_uses_deterministic_fallback_detects_fallback_note():
    grounded_answer = {
        "answer_summary": "Summary",
        "claims": [
            {
                "claim_text": "The retrieval evidence is incomplete.",
                "source_ids": ["S1"],
                "confidence": "low",
                "note": "Deterministic retrieval fallback",
            }
        ],
    }

    assert grounded_answer_uses_deterministic_fallback(grounded_answer) is True


def test_build_markdown_report_includes_fallback_summary_and_per_question_status():
    report = build_markdown_report(
        [
            {
                "id": "question-a",
                "query": "What changed?",
                "retrieval": {
                    "route": {"label": "comparison", "backend": "deterministic"},
                    "papers_considered": 2,
                    "evidence_chunks": 4,
                },
                "expectation_result": {"status": "pass"},
                "route_result": {"status": "pass", "expected": "comparison"},
                "sources": [],
                "fallback_used": True,
                "grounding_result": {"status": "pass"},
                "answer": "Answer text",
            },
            {
                "id": "question-b",
                "query": "Explain method",
                "retrieval": {
                    "route": {"label": "single_paper_explanation", "backend": "deterministic"},
                    "papers_considered": 1,
                    "evidence_chunks": 3,
                },
                "expectation_result": {"status": "pass"},
                "route_result": {"status": "pass", "expected": "single_paper_explanation"},
                "sources": [],
                "fallback_used": False,
                "grounding_result": {"status": "pass"},
            },
        ]
    )

    assert "Deterministic fallback answers: 1" in report
    assert "Fallback question IDs: question-a" in report
    assert "| question-a | pass | comparison | pass | pass | yes | 2 | 4 | What changed? |" in report
    assert "Deterministic Fallback Used: yes" in report
    assert "Deterministic Fallback Used: no" in report
