import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    MIN_GROUNDED_CLAIM_SUPPORT,
    compute_claim_support_against_texts,
    infer_calibrated_confidence,
    format_grounded_answer_text,
    format_sources_from_evidence,
    get_collection_index,
    run_retrieval_pipeline,
    synthesize_grounded_answer,
)


def load_question_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        questions = payload
    else:
        questions = payload.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question set must be a non-empty list or an object containing a non-empty 'questions' list.")
    return questions


def evaluate_expectations(expect: dict[str, Any] | None, sources: list[dict[str, Any]], retrieval_summary: dict[str, Any]) -> dict[str, Any]:
    if not expect:
        return {"status": "no-expectations", "checks": []}

    checks: list[dict[str, Any]] = []
    paper_count = retrieval_summary["papers_considered"]
    evidence_count = retrieval_summary["evidence_chunks"]
    source_labels = {source.get("section_label") for source in sources if source.get("section_label")}
    source_text = " ".join(
        " ".join(
            filter(
                None,
                [
                    str(source.get("title") or ""),
                    str(source.get("section_heading") or ""),
                    str(source.get("snippet") or ""),
                ],
            )
        )
        for source in sources
    ).lower()

    min_papers = expect.get("min_papers")
    if min_papers is not None:
        checks.append(
            {
                "name": "min_papers",
                "passed": paper_count >= int(min_papers),
                "actual": paper_count,
                "expected": int(min_papers),
            }
        )

    max_papers = expect.get("max_papers")
    if max_papers is not None:
        checks.append(
            {
                "name": "max_papers",
                "passed": paper_count <= int(max_papers),
                "actual": paper_count,
                "expected": int(max_papers),
            }
        )

    min_evidence_chunks = expect.get("min_evidence_chunks")
    if min_evidence_chunks is not None:
        checks.append(
            {
                "name": "min_evidence_chunks",
                "passed": evidence_count >= int(min_evidence_chunks),
                "actual": evidence_count,
                "expected": int(min_evidence_chunks),
            }
        )

    section_labels_any = expect.get("section_labels_any") or []
    if section_labels_any:
        normalized_expected = {label.strip() for label in section_labels_any}
        checks.append(
            {
                "name": "section_labels_any",
                "passed": bool(source_labels & normalized_expected),
                "actual": sorted(source_labels),
                "expected": sorted(normalized_expected),
            }
        )

    keywords_any = expect.get("keywords_any") or []
    if keywords_any:
        normalized_keywords = [keyword.lower() for keyword in keywords_any]
        checks.append(
            {
                "name": "keywords_any",
                "passed": any(keyword in source_text for keyword in normalized_keywords),
                "actual": normalized_keywords,
                "expected": normalized_keywords,
            }
        )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return {"status": status, "checks": checks}


def evaluate_route_expectation(expected_route: str | None, retrieval_summary: dict[str, Any]) -> dict[str, Any]:
    actual_route = retrieval_summary["route"]["label"]
    if not expected_route:
        return {"status": "no-expectation", "expected": None, "actual": actual_route}
    passed = actual_route == expected_route
    return {
        "status": "pass" if passed else "fail",
        "expected": expected_route,
        "actual": actual_route,
    }


def evaluate_grounding_contract(answer: dict[str, Any] | None, retrieval_summary: dict[str, Any]) -> dict[str, Any]:
    if not answer:
        return {"status": "no-answer", "checks": []}

    source_lookup = {
        source.get("source_id"): source
        for source in retrieval_summary.get("sources", [])
        if source.get("source_id")
    }
    valid_source_ids = set(source_lookup)
    checks: list[dict[str, Any]] = []
    claims = answer.get("claims") or []

    checks.append(
        {
            "name": "answer_summary_present",
            "passed": bool(str(answer.get("answer_summary") or "").strip()),
            "actual": bool(str(answer.get("answer_summary") or "").strip()),
            "expected": True,
        }
    )
    checks.append(
        {
            "name": "all_claims_cited",
            "passed": all(claim.get("source_ids") for claim in claims),
            "actual": len([claim for claim in claims if claim.get("source_ids")]),
            "expected": len(claims),
        }
    )
    checks.append(
        {
            "name": "all_citations_valid",
            "passed": all(source_id in valid_source_ids for claim in claims for source_id in (claim.get("source_ids") or [])),
            "actual": sorted({source_id for claim in claims for source_id in (claim.get("source_ids") or [])}),
            "expected": sorted(valid_source_ids),
        }
    )

    support_scores: list[float] = []
    calibrated_confidences_match = True
    for claim in claims:
        supporting_texts = []
        for source_id in claim.get("source_ids") or []:
            source = source_lookup.get(source_id)
            if not source:
                continue
            supporting_texts.append(
                str(
                    source.get("support_text")
                    or " ".join(
                        filter(
                            None,
                            [
                                str(source.get("title") or ""),
                                str(source.get("section_heading") or source.get("section_label") or ""),
                                str(source.get("snippet") or ""),
                            ],
                        )
                    )
                )
            )
        max_support, avg_support = compute_claim_support_against_texts(str(claim.get("claim_text") or ""), supporting_texts)
        support_scores.append(max_support)
        calibrated_confidence = infer_calibrated_confidence(len(claim.get("source_ids") or []), max_support, avg_support)
        if claim.get("confidence") not in {calibrated_confidence, claim.get("calibrated_confidence")}:
            if claim.get("confidence") not in {"low", "medium", "high"}:
                calibrated_confidences_match = False
            elif {"low": 0, "medium": 1, "high": 2}[claim.get("confidence")] > {"low": 0, "medium": 1, "high": 2}[calibrated_confidence]:
                calibrated_confidences_match = False

    checks.append(
        {
            "name": "claims_have_support_overlap",
            "passed": all(score >= MIN_GROUNDED_CLAIM_SUPPORT for score in support_scores) if claims else True,
            "actual": [round(score, 3) for score in support_scores],
            "expected": f">= {MIN_GROUNDED_CLAIM_SUPPORT}",
        }
    )
    checks.append(
        {
            "name": "confidence_not_overstated",
            "passed": calibrated_confidences_match,
            "actual": [claim.get("confidence") for claim in claims],
            "expected": "backend-calibrated confidence",
        }
    )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return {"status": status, "checks": checks}


def build_retrieval_summary(result) -> dict[str, Any]:
    return {
        "route": {
            "label": result.plan.route_label,
            "backend": result.plan.route_backend,
            "features": result.plan.route_features,
        },
        "comparative": result.plan.comparative,
        "section_labels": result.plan.section_labels,
        "year_from": result.plan.year_from,
        "year_to": result.plan.year_to,
        "dense_candidates": len(result.dense_candidates),
        "filtered_candidates": len(result.filtered_candidates),
        "evidence_chunks": len(result.selected_chunks),
        "papers_considered": len({chunk.paper_id for chunk in result.selected_chunks}),
    }


def grounded_answer_uses_deterministic_fallback(grounded_answer: dict[str, Any]) -> bool:
    for claim in grounded_answer.get("claims") or []:
        if claim.get("note") == "Deterministic retrieval fallback":
            return True
    return False


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    fallback_count = sum(1 for result in results if result.get("fallback_used"))
    fallback_ids = [result["id"] for result in results if result.get("fallback_used")]
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Questions: {len(results)}",
        f"Deterministic fallback answers: {fallback_count}",
        f"Fallback question IDs: {', '.join(fallback_ids) if fallback_ids else 'none'}",
        "",
        "| ID | Status | Route | Route Check | Grounding | Fallback | Papers | Evidence | Query |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        summary = result["retrieval"]
        grounding_status = result.get("grounding_result", {}).get("status", "n/a")
        fallback_status = "yes" if result.get("fallback_used") else "no"
        lines.append(
            f"| {result['id']} | {result['expectation_result']['status']} | {summary['route']['label']} | {result['route_result']['status']} | {grounding_status} | {fallback_status} | {summary['papers_considered']} | {summary['evidence_chunks']} | {result['query']} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result['id']}",
                "",
                f"Query: {result['query']}",
                "",
                f"Status: {result['expectation_result']['status']}",
                f"Route: {result['retrieval']['route']['label']} ({result['retrieval']['route']['backend']})",
                f"Expected Route: {result['route_result']['expected'] or 'n/a'}",
                f"Route Check: {result['route_result']['status']}",
                "",
                "Top Sources:",
            ]
        )
        for source in result["sources"][:5]:
            page_text = f"p.{source['page_number']}" if source.get("page_number") is not None else "page unknown"
            lines.append(
                f"- {source['source_id']}: {source.get('title') or 'Untitled'} | {source.get('section_heading') or source.get('section_label') or 'Unlabeled'} | {page_text} | score={source.get('score')}"
            )
        if result.get("answer"):
            lines.extend(["", "Answer Preview:", "", result["answer"]])
        if result.get("grounding_result"):
            lines.extend(["", f"Grounding Check: {result['grounding_result']['status']}"])
        if "fallback_used" in result:
            lines.append(f"Deterministic Fallback Used: {'yes' if result['fallback_used'] else 'no'}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retrieval evaluation question set against an indexed Zotero collection.")
    parser.add_argument("--collection-id", type=int, required=True, help="Indexed Zotero collection id to evaluate.")
    parser.add_argument("--questions-file", type=Path, required=True, help="Path to a JSON question set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_output") / "retrieval_eval",
        help="Directory where JSON and Markdown evaluation outputs will be written.",
    )
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="Also synthesize an answer for each query using the selected evidence. This consumes LLM calls.",
    )
    args = parser.parse_args()

    index = get_collection_index(args.collection_id)
    if index is None:
        raise SystemExit(f"Collection {args.collection_id} is not indexed. Index it first.")

    questions = load_question_set(args.questions_file)
    results: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        query = str(question.get("query") or "").strip()
        if not query:
            raise SystemExit(f"Question entry {idx} is missing a non-empty 'query'.")

        retrieval = run_retrieval_pipeline(index, query, collection_id=args.collection_id)
        retrieval_summary = build_retrieval_summary(retrieval)
        sources = format_sources_from_evidence(retrieval.selected_chunks)
        expectation_result = evaluate_expectations(question.get("expect"), sources, retrieval_summary)
        route_result = evaluate_route_expectation(question.get("expected_route"), retrieval_summary)

        result = {
            "id": question.get("id") or f"question-{idx}",
            "query": query,
            "notes": question.get("notes") or "",
            "expected_route": question.get("expected_route"),
            "retrieval": retrieval_summary,
            "expect": question.get("expect") or {},
            "expectation_result": expectation_result,
            "route_result": route_result,
            "sources": sources,
        }
        if args.with_answer:
            grounded_answer = synthesize_grounded_answer(query, "", retrieval)
            result["grounded_answer"] = grounded_answer.model_dump()
            result["fallback_used"] = grounded_answer_uses_deterministic_fallback(result["grounded_answer"])
            result["answer"] = format_grounded_answer_text(grounded_answer)
            retrieval_summary_with_sources = dict(retrieval_summary)
            retrieval_summary_with_sources["sources"] = sources
            result["grounding_result"] = evaluate_grounding_contract(result["grounded_answer"], retrieval_summary_with_sources)
        results.append(result)

    summary = {
        "collection_id": args.collection_id,
        "question_count": len(results),
        "pass_count": sum(result["expectation_result"]["status"] == "pass" for result in results),
        "fail_count": sum(result["expectation_result"]["status"] == "fail" for result in results),
        "route_pass_count": sum(result["route_result"]["status"] == "pass" for result in results),
        "route_fail_count": sum(result["route_result"]["status"] == "fail" for result in results),
        "fallback_count": sum(1 for result in results if result.get("fallback_used")),
        "fallback_question_ids": [result["id"] for result in results if result.get("fallback_used")],
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown_report(results), encoding="utf-8")

    print(f"Evaluated {len(results)} questions")
    print(f"Summary: {summary_path}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
