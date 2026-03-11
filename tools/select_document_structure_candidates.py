import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.readers.file import PDFReader

from ingestion.document_structure import analyze_document_structure, parse_page_number
from main import get_paper_records_for_collection, get_zotero_collections
from tools.review_document_structure_batch import summarize_report


def analyze_paper_record(paper_record) -> dict:
    docs = PDFReader().load_data(file=paper_record.file_path)
    pages = [
        (parse_page_number(doc.metadata.get("page_label") or doc.metadata.get("page")), doc.text)
        for doc in docs
    ]
    paper_record.parsed_text = "\n\n".join(text for _, text in pages if text)
    report = analyze_document_structure(paper_record, pages)
    summary = summarize_report(paper_record.file_path, report)
    summary.update(
        {
            "paper_id": paper_record.paper_id,
            "title": paper_record.metadata.title,
            "authors": paper_record.metadata.authors,
            "year": paper_record.metadata.year,
            "collection_id": paper_record.metadata.collection_id,
        }
    )
    return {
        "summary": summary,
        "report": report.model_dump(),
    }


def build_candidate_pool(max_per_collection: int) -> list[dict]:
    collections = []
    for collection in get_zotero_collections():
        records = get_paper_records_for_collection(collection["id"])
        if records:
            collections.append(
                {
                    "id": collection["id"],
                    "name": collection["name"],
                    "records": records,
                }
            )

    collections.sort(key=lambda item: len(item["records"]), reverse=True)

    pool = []
    seen_paths: set[str] = set()
    for collection in collections:
        sorted_records = sorted(
            collection["records"],
            key=lambda record: (
                record.metadata.year or 0,
                record.metadata.title.lower(),
            ),
            reverse=True,
        )
        for record in sorted_records[:max_per_collection]:
            path_key = str(record.file_path).lower()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            pool.append(
                {
                    "collection_id": collection["id"],
                    "collection_name": collection["name"],
                    "record": record,
                }
            )
    return pool


def difficulty_score(summary: dict) -> float:
    extreme_noise_penalty = 0
    if summary["sections"] > 200:
        extreme_noise_penalty += min(summary["sections"] - 200, 300) * 0.5
    if summary["rejected_candidates"] > 80:
        extreme_noise_penalty += (summary["rejected_candidates"] - 80) * 0.8

    return (
        len(summary["tags"]) * 10
        + summary["rejected_candidates"] * 1.5
        + summary["max_heading_level"] * 4
        + min(summary["sections"], 30) * 0.3
        + min(summary["headings"], 40) * 0.2
        - extreme_noise_penalty
    )


def choose_recommendations(analyzed_candidates: list[dict], recommendation_count: int) -> list[dict]:
    remaining = analyzed_candidates[:]
    selected = []
    seen_tags: set[str] = set()
    used_collections: set[int] = set()

    while remaining and len(selected) < recommendation_count:
        best_idx = None
        best_score = None
        for idx, candidate in enumerate(remaining):
            summary = candidate["analysis"]["summary"]
            tags = set(summary["tags"])
            novelty = len(tags - seen_tags) * 25
            collection_bonus = 8 if summary["collection_id"] not in used_collections else 0
            score = novelty + collection_bonus + difficulty_score(summary)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        seen_tags.update(chosen["analysis"]["summary"]["tags"])
        used_collections.add(chosen["analysis"]["summary"]["collection_id"])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select structurally diverse document-structure review candidates from the Zotero library."
    )
    parser.add_argument(
        "--max-per-collection",
        type=int,
        default=3,
        help="Maximum number of papers to analyze per collection. Default: 3",
    )
    parser.add_argument(
        "--recommendation-count",
        type=int,
        default=10,
        help="Number of promotion candidates to recommend. Default: 10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_output") / "document_structure_review" / "candidate_selection.json",
        help="Path for the candidate-selection output JSON.",
    )
    args = parser.parse_args()

    pool = build_candidate_pool(args.max_per_collection)
    analyzed_candidates = []
    for candidate in pool:
        analysis = analyze_paper_record(candidate["record"])
        analyzed_candidates.append(
            {
                "collection_id": candidate["collection_id"],
                "collection_name": candidate["collection_name"],
                "analysis": analysis,
            }
        )

    recommendations = choose_recommendations(
        analyzed_candidates,
        args.recommendation_count,
    )

    payload = {
        "analyzed_count": len(analyzed_candidates),
        "max_per_collection": args.max_per_collection,
        "recommendation_count": args.recommendation_count,
        "recommendations": recommendations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Analyzed {len(analyzed_candidates)} papers")
    print(f"Wrote recommendations to {args.output}")
    print("Top candidates:")
    for candidate in recommendations:
        summary = candidate["analysis"]["summary"]
        print(
            f"- [{candidate['collection_name']}] {summary['title']} | tags={','.join(summary['tags'])} | sections={summary['sections']} | rejected={summary['rejected_candidates']}"
        )


if __name__ == "__main__":
    main()
