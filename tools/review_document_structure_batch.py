import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.debug_document_structure import load_pdf
from ingestion.document_structure import analyze_document_structure, create_passage_chunks
from tools.document_structure_review.html_report import render_batch_index_html, render_document_structure_html


def infer_structure_tags(report) -> list[str]:
    tags: set[str] = set()
    heading_titles = [heading.heading_title for heading in report.headings]
    raw_lines = [heading.raw_line for heading in report.headings]

    if any(any(char.isdigit() for char in line.split()[0]) for line in raw_lines if line.split()):
        tags.add("numbered-headings")
    if any(line.split()[0].rstrip('.)').upper() in {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"} for line in raw_lines if line.split()):
        tags.add("roman-headings")
    if any(title.isupper() and len(title.split()) <= 6 for title in heading_titles):
        tags.add("uppercase-headings")
    if any(heading.detection_reason == "abstract-inline" for heading in report.headings):
        tags.add("inline-abstract")
    if any(heading.level > 1 for heading in report.headings):
        tags.add("subsections")
    if any(section.page_start != section.page_end for section in report.sections):
        tags.add("multi-page-sections")
    if len({section.label for section in report.sections}) < len(report.sections):
        tags.add("repeated-section-labels")
    if report.rejected_candidates:
        tags.add("rejected-heading-candidates")

    return sorted(tags)


def summarize_report(pdf_path: Path, report) -> dict:
    chunk_counts = [len(create_passage_chunks(section)) for section in report.sections]
    return {
        "file": pdf_path.name,
        "path": str(pdf_path),
        "headings": len(report.headings),
        "sections": len(report.sections),
        "rejected_candidates": len(report.rejected_candidates),
        "max_heading_level": max((heading.level for heading in report.headings), default=0),
        "section_labels": [section.label for section in report.sections],
        "heading_titles": [heading.heading_title for heading in report.headings],
        "chunk_count": sum(chunk_counts),
        "max_chunks_in_section": max(chunk_counts, default=0),
        "tags": infer_structure_tags(report),
    }


def load_pdf_with_messages(pdf_path: Path):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        paper_record, pages = load_pdf(pdf_path, title=None, abstract=None)

    raw_messages = [
        line.strip()
        for line in (stdout_buffer.getvalue() + "\n" + stderr_buffer.getvalue()).splitlines()
        if line.strip()
    ]
    unique_messages = []
    for message in raw_messages:
        if message not in unique_messages:
            unique_messages.append(message)
    return paper_record, pages, unique_messages


def write_report_bundle(pdf_path: Path, output_dir: Path, paper_record, pages, report, parse_messages: list[str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    report_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"
    summary = summarize_report(pdf_path, report)
    summary["parse_messages"] = parse_messages or []
    payload = {
        "summary": summary,
        "report": report.model_dump(),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(
        render_document_structure_html(paper_record, report, pages, summary=summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate document-structure review reports for a local PDF corpus."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing PDFs to review.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_output") / "document_structure_review",
        help="Directory where per-paper reports and the corpus summary will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for selecting PDFs inside the input directory. Default: *.pdf",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under the input directory.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    globber = args.input_dir.rglob if args.recursive else args.input_dir.glob
    pdf_paths = sorted(path for path in globber(args.pattern) if path.is_file())
    if not pdf_paths:
        raise SystemExit("No PDF files matched the provided directory and pattern.")

    summaries = []
    reports_dir = args.output_dir / "reports"
    for pdf_path in pdf_paths:
        paper_record, pages, parse_messages = load_pdf_with_messages(pdf_path)
        report = analyze_document_structure(paper_record, pages)
        summaries.append(write_report_bundle(pdf_path, reports_dir, paper_record, pages, report, parse_messages=parse_messages))

    corpus_summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "pdf_count": len(summaries),
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(corpus_summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        render_batch_index_html(corpus_summary),
        encoding="utf-8",
    )

    print(f"Reviewed {len(summaries)} PDFs")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"HTML Index: {args.output_dir / 'index.html'}")
    print(f"Reports: {reports_dir}")


if __name__ == "__main__":
    main()
