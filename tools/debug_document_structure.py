import sys
import argparse
import json
from pathlib import Path

from llama_index.readers.file import PDFReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.document_structure import (
    PaperMetadata,
    PaperRecord,
    analyze_document_structure,
    create_passage_chunks,
    parse_page_number,
)
from tools.document_structure_review.html_report import render_document_structure_html


def load_fixture(path: Path) -> tuple[PaperRecord, list[tuple[int | None, str]]]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    paper = fixture["paper"]
    paper_record = PaperRecord(
        paper_id=paper["paper_id"],
        file_path=Path(f"{paper['paper_id']}.pdf"),
        metadata=PaperMetadata(
            zotero_item_key=paper["paper_id"],
            attachment_key=f"{paper['paper_id']}-attachment",
            title=paper["title"],
            abstract=paper.get("abstract", ""),
            collection_id=paper.get("collection_id", 0),
            parent_paper_id=paper["paper_id"],
        ),
    )
    pages = [(page.get("page_number"), page["text"]) for page in fixture["pages"]]
    paper_record.parsed_text = "\n\n".join(text for _, text in pages if text)
    return paper_record, pages


def load_pdf(path: Path, title: str | None, abstract: str | None) -> tuple[PaperRecord, list[tuple[int | None, str]]]:
    docs = PDFReader().load_data(file=path)
    pages = [
        (parse_page_number(doc.metadata.get("page_label") or doc.metadata.get("page")), doc.text)
        for doc in docs
    ]
    paper_id = path.stem
    paper_record = PaperRecord(
        paper_id=paper_id,
        file_path=path,
        metadata=PaperMetadata(
            zotero_item_key=paper_id,
            attachment_key=f"{paper_id}-attachment",
            title=title or path.stem,
            abstract=abstract or "",
            collection_id=0,
            parent_paper_id=paper_id,
        ),
    )
    paper_record.parsed_text = "\n\n".join(text for _, text in pages if text)
    return paper_record, pages


def render_text_report(paper_record: PaperRecord, report) -> str:
    lines = [f"Paper: {paper_record.metadata.title}", "", "Detected headings:"]
    for heading in report.headings:
        lines.append(
            f"- page={heading.page_number} level={heading.level} heading={heading.heading_title} reason={heading.detection_reason}"
        )

    lines.append("")
    lines.append("Sections:")
    for section in report.sections:
        chunks = create_passage_chunks(section)
        path = " > ".join(section.path)
        lines.append(
            f"- {path} | label={section.label} | level={section.level} | pages={section.page_start}-{section.page_end} | chunks={len(chunks)}"
        )

    if report.rejected_candidates:
        lines.append("")
        lines.append("Rejected heading candidates:")
        for candidate in report.rejected_candidates:
            lines.append(f"- page={candidate.page_number} reason={candidate.reason} line={candidate.raw_line}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect document structure parsing for a fixture or PDF.")
    parser.add_argument("--fixture", type=Path, help="Path to a JSON fixture file.")
    parser.add_argument("--file", type=Path, help="Path to a PDF file.")
    parser.add_argument("--title", help="Optional paper title override for PDF mode.")
    parser.add_argument("--abstract", help="Optional abstract override for PDF mode.")
    parser.add_argument("--json-out", type=Path, help="Write the full parse report as JSON.")
    parser.add_argument("--html-out", type=Path, help="Write a static HTML review report.")
    args = parser.parse_args()

    if bool(args.fixture) == bool(args.file):
        raise SystemExit("Provide exactly one of --fixture or --file.")

    if args.fixture:
        paper_record, pages = load_fixture(args.fixture)
    else:
        paper_record, pages = load_pdf(args.file, args.title, args.abstract)

    report = analyze_document_structure(paper_record, pages)
    print(render_text_report(paper_record, report))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if args.html_out:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        html = render_document_structure_html(paper_record, report, pages)
        args.html_out.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
