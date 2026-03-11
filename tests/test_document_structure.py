import json
from pathlib import Path

from ingestion.document_structure import (
    PaperMetadata,
    PaperRecord,
    analyze_document_structure,
    create_passage_chunks,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "document_structure"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def make_paper_record(fixture: dict) -> tuple[PaperRecord, list[tuple[int, str]]]:
    paper = fixture["paper"]
    record = PaperRecord(
        paper_id=paper["paper_id"],
        file_path=Path(f"{paper['paper_id']}.pdf"),
        metadata=PaperMetadata(
            zotero_item_key=paper["paper_id"],
            attachment_key=f"{paper['paper_id']}-attachment",
            title=paper["title"],
            abstract=paper["abstract"],
            collection_id=paper["collection_id"],
            parent_paper_id=paper["paper_id"],
        ),
    )
    pages = [(page["page_number"], page["text"]) for page in fixture["pages"]]
    record.parsed_text = "\n\n".join(text for _, text in pages)
    return record, pages


def test_numbered_headings_and_subsections_are_detected():
    fixture = load_fixture("numbered_subsections")
    paper_record, pages = make_paper_record(fixture)
    report = analyze_document_structure(paper_record, pages)

    assert [heading.heading_title for heading in report.headings] == fixture["expected"]["headings"]
    assert [section.label for section in report.sections] == fixture["expected"]["labels"]
    assert [section.level for section in report.sections] == fixture["expected"]["levels"]
    assert [section.path for section in report.sections] == fixture["expected"]["paths"]


def test_inline_body_phrases_are_not_promoted_to_headings():
    fixture = load_fixture("false_positive_guardrails")
    paper_record, pages = make_paper_record(fixture)
    report = analyze_document_structure(paper_record, pages)

    assert [heading.heading_title for heading in report.headings] == fixture["expected"]["headings"]
    assert [section.label for section in report.sections] == fixture["expected"]["labels"]
    assert [section.level for section in report.sections] == fixture["expected"]["levels"]
    assert [section.path for section in report.sections] == fixture["expected"]["paths"]

    rejected_lines = [candidate.raw_line for candidate in report.rejected_candidates]
    assert fixture["expected"]["rejected_contains"][0] in rejected_lines


def test_passage_chunks_are_created_from_detected_sections():
    fixture = load_fixture("numbered_subsections")
    paper_record, pages = make_paper_record(fixture)
    report = analyze_document_structure(paper_record, pages)

    first_section_chunks = create_passage_chunks(report.sections[0])

    assert first_section_chunks
    assert first_section_chunks[0].section_id == report.sections[0].section_id
    assert first_section_chunks[0].page_start == report.sections[0].page_start
