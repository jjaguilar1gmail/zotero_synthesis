from pathlib import Path

from ingestion.document_structure import PaperMetadata, PaperRecord, analyze_document_structure
from tools.document_structure_review.html_report import render_batch_index_html, render_document_structure_html


def make_sample_record() -> tuple[PaperRecord, list[tuple[int, str]]]:
    record = PaperRecord(
        paper_id="html-smoke",
        file_path=Path("html-smoke.pdf"),
        metadata=PaperMetadata(
            zotero_item_key="html-smoke",
            attachment_key="html-smoke-attachment",
            title="HTML Smoke Paper",
            collection_id=1,
            parent_paper_id="html-smoke",
        ),
    )
    pages = [
        (1, "ABSTRACT\nA short abstract line.\n1 INTRODUCTION\nSome intro text.\nFigure 1: Example chart\nRESULTS\nA result line."),
    ]
    record.parsed_text = "\n\n".join(text for _, text in pages)
    return record, pages


def test_render_document_structure_html_contains_sections_and_rejections():
    record, pages = make_sample_record()
    report = analyze_document_structure(record, pages)

    html = render_document_structure_html(record, report, pages, summary={"tags": ["subsections"]})

    assert "<html" in html
    assert "HTML Smoke Paper" in html
    assert "Detected Headings" in html
    assert "Rejected Candidates" in html
    assert "Figure 1: Example chart" in html


def test_render_batch_index_html_links_report_files():
    html = render_batch_index_html(
        {
            "input_dir": "sample-dir",
            "pdf_count": 1,
            "summaries": [
                {
                    "file": "sample.pdf",
                    "sections": 2,
                    "headings": 3,
                    "rejected_candidates": 1,
                    "tags": ["inline-abstract"],
                }
            ],
        }
    )

    assert "reports/sample.html" in html
    assert "inline-abstract" in html
