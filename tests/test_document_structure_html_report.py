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
            "pdf_count": 3,
            "summaries": [
                {
                    "file": "sample.pdf",
                    "sections": 2,
                    "headings": 3,
                    "rejected_candidates": 1,
                    "chunk_count": 5,
                    "max_heading_level": 1,
                    "max_chunks_in_section": 3,
                    "tags": ["inline-abstract"],
                },
                {
                    "file": "other.pdf",
                    "sections": 4,
                    "headings": 6,
                    "rejected_candidates": 2,
                    "chunk_count": 9,
                    "max_heading_level": 2,
                    "max_chunks_in_section": 4,
                    "tags": ["inline-abstract", "subsections"],
                },
                {
                    "file": "noisy.pdf",
                    "sections": 22,
                    "headings": 20,
                    "rejected_candidates": 8,
                    "chunk_count": 25,
                    "max_heading_level": 4,
                    "max_chunks_in_section": 9,
                    "tags": ["repeated-section-labels", "rejected-heading-candidates"],
                }
            ],
        }
    )

    assert "reports/sample.html" in html
    assert "inline-abstract" in html
    assert "Tag Breakdown" in html
    assert "Total Sections" in html
    assert "Outlier Papers" in html
    assert "reports/noisy.html" in html
    assert "high rejected count (8)" in html
    assert "deep heading nesting (level 4)" in html
    assert ">6<" in html
    assert ">39<" in html
    assert "67%" in html
