from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Optional

from ingestion.document_structure import create_passage_chunks, normalize_page_text


def _html_document(title: str, body: str) -> str:
    styles = """
    :root {
      --bg: #f4f1ea;
      --surface: #fffaf0;
      --surface-strong: #f1eadb;
      --border: #d8cdb7;
      --text: #1f2421;
      --muted: #6f6a5f;
      --accent: #2f6f5e;
      --accent-soft: #dcefe7;
      --warn: #b45b3e;
      --warn-soft: #f8e1d8;
      --code: #efe7d6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      font-family: Georgia, "Times New Roman", serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 4px 18px rgba(55, 43, 20, 0.06);
    }
    h1, h2, h3 { margin: 0 0 10px; line-height: 1.15; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.2rem; }
    h3 { font-size: 1rem; }
    p { margin: 0; }
    .meta, .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .meta-item, .summary-item {
      padding: 10px 12px;
      border-radius: 10px;
      background: var(--surface-strong);
      border: 1px solid var(--border);
    }
    .label {
      display: block;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .value { font-size: 0.95rem; }
    .tag-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .tag {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(47, 111, 94, 0.2);
      font-size: 0.8rem;
    }
    ul.clean {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }
    li.card {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: var(--surface);
    }
    .muted { color: var(--muted); }
    .page-grid {
      display: grid;
      gap: 18px;
    }
    .page-block {
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: var(--surface);
    }
    .page-head {
      padding: 12px 14px;
      background: var(--surface-strong);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    .page-lines { padding: 10px 0; }
    .line {
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 10px;
      padding: 4px 14px;
      border-left: 4px solid transparent;
    }
    .line-number { color: var(--muted); font-size: 0.8rem; }
    .line-text {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 0.85rem;
    }
    .line.heading {
      background: var(--accent-soft);
      border-left-color: var(--accent);
    }
    .line.rejected {
      background: var(--warn-soft);
      border-left-color: var(--warn);
    }
    .line.both {
      background: linear-gradient(90deg, var(--accent-soft), var(--warn-soft));
      border-left-color: var(--warn);
    }
    .annotation {
      display: inline-block;
      margin-left: 8px;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface);
    }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    .table th, .table td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    .table th {
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code {
      background: var(--code);
      padding: 1px 6px;
      border-radius: 6px;
      font-family: "Consolas", "SFMono-Regular", monospace;
    }
    .two-col {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }
    """
    return f"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{escape(title)}</title><style>{styles}</style></head><body><main>{body}</main></body></html>"


def _render_page_lines(pages: list[tuple[Optional[int], str]], report) -> str:
    headings_by_page = defaultdict(dict)
    for heading in report.headings:
        headings_by_page[heading.page_number][heading.line_index] = heading

    rejected_by_page = defaultdict(dict)
    for candidate in report.rejected_candidates:
        rejected_by_page[candidate.page_number][candidate.line_index] = candidate

    page_blocks = []
    for page_number, page_text in pages:
        normalized = normalize_page_text(page_text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        rendered_lines = []
        for idx, line in enumerate(lines):
            heading = headings_by_page[page_number].get(idx)
            rejected = rejected_by_page[page_number].get(idx)
            state = "both" if heading and rejected else "heading" if heading else "rejected" if rejected else ""
            badges = []
            if heading:
                badges.append(f"<span class=\"annotation\">heading: {escape(heading.heading_title)} | level {heading.level}</span>")
            if rejected:
                badges.append(f"<span class=\"annotation\">rejected: {escape(rejected.reason)}</span>")
            rendered_lines.append(
                f"<div class=\"line {state}\"><div class=\"line-number\">{idx + 1}</div><div class=\"line-text\">{escape(line)}{''.join(badges)}</div></div>"
            )

        page_blocks.append(
            f"<section class=\"page-block\"><div class=\"page-head\"><h3>Page {escape(str(page_number))}</h3><div class=\"muted\">{len(lines)} lines</div></div><div class=\"page-lines\">{''.join(rendered_lines)}</div></section>"
        )
    return "".join(page_blocks)


def render_document_structure_html(paper_record, report, pages, summary: Optional[dict] = None) -> str:
    if summary is None:
        summary = {}

    summary_items = []
    for key, value in [
        ("Headings", len(report.headings)),
        ("Sections", len(report.sections)),
        ("Rejected", len(report.rejected_candidates)),
        ("Chunks", sum(len(create_passage_chunks(section)) for section in report.sections)),
    ]:
        summary_items.append(f"<div class=\"summary-item\"><span class=\"label\">{key}</span><span class=\"value\">{escape(str(value))}</span></div>")

    tags = summary.get("tags", [])
    tag_html = "".join(f"<span class=\"tag\">{escape(tag)}</span>" for tag in tags) or "<span class=\"muted\">No tags</span>"

    headings_html = "".join(
        f"<li class=\"card\"><strong>{escape(heading.heading_title)}</strong><div class=\"muted\">page {escape(str(heading.page_number))} | level {heading.level} | {escape(heading.detection_reason)}</div><div>{escape(heading.raw_line)}</div></li>"
        for heading in report.headings
    ) or "<li class=\"card muted\">No headings detected.</li>"

    rejected_html = "".join(
        f"<li class=\"card\"><strong>{escape(candidate.reason)}</strong><div class=\"muted\">page {escape(str(candidate.page_number))} | line {candidate.line_index + 1}</div><div>{escape(candidate.raw_line)}</div></li>"
        for candidate in report.rejected_candidates
    ) or "<li class=\"card muted\">No rejected candidates.</li>"

    section_rows = []
    for section in report.sections:
        section_rows.append(
            f"<tr><td>{escape(' > '.join(section.path))}</td><td>{escape(section.label)}</td><td>{section.level}</td><td>{escape(str(section.page_start))}-{escape(str(section.page_end))}</td><td>{len(create_passage_chunks(section))}</td></tr>"
        )

    body = f"""
    <section class=\"panel\">
      <h1>{escape(paper_record.metadata.title)}</h1>
      <div class=\"meta\">
        <div class=\"meta-item\"><span class=\"label\">Paper Id</span><span class=\"value\">{escape(paper_record.paper_id)}</span></div>
        <div class=\"meta-item\"><span class=\"label\">Source</span><span class=\"value\">{escape(str(paper_record.file_path))}</span></div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Summary</h2>
      <div class=\"summary-grid\">{''.join(summary_items)}</div>
      <div style=\"margin-top: 12px\"><span class=\"label\">Tags</span><div class=\"tag-list\">{tag_html}</div></div>
    </section>
    <section class=\"panel two-col\">
      <div>
        <h2>Detected Headings</h2>
        <ul class=\"clean\">{headings_html}</ul>
      </div>
      <div>
        <h2>Rejected Candidates</h2>
        <ul class=\"clean\">{rejected_html}</ul>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Sections</h2>
      <table class=\"table\">
        <thead><tr><th>Path</th><th>Label</th><th>Level</th><th>Pages</th><th>Chunks</th></tr></thead>
        <tbody>{''.join(section_rows)}</tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Page Review</h2>
      <div class=\"page-grid\">{_render_page_lines(pages, report)}</div>
    </section>
    """
    return _html_document(paper_record.metadata.title, body)


def render_batch_index_html(corpus_summary: dict) -> str:
    rows = []
    for summary in corpus_summary.get("summaries", []):
        html_name = f"{Path(summary['file']).stem}.html"
        rows.append(
            f"<tr><td><a href=\"reports/{escape(html_name)}\">{escape(summary['file'])}</a></td><td>{summary['sections']}</td><td>{summary['headings']}</td><td>{summary['rejected_candidates']}</td><td>{escape(', '.join(summary.get('tags', [])))}</td></tr>"
        )

    body = f"""
    <section class=\"panel\">
      <h1>Document Structure Review</h1>
      <div class=\"meta\">
        <div class=\"meta-item\"><span class=\"label\">Input Dir</span><span class=\"value\">{escape(corpus_summary.get('input_dir', ''))}</span></div>
        <div class=\"meta-item\"><span class=\"label\">PDF Count</span><span class=\"value\">{escape(str(corpus_summary.get('pdf_count', 0)))}</span></div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Corpus Summary</h2>
      <table class=\"table\">
        <thead><tr><th>Paper</th><th>Sections</th><th>Headings</th><th>Rejected</th><th>Tags</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """
    return _html_document("Document Structure Review", body)
