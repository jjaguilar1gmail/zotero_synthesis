from __future__ import annotations

from collections import Counter, defaultdict
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
    parse_messages = summary.get("parse_messages", [])
    parse_message_html = "".join(f"<li class=\"card\">{escape(message)}</li>" for message in parse_messages)

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
    <section class=\"panel\">
      <h2>Parse Messages</h2>
      <ul class=\"clean\">{parse_message_html or '<li class="card muted">No parse messages.</li>'}</ul>
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


def _outlier_signals(summary: dict) -> list[str]:
    headings = summary.get("headings", 0)
    sections = summary.get("sections", 0)
    rejected = summary.get("rejected_candidates", 0)
    max_heading_level = summary.get("max_heading_level", 0)
    max_chunks_in_section = summary.get("max_chunks_in_section", 0)
    tags = set(summary.get("tags", []))
    parse_messages = summary.get("parse_messages", [])
    signals = []

    if headings == 0:
        signals.append("no headings detected")
    if sections == 0:
        signals.append("no sections built")
    if rejected >= 5:
        signals.append(f"high rejected count ({rejected})")
    elif headings and rejected / max(headings, 1) >= 0.5:
        signals.append(f"high rejected ratio ({rejected}/{headings})")
    if headings >= 18:
        signals.append(f"dense heading count ({headings})")
    if sections >= 20:
        signals.append(f"dense section count ({sections})")
    if max_heading_level >= 4:
        signals.append(f"deep heading nesting (level {max_heading_level})")
    if max_chunks_in_section >= 8:
        signals.append(f"large section chunk fanout ({max_chunks_in_section})")
    if "repeated-section-labels" in tags:
        signals.append("repeated section labels")
    if "rejected-heading-candidates" in tags and rejected > 0:
        signals.append("rejected heading candidates present")
    if parse_messages:
      signals.append(f"parse warnings ({len(parse_messages)})")

    return signals


def _outlier_score(summary: dict) -> int:
    score = 0
    headings = summary.get("headings", 0)
    sections = summary.get("sections", 0)
    rejected = summary.get("rejected_candidates", 0)
    max_heading_level = summary.get("max_heading_level", 0)
    max_chunks_in_section = summary.get("max_chunks_in_section", 0)
    tags = set(summary.get("tags", []))
    parse_messages = summary.get("parse_messages", [])

    if headings == 0:
        score += 4
    if sections == 0:
        score += 4
    if headings >= 18:
        score += 2
    if sections >= 20:
        score += 2
    if rejected >= 5:
        score += 4
    elif headings and rejected / max(headings, 1) >= 0.5:
        score += 3
    if max_heading_level >= 4:
        score += 2
    if max_chunks_in_section >= 8:
        score += 2
    if "repeated-section-labels" in tags:
        score += 2
    if "rejected-heading-candidates" in tags:
        score += 1
    if parse_messages:
      score += 3

    return score


def render_batch_index_html(corpus_summary: dict) -> str:
    summaries = corpus_summary.get("summaries", [])
    pdf_count = corpus_summary.get("pdf_count", len(summaries))
    total_sections = sum(summary.get("sections", 0) for summary in summaries)
    total_headings = sum(summary.get("headings", 0) for summary in summaries)
    total_rejected = sum(summary.get("rejected_candidates", 0) for summary in summaries)
    total_chunks = sum(summary.get("chunk_count", 0) for summary in summaries)
    avg_sections = f"{(total_sections / pdf_count):.1f}" if pdf_count else "0.0"
    avg_headings = f"{(total_headings / pdf_count):.1f}" if pdf_count else "0.0"
    tag_counts = Counter(tag for summary in summaries for tag in summary.get("tags", []))

    rows = []
    for summary in summaries:
        html_name = f"{Path(summary['file']).stem}.html"
        parse_message_count = len(summary.get("parse_messages", []))
        rows.append(
            f"<tr><td><a href=\"reports/{escape(html_name)}\">{escape(summary['file'])}</a></td><td>{summary['sections']}</td><td>{summary['headings']}</td><td>{summary['rejected_candidates']}</td><td>{parse_message_count}</td><td>{escape(', '.join(summary.get('tags', [])))}</td></tr>"
        )

    tag_rows = []
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        tag_rows.append(
            f"<tr><td>{escape(tag)}</td><td>{count}</td><td>{count / pdf_count:.0%}</td></tr>"
        )

    tag_panel = (
        f"<table class=\"table\"><thead><tr><th>Tag</th><th>Papers</th><th>Coverage</th></tr></thead><tbody>{''.join(tag_rows)}</tbody></table>"
        if tag_rows
        else "<p class=\"muted\">No structure tags were inferred for this corpus.</p>"
    )

    outlier_items = []
    ranked_outliers = []
    for summary in summaries:
        signals = _outlier_signals(summary)
        if signals:
            ranked_outliers.append((
                _outlier_score(summary),
                summary.get("file", "unknown"),
                summary,
                signals,
            ))

    ranked_outliers.sort(key=lambda item: (-item[0], item[1]))
    for score, _, summary, signals in ranked_outliers[:5]:
        html_name = f"{Path(summary['file']).stem}.html"
        signal_html = "".join(f"<span class=\"tag\">{escape(signal)}</span>" for signal in signals)
        parse_message_count = len(summary.get("parse_messages", []))
        outlier_items.append(
            f"<li class=\"card\"><strong><a href=\"reports/{escape(html_name)}\">{escape(summary['file'])}</a></strong><div class=\"muted\">outlier score {score} | sections {summary.get('sections', 0)} | headings {summary.get('headings', 0)} | rejected {summary.get('rejected_candidates', 0)} | parse messages {parse_message_count}</div><div class=\"tag-list\">{signal_html}</div></li>"
        )

    outlier_panel = (
        f"<ul class=\"clean\">{''.join(outlier_items)}</ul>"
        if outlier_items
        else "<p class=\"muted\">No obvious outliers detected in this corpus.</p>"
    )

    body = f"""
    <section class=\"panel\">
      <h1>Document Structure Review</h1>
      <div class=\"meta\">
        <div class=\"meta-item\"><span class=\"label\">Input Dir</span><span class=\"value\">{escape(corpus_summary.get('input_dir', ''))}</span></div>
        <div class=\"meta-item\"><span class=\"label\">PDF Count</span><span class=\"value\">{escape(str(pdf_count))}</span></div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Corpus Summary</h2>
      <div class=\"summary-grid\">
        <div class=\"summary-item\"><span class=\"label\">Total Sections</span><span class=\"value\">{total_sections}</span></div>
        <div class=\"summary-item\"><span class=\"label\">Total Headings</span><span class=\"value\">{total_headings}</span></div>
        <div class=\"summary-item\"><span class=\"label\">Total Rejected</span><span class=\"value\">{total_rejected}</span></div>
        <div class=\"summary-item\"><span class=\"label\">Total Chunks</span><span class=\"value\">{total_chunks}</span></div>
        <div class=\"summary-item\"><span class=\"label\">Avg Sections / Paper</span><span class=\"value\">{avg_sections}</span></div>
        <div class=\"summary-item\"><span class=\"label\">Avg Headings / Paper</span><span class=\"value\">{avg_headings}</span></div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Tag Breakdown</h2>
      {tag_panel}
    </section>
    <section class=\"panel\">
      <h2>Outlier Papers</h2>
      {outlier_panel}
    </section>
    <section class=\"panel\">
      <h2>Per-Paper Review</h2>
      <table class=\"table\">
        <thead><tr><th>Paper</th><th>Sections</th><th>Headings</th><th>Rejected</th><th>Parse Msgs</th><th>Tags</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """
    return _html_document("Document Structure Review", body)
