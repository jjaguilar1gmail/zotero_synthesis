from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel, Field

PASSAGE_SPLITTER = SentenceSplitter(
    chunk_size=700,
    chunk_overlap=120,
    include_metadata=False,
    include_prev_next_rel=False,
)

SECTION_ALIASES = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction"],
    "Background": ["background"],
    "Related Work": ["related work", "related works", "literature review", "prior work"],
    "Methods": [
        "method",
        "methods",
        "methodology",
        "materials and methods",
        "experimental setup",
        "approach",
        "implementation details",
    ],
    "Results": ["result", "results", "experiments", "evaluation", "findings"],
    "Discussion": ["discussion", "analysis"],
    "Conclusion": ["conclusion", "conclusions", "concluding remarks"],
}

SECTION_ORDER = {
    "Abstract": 0,
    "Introduction": 1,
    "Background": 2,
    "Related Work": 3,
    "Methods": 4,
    "Results": 5,
    "Discussion": 6,
    "Conclusion": 7,
    "Body": 8,
}

CORE_SECTION_LABELS = {
    "Abstract",
    "Introduction",
    "Background",
    "Related Work",
    "Methods",
    "Results",
    "Discussion",
    "Conclusion",
}

ALLOWED_NON_CANONICAL_SHORT_HEADINGS = {
    "references",
    "acknowledgements",
    "acknowledgments",
    "examples",
    "appendix",
    "appendices",
}

DIAGRAM_LABEL_WORDS = {
    "image",
    "encoder",
    "audio",
    "text",
    "metadata",
    "fusion",
    "prompt",
    "visualization",
    "embedding",
    "mask",
    "bbox",
    "module",
    "agent",
    "execution",
    "input",
    "output",
    "generate",
}

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<prefix>(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z]))[.)]?\s+(?P<title>.+)$",
    re.IGNORECASE,
)


class PaperMetadata(BaseModel):
    zotero_item_key: str
    attachment_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal_venue: str = ""
    abstract: str = ""
    tags: list[str] = Field(default_factory=list)
    collection_id: int
    page_number: Optional[int] = None
    section_label: str = ""
    parent_paper_id: str


class HeadingMatch(BaseModel):
    page_number: Optional[int] = None
    line_index: int
    raw_line: str
    heading_title: str
    canonical_label: Optional[str] = None
    level: int = 1
    inline_text: str = ""
    detection_reason: str


class RejectedHeadingCandidate(BaseModel):
    page_number: Optional[int] = None
    line_index: int
    raw_line: str
    reason: str


class PassageChunkRecord(BaseModel):
    chunk_id: str
    section_id: str
    text: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class SectionRecord(BaseModel):
    section_id: str
    label: str
    canonical_label: str
    heading: str
    path: list[str] = Field(default_factory=list)
    level: int = 1
    order: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunks: list[PassageChunkRecord] = Field(default_factory=list)


class DocumentStructureReport(BaseModel):
    headings: list[HeadingMatch] = Field(default_factory=list)
    rejected_candidates: list[RejectedHeadingCandidate] = Field(default_factory=list)
    sections: list[SectionRecord] = Field(default_factory=list)


class PaperRecord(BaseModel):
    paper_id: str
    file_path: Path
    metadata: PaperMetadata
    parsed_text: str = ""
    sections: list[SectionRecord] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    generated_summary: Optional[str] = None


def parse_page_number(value: Optional[str | int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def normalize_page_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_section_heading(line: str) -> str:
    heading = line.strip().lower()
    heading = re.sub(r"^[\divxlcdm]+(?:\.[\divxlcdm]+)*[.)]?\s+", "", heading)
    heading = heading.replace("&", "and")
    heading = re.sub(r"[^a-z0-9\s-]", "", heading)
    heading = re.sub(r"\s+", " ", heading)
    return heading.strip(" -")


def find_canonical_section_label(normalized_heading: str) -> Optional[str]:
    for label, aliases in SECTION_ALIASES.items():
        if normalized_heading in aliases:
            return label
    return None


def split_heading_prefix(line: str) -> tuple[int, str, bool]:
    match = NUMBERED_HEADING_RE.match(line.strip())
    if not match:
        return 1, line.strip(), False

    prefix = match.group("prefix")
    title = match.group("title").strip()
    if prefix[0].isdigit():
        level = prefix.count(".") + 1
    else:
        level = 1
    return level, title, True


def clean_heading_title(title: str) -> str:
    cleaned = title.strip()
    cleaned = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\.{2,}\s*$", "", cleaned)
    return cleaned.strip(" :-")


def alpha_ratio(text: str) -> float:
    alpha_chars = [char for char in text if char.isalpha()]
    alnum_chars = [char for char in text if char.isalnum()]
    if not alnum_chars:
        return 0.0
    return len(alpha_chars) / len(alnum_chars)


def digit_ratio(text: str) -> float:
    digits = [char for char in text if char.isdigit()]
    alnum_chars = [char for char in text if char.isalnum()]
    if not alnum_chars:
        return 0.0
    return len(digits) / len(alnum_chars)


def looks_like_table_of_contents_marker(line: str) -> bool:
    return line.strip().lower() == "table of contents"


def looks_like_table_of_contents_entry(line: str) -> bool:
    stripped = line.strip()
    if re.search(r"\.{2,}\s*\d+\s*$", stripped):
        return True
    return bool(re.match(r"^(?:\d+\.|[IVXLCDM]+\.)\s+.+\.{2,}\s*\d+\s*$", stripped, re.IGNORECASE))


def looks_like_front_matter_metadata_line(line: str) -> bool:
    lower = line.strip().lower()
    metadata_tokens = [
        "received:",
        "accepted:",
        "published online",
        "copyright",
        "doi",
        "acm isbn",
        "permission to make digital or hard copies",
        "school of",
        "institute of",
        "university",
        "laboratory",
        "department",
        "e-mail:",
        "keywords",
        "key words",
        "communicated by",
    ]
    if any(token in lower for token in metadata_tokens):
        return True
    if "@" in line:
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}$", line.strip()):
        return True
    if re.match(r"^[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)+$", line.strip()):
        return True
    return False


def looks_like_figure_or_table_heading(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if lower.startswith(("figure ", "fig. ", "table ")):
        return True
    if re.match(r"^(figure|fig\.|table)\s*\d+", lower):
        return True
    normalized_words = [word for word in normalize_section_heading(stripped).split() if word]
    if normalized_words and len(normalized_words) <= 3 and all(word in DIAGRAM_LABEL_WORDS for word in normalized_words):
        return True
    if digit_ratio(stripped) > 0.35 and alpha_ratio(stripped) < 0.5:
        return True
    return False


def is_allowed_short_heading(normalized_heading: str) -> bool:
    return normalized_heading in ALLOWED_NON_CANONICAL_SHORT_HEADINGS or normalized_heading in {
        alias
        for aliases in SECTION_ALIASES.values()
        for alias in aliases
    }


def is_short_heading_candidate(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 12 or len(line) > 90:
        return False
    if line.endswith((".", ";", ",")):
        return False
    normalized = normalize_section_heading(line)
    if len(words) < 2 and not is_allowed_short_heading(normalized):
        return False
    if digit_ratio(line) > 0.3 or alpha_ratio(line) < 0.55:
        return False
    return line.isupper() or line == line.title()


def detect_section_heading(
    line: str,
    page_number: Optional[int] = None,
    line_index: int = 0,
) -> tuple[Optional[HeadingMatch], Optional[RejectedHeadingCandidate]]:
    stripped = line.strip()
    if not stripped:
        return None, None

    normalized = normalize_section_heading(stripped)
    if not normalized:
        return None, None

    numbered_level, numbered_title, has_numbering = split_heading_prefix(stripped)

    for label, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized == alias:
                return HeadingMatch(
                    page_number=page_number,
                    line_index=line_index,
                    raw_line=stripped,
                    heading_title=clean_heading_title(numbered_title),
                    canonical_label=label,
                    level=numbered_level if has_numbering else 1,
                    detection_reason="alias-exact",
                ), None

            if label == "Abstract" and normalized.startswith(f"{alias} "):
                original_match = re.match(
                    rf"^(?:[\divxlcdm]+(?:\.[\divxlcdm]+)*[.)]?\s+)?{re.escape(alias)}(?:\s*[:.\-]\s+|\s+)(.+)$",
                    stripped,
                    re.IGNORECASE,
                )
                if original_match:
                    return HeadingMatch(
                        page_number=page_number,
                        line_index=line_index,
                        raw_line=stripped,
                        heading_title="Abstract",
                        canonical_label=label,
                        level=1,
                        inline_text=original_match.group(1).strip(),
                        detection_reason="abstract-inline",
                    ), None

    level, title, has_numbering = numbered_level, numbered_title, has_numbering
    normalized_title = normalize_section_heading(title)
    canonical_label = find_canonical_section_label(normalized_title)

    if has_numbering:
        if (
            title.endswith((".", ";", ","))
            or len(title) > 120
            or len(title.split()) > 14
            or (canonical_label is None and title and title[0].islower())
            or looks_like_figure_or_table_heading(title)
        ):
            return None, RejectedHeadingCandidate(
                page_number=page_number,
                line_index=line_index,
                raw_line=stripped,
                reason="numbered-line-rejected",
            )

        heading_title = clean_heading_title(title)
        return HeadingMatch(
            page_number=page_number,
            line_index=line_index,
            raw_line=stripped,
            heading_title=heading_title,
            canonical_label=canonical_label,
            level=level,
            detection_reason="numbered-heading",
        ), None

    if is_short_heading_candidate(stripped):
        heading_title = clean_heading_title(stripped)
        return HeadingMatch(
            page_number=page_number,
            line_index=line_index,
            raw_line=stripped,
            heading_title=heading_title,
            canonical_label=canonical_label,
            level=1,
            detection_reason="short-heading",
        ), None

    for aliases in SECTION_ALIASES.values():
        for alias in aliases:
            if normalized.startswith(f"{alias} "):
                return None, RejectedHeadingCandidate(
                    page_number=page_number,
                    line_index=line_index,
                    raw_line=stripped,
                    reason="inline-section-phrase",
                )

    return None, None


def build_section_record(
    paper_record: PaperRecord,
    section_index: int,
    label: str,
    canonical_label: str,
    heading: str,
    path: list[str],
    level: int,
    lines: list[str],
    pages: list[int],
) -> Optional[SectionRecord]:
    text = "\n".join(line for line in lines if line).strip()
    if not text:
        return None

    page_values = [page for page in pages if page is not None]
    slug = re.sub(r"[^a-z0-9]+", "-", "-".join(path).lower()).strip("-") or "body"
    return SectionRecord(
        section_id=f"{paper_record.paper_id}:section:{section_index}:{slug}",
        label=label,
        canonical_label=canonical_label,
        heading=heading,
        path=path,
        level=level,
        order=SECTION_ORDER.get(canonical_label, SECTION_ORDER["Body"]),
        text=text,
        page_start=min(page_values) if page_values else None,
        page_end=max(page_values) if page_values else None,
    )


def analyze_document_structure(
    paper_record: PaperRecord,
    pages: list[tuple[Optional[int], str]],
) -> DocumentStructureReport:
    report = DocumentStructureReport()
    current_label = "Body"
    current_canonical_label = "Body"
    current_heading = "Front Matter"
    current_path = [current_heading]
    current_level = 1
    current_lines: list[str] = []
    current_pages: list[int] = []
    active_top_level_label = "Body"
    content_started = False
    inside_table_of_contents = False

    for page_number, page_text in pages:
        normalized_page = normalize_page_text(page_text)
        if not normalized_page:
            continue

        page_lines = [line.strip() for line in normalized_page.splitlines() if line.strip()]
        for line_index, line in enumerate(page_lines):
            if looks_like_table_of_contents_marker(line):
                inside_table_of_contents = True
                report.rejected_candidates.append(
                    RejectedHeadingCandidate(
                        page_number=page_number,
                        line_index=line_index,
                        raw_line=line,
                        reason="table-of-contents-marker",
                    )
                )
                continue

            if inside_table_of_contents and looks_like_table_of_contents_entry(line):
                report.rejected_candidates.append(
                    RejectedHeadingCandidate(
                        page_number=page_number,
                        line_index=line_index,
                        raw_line=line,
                        reason="table-of-contents-entry",
                    )
                )
                continue

            if looks_like_figure_or_table_heading(line):
                report.rejected_candidates.append(
                    RejectedHeadingCandidate(
                        page_number=page_number,
                        line_index=line_index,
                        raw_line=line,
                        reason="figure-table-label",
                    )
                )
                continue

            detected_heading, rejected = detect_section_heading(line, page_number, line_index)
            if rejected:
                report.rejected_candidates.append(rejected)

            if not content_started and looks_like_front_matter_metadata_line(line):
                report.rejected_candidates.append(
                    RejectedHeadingCandidate(
                        page_number=page_number,
                        line_index=line_index,
                        raw_line=line,
                        reason="front-matter-metadata",
                    )
                )
                continue
            if detected_heading:
                if inside_table_of_contents and detected_heading.canonical_label in CORE_SECTION_LABELS:
                    inside_table_of_contents = False

                if not content_started and detected_heading.canonical_label is None:
                    report.rejected_candidates.append(
                        RejectedHeadingCandidate(
                            page_number=page_number,
                            line_index=line_index,
                            raw_line=line,
                            reason="front-matter-heading",
                        )
                    )
                    continue

                section = build_section_record(
                    paper_record,
                    len(report.sections),
                    current_label,
                    current_canonical_label,
                    current_heading,
                    current_path,
                    current_level,
                    current_lines,
                    current_pages,
                )
                if section:
                    report.sections.append(section)

                report.headings.append(detected_heading)
                current_lines = []
                current_pages = [page_number] if page_number is not None else []

                if detected_heading.level <= 1:
                    current_path = [detected_heading.heading_title]
                else:
                    current_path = current_path[: detected_heading.level - 1] + [detected_heading.heading_title]

                if detected_heading.level == 1:
                    active_top_level_label = detected_heading.canonical_label or detected_heading.heading_title

                current_label = detected_heading.canonical_label or active_top_level_label
                current_canonical_label = current_label
                current_heading = detected_heading.heading_title
                current_level = detected_heading.level

                if detected_heading.canonical_label in CORE_SECTION_LABELS:
                    content_started = True

                if detected_heading.inline_text:
                    current_lines.append(detected_heading.inline_text)
                continue

            if page_number is not None and (not current_pages or current_pages[-1] != page_number):
                current_pages.append(page_number)

            if not content_started:
                continue

            current_lines.append(line)

    final_section = build_section_record(
        paper_record,
        len(report.sections),
        current_label,
        current_canonical_label,
        current_heading,
        current_path,
        current_level,
        current_lines,
        current_pages,
    )
    if final_section:
        report.sections.append(final_section)

    if not report.sections and paper_record.parsed_text:
        fallback = build_section_record(
            paper_record,
            0,
            "Body",
            "Body",
            "Body",
            ["Body"],
            1,
            paper_record.parsed_text.splitlines(),
            [page for page, _ in pages if page is not None],
        )
        if fallback:
            report.sections = [fallback]

    merged_sections: list[SectionRecord] = []
    for section in report.sections:
        if merged_sections and merged_sections[-1].path == section.path:
            merged_sections[-1].text = f"{merged_sections[-1].text}\n{section.text}".strip()
            merged_sections[-1].page_end = section.page_end or merged_sections[-1].page_end
            continue
        merged_sections.append(section)

    for index, section in enumerate(merged_sections):
        slug = re.sub(r"[^a-z0-9]+", "-", "-".join(section.path).lower()).strip("-") or "body"
        section.section_id = f"{paper_record.paper_id}:section:{index}:{slug}"

    report.sections = merged_sections
    return report


def extract_sections_from_pages(
    paper_record: PaperRecord,
    pages: list[tuple[Optional[int], str]],
) -> list[SectionRecord]:
    return analyze_document_structure(paper_record, pages).sections


def create_passage_chunks(section: SectionRecord) -> list[PassageChunkRecord]:
    chunk_texts = [chunk.strip() for chunk in PASSAGE_SPLITTER.split_text(section.text) if chunk.strip()]
    return [
        PassageChunkRecord(
            chunk_id=f"{section.section_id}:chunk:{chunk_index}",
            section_id=section.section_id,
            text=chunk_text,
            chunk_index=chunk_index,
            page_start=section.page_start,
            page_end=section.page_end,
        )
        for chunk_index, chunk_text in enumerate(chunk_texts)
    ]
