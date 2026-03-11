# Document Structure Review Workflow

This workflow is for improving heading and subsection detection without
turning the repository into a dump of PDFs and one-off debug artifacts.

## Goals

- build a small committed gold set for exact parser assertions
- maintain a larger local review corpus for broader inspection
- promote only high-value failures into committed fixtures

## Two-Tier Corpus Strategy

### Gold Fixtures

Committed under `tests/fixtures/document_structure/`.

Use these for:

- exact expected headings
- expected section labels
- subsection levels and paths
- false-positive guardrails
- page-span behavior

Recommended size: `8-12` real-paper-derived fixtures.

### Local Review Corpus

Kept outside git as local PDFs plus generated reports under
`debug_output/document_structure_review/`.

Use this for:

- broader layout coverage
- rapid parser triage
- discovering new failure modes
- deciding which papers deserve promotion into the gold set

Recommended size: `25-40` papers.

## How To Choose Candidate Papers

Do not choose by topic. Choose by structural diversity.

Prioritize papers that add a new parsing condition:

- single-column academic layout
- two-column layout
- numbered sections (`1`, `1.1`, `2.3`)
- Roman numeral sections
- all-caps headings
- inline abstract headings
- appendix-heavy or reference-heavy endings
- noisy OCR or weak extraction quality
- deep subsection nesting
- inconsistent heading formatting

Avoid over-sampling one publisher template.

## Selection Rubric

Score each candidate informally across these dimensions:

- heading style variety
- subsection depth
- extraction cleanliness
- column complexity
- appendix/reference noise
- repeated heading ambiguity
- whether it reveals a failure mode not already covered

Promote a paper into a gold fixture if it does at least one of these:

- exposes a new parser failure mode
- represents a common layout family in your library
- provides a clean regression target after a parser fix

## Review Commands

Run the committed parser fixtures:

```bash
pytest -q
```

Inspect a single paper or fixture:

```bash
python tools/debug_document_structure.py --file path/to/paper.pdf
python tools/debug_document_structure.py --fixture tests/fixtures/document_structure/numbered_subsections.json
python tools/debug_document_structure.py --fixture tests/fixtures/document_structure/numbered_subsections.json --html-out debug_output/fixture-report.html
```

Batch-review a local corpus:

```bash
python tools/review_document_structure_batch.py --input-dir path/to/pdf/corpus --recursive
```

Select a structurally diverse recommendation set directly from the Zotero library:

```bash
python tools/select_document_structure_candidates.py --max-per-collection 3 --recommendation-count 10
```

This writes:

- `debug_output/document_structure_review/summary.json`
- `debug_output/document_structure_review/index.html`
- `debug_output/document_structure_review/reports/*.json`
- `debug_output/document_structure_review/reports/*.html`

## Promotion Workflow

1. Run the batch review tool over a local candidate corpus.
2. Inspect `summary.json` for structural diversity and suspicious cases.
3. Open a few per-paper reports with unusual tags or bad section counts.
4. Pick the highest-value failures.
5. Convert those papers into minimal committed text fixtures.
6. Add exact expectations to the parser tests.

Alternative starting point:

1. Run the Zotero candidate selector.
2. Review the recommended papers and their tags.
3. Use those papers as the first promotion shortlist.

## Heuristic Tags In Batch Review

The batch review tool automatically tags papers when it sees signs such as:

- `numbered-headings`
- `roman-headings`
- `uppercase-headings`
- `inline-abstract`
- `subsections`
- `multi-page-sections`
- `repeated-section-labels`
- `rejected-heading-candidates`

These tags are not quality scores. They are triage hints.

## Repository Hygiene

- commit text fixtures, not large PDF corpora
- keep local review outputs under `debug_output/`
- keep the gold set small and high-signal
- promote failures intentionally rather than accumulating samples blindly
