"""Document scanner: discover extracted PDF markdown, validate, compute hashes.

Mirrors scanner.py, but for <vault>/documents/ instead of videos. Documents are
optional (the folder may not exist yet), so an empty result is not an error.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import ValidationIssue, _parse_frontmatter_full
from .utils import hash_text, documents_dir

CHUNKER_VERSION = "v1.0"  # matches scanner.py's content-hash scheme


@dataclass
class DocumentRecord:
    document_id: str
    folder: str
    folder_path: Path
    summary_path: Path
    frontmatter: dict
    body: str
    content_hash: str
    status: str
    page_count: int


@dataclass
class DocumentScannerResult:
    documents: list[DocumentRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _safe_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _find_summary_file(folder_path: Path) -> Path | None:
    """Find the markdown file carrying document_id: frontmatter in a folder."""
    for md in sorted(folder_path.glob("*.md")):
        content = md.read_text(encoding="utf-8", errors="replace")
        if content.startswith("---") and "document_id:" in content[:500]:
            return md
    md_files = sorted(folder_path.glob("*.md"))
    return md_files[0] if md_files else None


def _extract_document_id(folder_name: str, frontmatter: dict) -> str:
    if frontmatter.get("document_id"):
        return frontmatter["document_id"]
    return folder_name.split("_", 1)[0]


def scan(documents_dir_override: Path | None = None) -> DocumentScannerResult:
    """Scan the documents directory and return a validated result."""
    ddir = documents_dir_override or documents_dir()
    result = DocumentScannerResult()

    if not ddir.exists():
        return result  # documents are optional

    # Load documents/index.json (cross-validation manifest)
    index_data = {}
    idx_path = ddir / "index.json"
    if idx_path.exists():
        try:
            raw = json.loads(idx_path.read_text())
            for d in raw.get("documents", []):
                did = d.get("document_id", "")
                if did:
                    index_data[did] = d
        except (json.JSONDecodeError, KeyError) as e:
            result.issues.append(ValidationIssue(
                "warning", "bad_index",
                f"documents/index.json parse error: {e}", str(idx_path)
            ))

    indexed_in_index = set(index_data.keys())
    found_on_disk = set()

    for entry in sorted(ddir.iterdir()):
        if not entry.is_dir():
            continue

        summary_path = _find_summary_file(entry)
        if summary_path is None:
            result.issues.append(ValidationIssue(
                "warning", "missing_summary",
                f"No .md with document_id in {entry.name}", str(entry)
            ))
            continue

        try:
            content = summary_path.read_text(encoding="utf-8", errors="replace")
            fm, body = _parse_frontmatter_full(content)
        except Exception as e:
            result.issues.append(ValidationIssue(
                "warning", "bad_yaml",
                f"Failed to parse frontmatter in {summary_path.name}: {e}",
                str(summary_path)
            ))
            continue

        document_id = _extract_document_id(entry.name, fm)
        found_on_disk.add(document_id)

        content_hash = hash_text(body + CHUNKER_VERSION)

        result.documents.append(DocumentRecord(
            document_id=document_id,
            folder=entry.name,
            folder_path=entry,
            summary_path=summary_path,
            frontmatter=fm,
            body=body,
            content_hash=content_hash,
            status=fm.get("status", "active"),
            page_count=_safe_int(fm.get("page_count"), 0),
        ))

    # Cross-validate
    for did in found_on_disk - indexed_in_index:
        result.issues.append(ValidationIssue(
            "info", "orphan_entry",
            f"Document on disk but not in index.json: {did}", did
        ))
    for did in indexed_in_index - found_on_disk:
        result.issues.append(ValidationIssue(
            "info", "orphan_dir",
            f"Document in index.json but no folder on disk: {did}", did
        ))

    statuses = {}
    for d in result.documents:
        statuses[d.status] = statuses.get(d.status, 0) + 1

    result.stats = {
        "total": len(result.documents),
        "by_status": statuses,
        "issues": len(result.issues),
        "indexed_in_index": len(indexed_in_index),
        "found_on_disk": len(found_on_disk),
    }

    return result
