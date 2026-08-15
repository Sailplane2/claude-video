#!/usr/bin/env python3
"""Extract text from PDFs into the vault's documents/ folder (source of truth).

Parallel to watch.py for videos: the PDF stays OUT of the vault; its extracted
text is written as markdown into <vault>/documents/{id}_{title}/{title}.md,
which the RAG document scanner then indexes. Rebuilds documents/index.json.

Usage:
    python3 document_ingest.py                 # ingest every PDF in the source dir
    python3 document_ingest.py path/to/a.pdf    # ingest a single PDF
    python3 document_ingest.py path/to/dir      # ingest every PDF in a directory
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def vault_root() -> Path:
    return Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/michael-brain"


def documents_dir() -> Path:
    return vault_root() / "documents"


def docs_source_dir() -> Path:
    return Path.home() / "claude-video" / "docs" / "source"


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60] or "document"


def _quote(s: str) -> str:
    """Escape a string for double-quoted YAML frontmatter."""
    return (s or "").replace("\\", "\\\\").replace('"', "'")


def _pdf_date(doc) -> str:
    meta = doc.metadata or {}
    created = meta.get("creationDate") or meta.get("CreationDate")
    if created:
        m = re.search(r"D:(\d{4})(\d{2})(\d{2})", str(created))
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def _file_date(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return ""


def extract_pdf(pdf_path: Path) -> dict:
    """Extract text from a PDF and return a structured dict (no vault writes)."""
    import pymupdf

    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
        body = "\f".join(pages)
        document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or pdf_path.stem
        author = (meta.get("author") or "").strip()
        date = _pdf_date(doc) or _file_date(pdf_path)
        return {
            "document_id": document_id,
            "title": title,
            "author": author,
            "date": date,
            "page_count": doc.page_count,
            "body": body,
            "source": str(pdf_path),
        }
    finally:
        doc.close()


def write_document(info: dict) -> Path | None:
    """Write the extracted markdown into the vault. Returns the .md path, or None if it already exists."""
    ddir = documents_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    folder = ddir / f"{info['document_id']}_{_safe_name(info['title'])}"
    folder.mkdir(parents=True, exist_ok=True)
    note_path = folder / f"{_safe_name(info['title'])}.md"

    if note_path.exists():
        return None  # idempotent: don't re-extract

    fm_title = _quote(info["title"])
    fm_author = _quote(info["author"])
    fm_source = _quote(info["source"])
    frontmatter = (
        "---\n"
        f'title: "{fm_title}"\n'
        "aliases: []\n"
        f'source: "{fm_source}"\n'
        f'document_id: "{info["document_id"]}"\n'
        f'date: "{info["date"]}"\n'
        f'author: "{fm_author}"\n'
        f"page_count: {info['page_count']}\n"
        "tags: []\n"
        "status: active\n"
        "---\n\n"
        f"# {info['title']}\n\n"
        f"{info['body']}\n"
    )
    note_path.write_text(frontmatter, encoding="utf-8")
    return note_path


# ── index.json rebuild ─────────────────────────────────────────────────────

def rebuild_index() -> int:
    """Rebuild documents/index.json from the extracted markdown on disk."""
    ddir = documents_dir()
    entries = []
    if not ddir.exists():
        return 0

    for folder in sorted(ddir.iterdir()):
        if not folder.is_dir():
            continue
        summary = None
        for md in sorted(folder.glob("*.md")):
            content = md.read_text(encoding="utf-8", errors="replace")
            if content.startswith("---") and "document_id:" in content[:500]:
                summary = md
                break
        if summary is None:
            continue

        text = summary.read_text(encoding="utf-8", errors="replace")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            continue
        fm_block = m.group(1)

        def field(name, default=""):
            r = re.search(rf"^{name}:\s*[\"']?(.+?)[\"']?\s*$", fm_block, re.MULTILINE)
            return r.group(1).strip() if r else default

        def get_tags(fm_block):
            r = re.search(r"^tags:\s*\[(.+)\]$", fm_block, re.MULTILINE)
            if r:
                raw = r.group(1)
                if raw.strip() == "":
                    return []
                return [t.strip().strip("\"'") for t in raw.split(",") if t.strip()]
            return []

        pc = field("page_count")
        entries.append({
            "folder": folder.name,
            "document_id": field("document_id"),
            "title": field("title", folder.name),
            "author": field("author"),
            "source": field("source"),
            "date": field("date"),
            "page_count": int(pc) if pc.isdigit() else 0,
            "tags": get_tags(fm_block),
            "status": field("status", "active"),
            "summary_path": str(summary),
        })

    (ddir / "index.json").write_text(json.dumps({"documents": entries}, indent=2))
    return len(entries)


# ── CLI ────────────────────────────────────────────────────────────────────

def _find_pdfs(paths: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
    return pdfs


def main(argv: list[str]) -> int:
    source = docs_source_dir()
    source.mkdir(parents=True, exist_ok=True)

    if argv:
        pdfs = _find_pdfs([Path(a) for a in argv])
    else:
        pdfs = sorted(source.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found (source dir: {source}).")
        print("Drop PDFs into ~/claude-video/docs/source/ or pass a path/directory.")
        return 1

    ingested = skipped = errors = 0
    for pdf in pdfs:
        try:
            info = extract_pdf(pdf)
        except Exception as e:
            print(f"  ERROR {pdf.name}: {e}", file=sys.stderr)
            errors += 1
            continue

        note_path = write_document(info)
        if note_path is None:
            print(f"  SKIP {pdf.name} (already extracted)")
            skipped += 1
        else:
            print(f"  INGEST {pdf.name} → {note_path.parent.name}/")
            ingested += 1

    n = rebuild_index()
    print(f"\nDone: {ingested} ingested, {skipped} skipped, {errors} error(s) — "
          f"documents/index.json updated ({n} document(s))")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
