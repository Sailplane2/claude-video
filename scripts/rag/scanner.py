"""Corpus scanner: discover video notes, validate structure, compute hashes, produce VideoRecords."""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import hash_text, videos_dir, index_path

# ── Data types ─────────────────────────────────────────────────────────────

@dataclass
class VideoRecord:
    video_id: str
    folder: str
    folder_path: Path
    summary_path: Path
    frontmatter: dict
    body: str
    content_hash: str
    status: str
    has_transcript: bool = False
    has_frame_descriptions: bool = False


@dataclass
class ValidationIssue:
    severity: str       # "error", "warning", "info"
    category: str       # "orphan_dir", "orphan_entry", "missing_summary", "bad_yaml", "missing_transcript"
    description: str
    path: str


@dataclass
class ScannerResult:
    videos: list[VideoRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── YAML frontmatter parser ────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text. Returns (frontmatter_dict, body)."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    fm_text = parts[1].strip()
    body = parts[2].strip()

    fm = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            fm[key] = value
    return fm, body


def _parse_frontmatter_full(text: str) -> tuple[dict, str]:
    """Full frontmatter parser handling lists and nested values."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    fm_text = parts[1].strip()
    body = parts[2].strip()

    fm = {}
    current_key = None
    current_list = []

    for line in fm_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item
        if stripped.startswith("- "):
            if current_key is not None:
                current_list.append(stripped[2:].strip().strip('"').strip("'"))
            continue

        # Save pending list
        if current_key is not None and current_list:
            fm[current_key] = current_list
            current_list = []
            current_key = None

        # Key: value
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value == "":
                current_key = key
                current_list = []
            else:
                fm[key] = value
                current_key = None

    # Save final pending list
    if current_key is not None and current_list:
        fm[current_key] = current_list

    return fm, body


# ── Scanner ────────────────────────────────────────────────────────────────

def _find_summary_file(folder_path: Path) -> Path | None:
    """Find the summary markdown file in a video folder."""
    for md in sorted(folder_path.glob("*.md")):
        content = md.read_text(encoding="utf-8", errors="replace")
        if content.startswith("---") and "video_id:" in content[:500]:
            return md
    # Fallback: any .md file
    md_files = sorted(folder_path.glob("*.md"))
    return md_files[0] if md_files else None


def _extract_video_id(folder_name: str, frontmatter: dict) -> str:
    """Extract video_id from frontmatter or folder name."""
    if "video_id" in frontmatter and frontmatter["video_id"]:
        return frontmatter["video_id"]
    # Extract from folder name: <video_id>_<title>
    parts = folder_name.split("_", 1)
    return parts[0] if parts else folder_name


def scan(videos_dir_override: Path | None = None) -> ScannerResult:
    """Scan the videos directory and return a validated ScannerResult."""
    vdir = videos_dir_override or videos_dir()
    result = ScannerResult()

    if not vdir.exists():
        result.issues.append(ValidationIssue(
            "error", "missing_dir",
            f"Videos directory not found: {vdir}", str(vdir)
        ))
        return result

    # Load index.json
    index_data = {}
    idx_path = vdir / "index.json"
    if idx_path.exists():
        try:
            raw = json.loads(idx_path.read_text())
            for v in raw.get("videos", []):
                vid = v.get("video_id", "")
                if vid:
                    index_data[vid] = v
        except (json.JSONDecodeError, KeyError) as e:
            result.issues.append(ValidationIssue(
                "warning", "bad_index",
                f"index.json parse error: {e}", str(idx_path)
            ))

    # Scan subdirectories
    indexed_in_index = set(index_data.keys())
    found_on_disk = set()

    for entry in sorted(vdir.iterdir()):
        if not entry.is_dir():
            continue

        folder_name = entry.name
        summary_path = _find_summary_file(entry)

        if summary_path is None:
            result.issues.append(ValidationIssue(
                "warning", "missing_summary",
                f"No summary.md in {folder_name}", str(entry)
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

        video_id = _extract_video_id(folder_name, fm)
        found_on_disk.add(video_id)

        # Check for optional files
        has_transcript = (entry / "transcript.txt").exists()
        has_frame_descriptions = (entry / "frame_descriptions.txt").exists()

        # Compute content hash from body only (frontmatter excluded)
        chunker_version = "v1.0"
        content_hash = hash_text(body + chunker_version)

        status = fm.get("status", "active")

        result.videos.append(VideoRecord(
            video_id=video_id,
            folder=folder_name,
            folder_path=entry,
            summary_path=summary_path,
            frontmatter=fm,
            body=body,
            content_hash=content_hash,
            status=status,
            has_transcript=has_transcript,
            has_frame_descriptions=has_frame_descriptions,
        ))

    # Cross-validate
    for vid in found_on_disk - indexed_in_index:
        result.issues.append(ValidationIssue(
            "info", "orphan_entry",
            f"Video on disk but not in index.json: {vid}", vid
        ))

    for vid in indexed_in_index - found_on_disk:
        entry = index_data[vid]
        result.issues.append(ValidationIssue(
            "info", "orphan_dir",
            f"Video in index.json but no folder on disk: {vid} ({entry.get('title', 'unknown')})",
            vid
        ))

    # Build stats
    statuses = {}
    for v in result.videos:
        statuses[v.status] = statuses.get(v.status, 0) + 1

    result.stats = {
        "total": len(result.videos),
        "by_status": statuses,
        "issues": len(result.issues),
        "indexed_in_index": len(indexed_in_index),
        "found_on_disk": len(found_on_disk),
    }

    return result


def get_video_by_id(video_id: str) -> VideoRecord | None:
    """Look up a single video by ID. Returns None if not found."""
    result = scan()
    for v in result.videos:
        if v.video_id == video_id:
            return v
    return None
