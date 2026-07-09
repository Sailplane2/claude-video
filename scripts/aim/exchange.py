"""Exchange log, lockfile, decisions, and secret scanning.

Handles all file I/O for exchange logging, lock acquisition/release,
decision recording, and security scanning.
"""

import os
import re
import json
import time
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

import yaml

from .errors import LockActiveError, LockStaleError

logger = logging.getLogger(__name__)


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically: write to .tmp, then os.rename.

    On macOS within the same filesystem, os.rename is atomic — the
    destination either has the old content or the new content, never a
    partial write.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(content)
        os.rename(tmp_path, path)
    except Exception:
        # Clean up tmp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def acquire_lock(session_path: str, stale_timeout: int = 600) -> bool:
    """Acquire a lock for writing to a session.

    Args:
        session_path: Path to the session directory.
        stale_timeout: Seconds after which a lock is considered stale.

    Returns:
        True if lock acquired.

    Raises:
        LockActiveError: If lock is held by another process.
    """
    lock_path = os.path.join(session_path, "lock")
    now = datetime.now(timezone.utc).isoformat()

    # Build lock data before attempting atomic creation
    lock_data = {
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "acquired": now,
    }

    # Attempt atomic lock creation via O_CREAT|O_EXCL — this avoids the
    # TOCTOU race where two processes both see the lock as absent and
    # both write.  Only one process can create the file; the other gets
    # EEXIST and falls through to the staleness check.
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(lock_data, f)
        return True
    except FileExistsError:
        # Lock already exists — check staleness
        pass

    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Corrupt lock file, treat as stale
            logger.warning("Corrupt lock file detected, treating as stale.")
            data = {}

        acquired_str = data.get("acquired", "")
        if acquired_str:
            try:
                acquired_dt = datetime.fromisoformat(acquired_str)
                if acquired_dt.tzinfo is None:
                    acquired_dt = acquired_dt.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - acquired_dt).total_seconds()
                if age < stale_timeout:
                    raise LockActiveError(
                        os.path.basename(session_path) or "unknown",
                        data,
                    )
                # Stale lock — log warning and break it
                logger.warning(
                    f"Stale lock broken (acquired {acquired_str}, "
                    f"{age:.0f}s ago, exceeding {stale_timeout}s timeout)."
                )
            except (ValueError, TypeError):
                # Can't parse timestamp, treat as stale
                logger.warning("Lock has unparseable timestamp, treating as stale.")

    # Break stale lock by removing it, then retry atomic creation
    try:
        os.unlink(lock_path)
    except OSError:
        pass
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(lock_data, f)
        return True
    except FileExistsError:
        raise LockActiveError(
            os.path.basename(session_path) or "unknown",
            {"error": "Race: another process won the lock after stale break"},
        )


def release_lock(session_path: str) -> None:
    """Release the session lock."""
    lock_path = os.path.join(session_path, "lock")
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except (IOError, OSError) as e:
        logger.warning(f"Failed to remove lock file {lock_path}: {e}")


def read_lock(session_path: str) -> Optional[Dict]:
    """Read the current lock file contents, or None."""
    lock_path = os.path.join(session_path, "lock")
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def log_exchange(
    session_path: str,
    global_log_dir: str,
    exchange: Dict,
) -> str:
    """Write an exchange entry to session and global logs.

    Args:
        session_path: Path to the session directory.
        global_log_dir: Path to the global exchange log directory.
        exchange: Exchange data dict.

    Returns:
        The entry heading string.
    """
    ts = exchange.get("ts", datetime.now(timezone.utc).isoformat())
    from_ = exchange.get("from", "claude")
    to_ = exchange.get("to", "target-ai")
    phase = exchange.get("phase", "scoping")
    message = exchange.get("message", "")
    response_text = exchange.get("response_text", "")
    session_name = os.path.basename(session_path)

    # Format timestamp for display
    try:
        dt = datetime.fromisoformat(ts)
        display_ts = dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        display_ts = ts[:16] if len(ts) >= 16 else ts

    # Build JSON metadata for HTML comment
    meta = {
        "ts": ts,
        "from": from_,
        "to": to_,
        "phase": phase,
        "tokens_in": exchange.get("tokens_in", 0),
        "tokens_out": exchange.get("tokens_out", 0),
        "cost": exchange.get("cost", 0.0),
        "response_id": exchange.get("response_id", ""),
        "model": exchange.get("model", ""),
        "finish_reason": exchange.get("finish_reason", ""),
    }
    meta_json = json.dumps(meta, separators=(",", ":"))

    # Build markdown entry
    heading = f"## {display_ts} · {from_} → {to_} · {phase}"
    context = exchange.get("context_injected", {})
    context_lines = []
    if context:
        context_lines.append(f"Context injected: {json.dumps(context)}")
    method = exchange.get("method", "api")
    model = exchange.get("model", "unknown")
    context_lines.append(f"Method: {method} ({model})")

    error = exchange.get("error")
    if error:
        entry = (
            f"{heading}\n"
            f"<!-- {meta_json} -->\n"
            f"Message: {message}\n"
            + "\n".join(context_lines) + "\n\n"
            f"**FAILED:** {error}\n"
        )
    else:
        response_ts = exchange.get("response_ts", ts)
        entry = (
            f"{heading}\n"
            f"<!-- {meta_json} -->\n"
            f"Message: {message}\n"
            + "\n".join(context_lines) + "\n\n"
            f"**Response ({response_ts}):**\n"
            f"{response_text}\n"
        )

    # Markdown separator between entries
    separator = "\n---\n\n"

    # Write to session exchange log
    session_log_path = os.path.join(session_path, "exchange-log.md")
    try:
        with open(session_log_path, "a") as f:
            f.write(entry + separator)
    except (IOError, OSError) as e:
        logger.error(f"Failed to write session exchange log: {e}")

    # Write to global exchange log
    try:
        month_str = ts[:7] if len(ts) >= 7 else datetime.now().strftime("%Y-%m")
        global_log_path = os.path.join(global_log_dir, f"{month_str}.md")
        os.makedirs(global_log_dir, exist_ok=True)

        # Create file with header if new
        if not os.path.exists(global_log_path):
            try:
                month_name = datetime.strptime(month_str, "%Y-%m").strftime("%B %Y")
            except ValueError:
                month_name = month_str
            with open(global_log_path, "w") as f:
                f.write(f"# Exchange Log: {month_name}\n\n")

        global_heading = f"## {display_ts} · {session_name} · {from_} → {to_} · {phase}"
        message_preview = message[:200] if len(message) > 200 else message
        global_entry = (
            f"{global_heading}\n"
            f"<!-- {meta_json} -->\n"
            f"Message preview: {message_preview}\n"
        )
        with open(global_log_path, "a") as f:
            f.write(global_entry + separator)
    except (IOError, OSError) as e:
        logger.error(f"Failed to write global exchange log: {e}")

    # Write to transcript.jsonl
    transcript_path = os.path.join(session_path, "transcript.jsonl")
    try:
        # User message
        user_entry = json.dumps({
            "ts": ts,
            "role": "user",
            "content": message,
            "tokens": exchange.get("tokens_in", 0),
            "model": model,
        }, separators=(",", ":"))
        # Assistant message
        assistant_entry = json.dumps({
            "ts": exchange.get("response_ts", ts),
            "role": "assistant",
            "content": response_text,
            "tokens": exchange.get("tokens_out", 0),
            "model": model,
            "response_id": exchange.get("response_id", ""),
            "finish_reason": exchange.get("finish_reason", ""),
        }, separators=(",", ":"))
        with open(transcript_path, "a") as f:
            f.write(user_entry + "\n" + assistant_entry + "\n")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write transcript: {e}")

    return heading


def write_decision(session_path: str, decision: Dict) -> str:
    """Write a decision file to the session's decisions directory.

    Args:
        session_path: Path to the session directory.
        decision: Decision data dict.

    Returns:
        The filename of the created decision file.
    """
    date = decision.get("date", datetime.now().strftime("%Y-%m-%d"))
    question = decision.get("question", "")
    session_name = decision.get("session", os.path.basename(session_path))

    # Generate slug from question
    slug_raw = question[:40].lower()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug_raw).strip("-")
    if not slug:
        slug = "untitled"
    filename = f"{date}--{slug}.md"

    decisions_dir = os.path.join(session_path, "decisions")
    os.makedirs(decisions_dir, exist_ok=True)

    filepath = os.path.join(decisions_dir, filename)

    # Build YAML frontmatter
    frontmatter = {
        "date": date,
        "requested_by": decision.get("requested_by", "michael"),
        "session": session_name,
        "question": question,
        "alternatives": decision.get("alternatives"),
        "decision": decision.get("decision", ""),
        "rationale": decision.get("rationale", ""),
        "impact": decision.get("impact", ""),
        "superseded_by": decision.get("superseded_by"),
    }

    # Remove None values for cleaner output
    cleaned = {k: v for k, v in frontmatter.items() if v is not None}

    content = "---\n"
    content += yaml.dump(cleaned, default_flow_style=False, allow_unicode=True)
    content += "---\n"

    try:
        _atomic_write(filepath, content)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to write decision file {filepath}: {e}")

    return filename


SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key (sk-*)"),
    (r"sk-proj-[a-zA-Z0-9]{20,}", "OpenAI project API key (sk-proj-*)"),
    (r"sk-ant-[a-zA-Z0-9_-]{20,}", "Anthropic API key (sk-ant-*)"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key (AKIA*)"),
    (r"-----BEGIN [A-Z ]+-----", "Private key / certificate"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub PAT (ghp_*)"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth (gho_*)"),
    (r"ghu_[a-zA-Z0-9]{36}", "GitHub user token (ghu_*)"),
    (r"xox[baprs]-[0-9a-zA-Z-]{10,}", "Slack token (xox*)"),
    (r"eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{0,}", "JWT token"),
    (r"(?i)api[-_]?key[:=][\"']?[a-zA-Z0-9]{16,}", "Generic API key"),
    (r"(?i)secret[:=][\"']?[a-zA-Z0-9+/]{16,}", "Generic secret"),
]


def scan_secrets(text: str) -> List[Dict]:
    """Scan text for patterns matching known secret/key formats.

    Returns list of dicts: [{"pattern": "...", "match": "...", "position": N}]
    """
    matches = []
    for pattern, name in SECRET_PATTERNS:
        for m in re.finditer(pattern, text):
            matches.append({
                "pattern": name,
                "match": m.group()[:20] + "..." if len(m.group()) > 20 else m.group(),
                "position": m.start(),
            })
    return matches


def update_session_index(
    index_path: str,
    session_name: str,
    session_data: Dict,
) -> None:
    """Update the sessions index.yaml with session metadata.

    Args:
        index_path: Path to sessions/index.yaml.
        session_name: Session name.
        session_data: Dict with phase, goal, created, last_updated,
                     exchange_count, cost_accumulated fields.
    """
    # Read existing index using resilient reader
    index = read_session_index(index_path)
    if "sessions" not in index:
        index["sessions"] = {}

    index["sessions"][session_name] = {
        "goal": session_data.get("goal", ""),
        "phase": session_data.get("phase", "scoping"),
        "created": session_data.get("created", ""),
        "last_updated": session_data.get("last_updated", ""),
        "exchange_count": session_data.get("exchange_count", 0),
        "cost_accumulated": session_data.get("cost_accumulated", 0.0),
    }
    index["last_updated"] = session_data.get("last_updated", "")

    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        _atomic_write(index_path, yaml.dump(index, default_flow_style=False, allow_unicode=True))
    except (IOError, OSError) as e:
        logger.error(f"Failed to write index {index_path}: {e}")


def read_session_index(index_path: str) -> Dict:
    """Read and return the sessions index.

    Supports both the dict format (sessions: map) and legacy list format.
    """
    if not os.path.exists(index_path):
        return {"sessions": {}, "last_updated": ""}
    try:
        with open(index_path, "r") as f:
            content = f.read()
            if content.strip():
                data = yaml.safe_load(content) or {}
                # Handle legacy list format
                if isinstance(data, list):
                    sessions = {}
                    for entry in data:
                        name = entry.get("name", "unknown")
                        sessions[name] = {
                            "goal": entry.get("goal", ""),
                            "phase": entry.get("phase", "scoping"),
                            "created": entry.get("created", ""),
                            "last_updated": entry.get("last_updated", ""),
                            "exchange_count": entry.get("exchanges", entry.get("exchange_count", 0)),
                            "cost_accumulated": entry.get("cost_accumulated", 0.0),
                        }
                    return {"sessions": sessions, "last_updated": ""}
                return data
    except (yaml.YAMLError, IOError):
        pass
    return {"sessions": {}, "last_updated": ""}


def list_decision_files(decisions_dir: str) -> list:
    """List decision files in a session's decisions directory."""
    decisions_path = Path(decisions_dir)
    if not decisions_path.exists():
        return []
    return sorted([f.name for f in decisions_path.iterdir() if f.suffix == ".md"])
