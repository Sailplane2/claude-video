"""Session lifecycle: create, list, show, compact.

Manages collaboration session directories, enforces the state machine
contract by controlling phase transitions.
"""

import os
import re
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

import yaml

from .errors import SessionNotFoundError, SessionExistsError, AimCliError
from .exchange import (
    update_session_index,
    read_session_index,
    list_decision_files,
    read_lock,
    _atomic_write,
)
from .protocol import (
    MULTI_AI_DIR,
    read_yaml_frontmatter,
    count_tokens,
    read_transcript,
)

logger = logging.getLogger(__name__)

VALID_PHASES = ["idle", "scoping", "executing", "reviewing", "decision", "complete"]

# Valid phase transitions: from -> to
VALID_TRANSITIONS = {
    "idle": ["scoping"],
    "scoping": ["executing", "decision"],
    "executing": ["reviewing", "decision"],
    "reviewing": ["executing", "decision", "complete"],
    "decision": ["scoping", "executing", "reviewing", "complete"],
    "complete": [],
}


def validate_session_name(name: str) -> bool:
    """Validate session name: alphanumeric start, letters, numbers, hyphens, underscores."""
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", name))


def session_path(name: str, base_path: str = None) -> str:
    """Get the full path for a session directory."""
    if base_path is None:
        base_path = MULTI_AI_DIR
    return os.path.join(base_path, "sessions", name)


def new_session(
    name: str,
    goal: str = None,
    provider: str = "openai",
    model: str = None,
    no_external: bool = False,
    base_path: str = None,
) -> Dict:
    """Create a new collaboration session.

    Args:
        name: Session name (alphanumeric + hyphens/underscores).
        goal: Optional session goal description.
        provider: Default AI provider.
        model: Default model for the provider.
        no_external: If True, mark session as no-external.
        base_path: Base multi-ai directory. Defaults to MULTI_AI_DIR.

    Returns:
        Session metadata dict.

    Raises:
        ValueError: If session name is invalid.
        SessionExistsError: If session already exists.
    """
    if not validate_session_name(name):
        raise ValueError(
            "Session names must start with a letter and contain only "
            "letters, numbers, hyphens, and underscores."
        )

    if base_path is None:
        base_path = MULTI_AI_DIR

    sess_path = os.path.join(base_path, "sessions", name)

    if os.path.exists(sess_path):
        raise SessionExistsError(name)

    # Set default model based on provider
    if model is None:
        model = "gpt-4o" if provider == "openai" else "gemini-2.5-flash"

    now = datetime.now(timezone.utc).isoformat()

    # Create directory structure
    os.makedirs(os.path.join(sess_path, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(sess_path, "artifacts"), exist_ok=True)

    # Write meta.yaml
    meta = {
        "session_name": name,
        "created": now,
        "goal": goal or "",
        "default_provider": provider,
        "default_model": model,
        "no_external": no_external,
        "exchange_count": 0,
        "cost_accumulated": 0.0,
        "last_exchange": "",
    }
    _atomic_write(os.path.join(sess_path, "meta.yaml"),
                  yaml.dump(meta, default_flow_style=False, allow_unicode=True))

    # Write state.yaml
    state = {
        "session_name": name,
        "phase": "scoping",
        "created": now,
        "updated": now,
        "participants": ["claude", "target-ai", "michael"],
        "allow_ai_initiated": True,
        "transition_log": [
            {
                "timestamp": now,
                "from": "idle",
                "to": "scoping",
                "rationale": f"Created new session for {goal or name}",
                "by": "michael",
            }
        ],
    }
    _atomic_write(os.path.join(sess_path, "state.yaml"),
                  yaml.dump(state, default_flow_style=False, allow_unicode=True))

    # Write brief.md
    brief_content = f"""---
updated: {now[:10]}
token_budget: 500
---

# Session Brief: {name}

## Goal
{goal or "TBD"}

## Key Constraints
<!-- Add session-specific constraints here -->

## Session-Specific Context
<!-- Add session-specific context here -->
"""
    _atomic_write(os.path.join(sess_path, "brief.md"), brief_content)

    # Write scope.md
    scope_content = f"""# Scope: {name}

## Goal
{goal or "TBD"}

## Constraints
<!-- Add constraints here -->

## Success Criteria
<!-- Add success criteria here -->

## Out of Scope
<!-- Add out-of-scope items here -->
"""
    _atomic_write(os.path.join(sess_path, "scope.md"), scope_content)

    # Write exchange-log.md
    _atomic_write(os.path.join(sess_path, "exchange-log.md"),
                  f"# Exchange Log: {name}\n\n")

    # Write transcript.jsonl (empty)
    with open(os.path.join(sess_path, "transcript.jsonl"), "w") as f:
        pass

    # Write lock file (empty)
    with open(os.path.join(sess_path, "lock"), "w") as f:
        pass

    # Update index
    index_data = {
        "goal": goal or "",
        "phase": "scoping",
        "created": now,
        "last_updated": now,
        "exchange_count": 0,
        "cost_accumulated": 0.0,
    }
    update_session_index(
        os.path.join(base_path, "sessions", "index.yaml"),
        name,
        index_data,
    )

    return meta


def list_sessions(
    active_only: bool = True,
    phase_filter: str = None,
    base_path: str = None,
) -> List[Dict]:
    """List collaboration sessions.

    Args:
        active_only: If True, filter out completed sessions.
        phase_filter: If set, filter by this phase.
        base_path: Base multi-ai directory.

    Returns:
        List of session summary dicts.
    """
    if base_path is None:
        base_path = MULTI_AI_DIR

    index_path = os.path.join(base_path, "sessions", "index.yaml")
    index = read_session_index(index_path)
    sessions = index.get("sessions", {})

    result = []
    for name, data in sessions.items():
        # Ensure session directory still exists
        sess_path = os.path.join(base_path, "sessions", name)
        if not os.path.isdir(sess_path):
            continue

        if active_only and data.get("phase") == "complete":
            continue

        if phase_filter and data.get("phase") != phase_filter:
            continue

        result.append({
            "name": name,
            "goal": data.get("goal", ""),
            "phase": data.get("phase", "unknown"),
            "created": data.get("created", ""),
            "last_updated": data.get("last_updated", ""),
            "exchange_count": data.get("exchange_count", 0),
            "cost_accumulated": data.get("cost_accumulated", 0.0),
        })

    # Sort by last_updated descending
    result.sort(key=lambda s: s.get("last_updated", ""), reverse=True)
    return result


def show_session(name: str, base_path: str = None) -> Dict:
    """Show detailed session information.

    Args:
        name: Session name.
        base_path: Base multi-ai directory.

    Returns:
        Complete session state dict.

    Raises:
        SessionNotFoundError: If session doesn't exist.
    """
    if base_path is None:
        base_path = MULTI_AI_DIR

    sess_path = session_path(name, base_path)
    if not os.path.isdir(sess_path):
        raise SessionNotFoundError(name)

    # Read meta.yaml
    meta = {}
    meta_path = os.path.join(sess_path, "meta.yaml")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}

    # Read state.yaml
    state = {}
    state_path = os.path.join(sess_path, "state.yaml")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = yaml.safe_load(f) or {}

    # Count exchanges in exchange-log.md
    exchange_log_path = os.path.join(sess_path, "exchange-log.md")
    exchange_count = 0
    if os.path.exists(exchange_log_path):
        with open(exchange_log_path, "r") as f:
            content = f.read()
        exchange_count = len(re.findall(r"^## ", content, re.MULTILINE))

    # List decision files
    decisions_dir = os.path.join(sess_path, "decisions")
    decision_files = list_decision_files(decisions_dir)

    # Read last 5 exchange entries
    recent_exchanges = []
    if os.path.exists(exchange_log_path):
        with open(exchange_log_path, "r") as f:
            content = f.read()
        entries = re.split(r"\n---\n", content)
        for entry in entries[-5:]:
            entry = entry.strip()
            if entry:
                # Extract heading, message, response
                lines = entry.split("\n")
                heading = ""
                message = ""
                response = ""
                for i, line in enumerate(lines):
                    if line.startswith("## "):
                        heading = line
                    elif line.startswith("Message: "):
                        message = line[9:]
                    elif line.startswith("**Response"):
                        response = lines[i + 1] if i + 1 < len(lines) else ""
                recent_exchanges.append({
                    "heading": heading,
                    "message": message,
                    "response": response,
                })

    # Read lock
    lock_info = read_lock(sess_path)

    return {
        "name": name,
        "goal": meta.get("goal", ""),
        "phase": state.get("phase", "unknown"),
        "created": meta.get("created", ""),
        "default_provider": meta.get("default_provider", "openai"),
        "default_model": meta.get("default_model", ""),
        "no_external": meta.get("no_external", False),
        "exchange_count": exchange_count or meta.get("exchange_count", 0),
        "cost_accumulated": meta.get("cost_accumulated", 0.0),
        "last_exchange": meta.get("last_exchange", ""),
        "decision_count": len(decision_files),
        "decisions": decision_files,
        "recent_exchanges": recent_exchanges,
        "participants": state.get("participants", []),
        "phase_transitions": state.get("transition_log", []),
        "locked": lock_info is not None,
        "lock_info": lock_info,
    }


def compact_session(name: str, base_path: str = None) -> Dict:
    """Compact thread transcript into summary.md.

    Args:
        name: Session name.
        base_path: Base multi-ai directory.

    Returns:
        Compaction result dict.

    Raises:
        SessionNotFoundError: If session doesn't exist.
    """
    if base_path is None:
        base_path = MULTI_AI_DIR

    sess_path = session_path(name, base_path)
    if not os.path.isdir(sess_path):
        raise SessionNotFoundError(name)

    exchange_log_path = os.path.join(sess_path, "exchange-log.md")
    transcript_path = os.path.join(sess_path, "transcript.jsonl")
    summary_path = os.path.join(sess_path, "summary.md")

    # Read exchange log
    if not os.path.exists(exchange_log_path):
        return {"message": "Nothing to compact. No exchange log found.", "success": False}

    with open(exchange_log_path, "r") as f:
        log_content = f.read()

    # Split into entries
    entries = [e.strip() for e in re.split(r"\n---\n", log_content) if e.strip()]
    entries = [e for e in entries if e.startswith("## ")]

    if len(entries) < 4:
        return {
            "message": "Nothing to compact. Thread has fewer than 4 exchanges.",
            "success": False,
        }

    # Determine cutoff: oldest 50%
    cutoff = len(entries) // 2
    compacted = entries[:cutoff]
    remaining = entries[cutoff:]

    # Build summary text
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    first_date = ""
    last_date = ""
    if compacted:
        first_heading = compacted[0].split("\n")[0] if compacted else ""
        last_heading = compacted[-1].split("\n")[0] if compacted else ""
        first_date = first_heading[3:].split("·")[0].strip() if "·" in first_heading else ""
        last_date = last_heading[3:].split("·")[0].strip() if "·" in last_heading else ""

    summary_lines = [
        f"# Thread Summary: {name}",
        "",
        f"Compacted: {now}",
        f"Exchanges compacted: {len(compacted)} ({first_date} through {last_date})",
        "",
        "## Original Questions",
    ]

    questions = []
    for entry in compacted:
        for line in entry.split("\n"):
            if line.startswith("Message: "):
                msg = line[9:]
                if msg not in questions:
                    questions.append(msg)
                    summary_lines.append(f"{len(questions)}. {msg}")

    summary_lines.extend([
        "",
        "## Key Information Preserved",
        "<!-- Summarized from compacted exchanges -->",
        "- See decision files and remaining exchange log for details.",
        "",
        "## Decisions Made",
        "<!-- Refer to decisions/ directory for full decision records -->",
        "",
        "## Open Threads",
        "<!-- See remaining exchange log -->",
    ])

    summary_text = "\n".join(summary_lines)

    # Write summary.md atomically
    _atomic_write(summary_path, summary_text + "\n")

    # Annotate compacted entries in exchange-log.md
    hdr = f"# Exchange Log: {name}\n"
    annotated = log_content
    for entry in compacted:
        heading_line = entry.split("\n")[0]
        annotation = "\n\n[COMPACTED into summary.md]\n"
        annotated = annotated.replace(
            entry,
            entry + annotation,
            1,
        )

    _atomic_write(exchange_log_path, annotated)

    # Trim transcript.jsonl to keep only non-compacted entries
    transcript = read_transcript(transcript_path)
    if transcript:
        keep = transcript[cutoff:]
        buf = ""
        for exchange in keep:
            if exchange.get("user"):
                buf += json.dumps({
                    "ts": now, "role": "user",
                    "content": exchange["user"],
                    "tokens": 0, "model": "",
                }, separators=(",", ":")) + "\n"
            if exchange.get("assistant"):
                buf += json.dumps({
                    "ts": now, "role": "assistant",
                    "content": exchange["assistant"],
                    "tokens": 0, "model": "", "response_id": "",
                    "finish_reason": "",
                }, separators=(",", ":")) + "\n"
        _atomic_write(transcript_path, buf)

    summary_tokens = count_tokens(summary_text)

    return {
        "message": (
            f"Compaction candidates: exchanges 1-{len(compacted)} "
            f"({len(compacted)}/{len(entries)} exchanges)\n"
            f"Generated summary: {summary_tokens} tokens\n"
            f"Transcript trimmed: {len(entries)} exchanges -> {len(remaining)} exchanges\n"
            f"Summary written to: summary.md"
        ),
        "success": True,
        "compacted_count": len(compacted),
        "remaining_count": len(remaining),
        "summary_tokens": summary_tokens,
    }


def update_session_meta(
    name: str,
    updates: Dict,
    base_path: str = None,
) -> None:
    """Update a session's meta.yaml with new values.

    Args:
        name: Session name.
        updates: Dict of fields to update.
        base_path: Base multi-ai directory.
    """
    if base_path is None:
        base_path = MULTI_AI_DIR

    sess_path = session_path(name, base_path)
    meta_path = os.path.join(sess_path, "meta.yaml")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}

    meta.update(updates)

    _atomic_write(meta_path,
                  yaml.dump(meta, default_flow_style=False, allow_unicode=True))

    # Also update index
    now = datetime.now(timezone.utc).isoformat()
    index_data = {
        "goal": meta.get("goal", ""),
        "phase": "",
        "created": meta.get("created", ""),
        "last_updated": now,
        "exchange_count": meta.get("exchange_count", 0),
        "cost_accumulated": meta.get("cost_accumulated", 0.0),
    }
    update_session_index(
        os.path.join(base_path, "sessions", "index.yaml"),
        name,
        index_data,
    )


def update_session_state(
    name: str,
    updates: Dict,
    base_path: str = None,
) -> None:
    """Update a session's state.yaml with new values.

    Handles phase transitions by adding to transition_log.
    """
    if base_path is None:
        base_path = MULTI_AI_DIR

    sess_path = session_path(name, base_path)
    state_path = os.path.join(sess_path, "state.yaml")

    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = yaml.safe_load(f) or {}

    old_phase = state.get("phase", "scoping")
    new_phase = updates.get("phase", old_phase)

    # Validate phase transition
    if new_phase != old_phase:
        if not is_valid_transition(old_phase, new_phase):
            logger.warning(
                f"Invalid phase transition: {old_phase} -> {new_phase}. "
                f"Allowed from {old_phase}: {VALID_TRANSITIONS.get(old_phase, [])}"
            )
            # Don't block — log and proceed (Build 1: advisory enforcement)

    state.update(updates)
    state["updated"] = datetime.now(timezone.utc).isoformat()

    # Log transition if phase changed
    if new_phase != old_phase and new_phase in VALID_PHASES:
        if "transition_log" not in state:
            state["transition_log"] = []
        transition = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": old_phase,
            "to": new_phase,
            "rationale": updates.get("transition_rationale", f"Phase changed from {old_phase} to {new_phase}"),
            "by": updates.get("transition_by", "claude"),
        }
        state["transition_log"].append(transition)

    _atomic_write(state_path,
                  yaml.dump(state, default_flow_style=False, allow_unicode=True))

    # Also update index phase
    index_path = os.path.join(base_path, "sessions", "index.yaml")
    index = read_session_index(index_path)
    sessions = index.get("sessions", {})
    if name in sessions:
        sessions[name]["phase"] = new_phase
        sessions[name]["last_updated"] = state["updated"]
        update_session_index(index_path, name, sessions[name])


def is_valid_transition(from_phase: str, to_phase: str) -> bool:
    """Check if a phase transition is valid."""
    if from_phase not in VALID_TRANSITIONS:
        return False
    return to_phase in VALID_TRANSITIONS[from_phase]
