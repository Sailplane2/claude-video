"""Context assembly, token budgeting, and metadata parsing.

Builds the message array for API calls by merging global brief, session brief,
thread history, and collaboration contract. Enforces token budgets.
Parses target AI response metadata.
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import yaml

from .errors import BudgetExceededError

logger = logging.getLogger(__name__)


MULTI_AI_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/michael-brain/"
    "ai-brain/context/multi-ai/"
)

DEFAULT_TOKEN_BUDGET = 6000
DEFAULT_MONTHLY_CAP = 100.0


def read_yaml_frontmatter(filepath: str) -> Tuple[dict, str]:
    """Read a file with YAML frontmatter (delimited by ---).

    Returns (frontmatter_dict, body_text).
    """
    if not os.path.exists(filepath):
        return {}, ""

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (IOError, OSError) as e:
        logger.warning(f"Cannot read {filepath}: {e}")
        return {}, ""

    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        # Malformed frontmatter, treat as body
        logger.warning(f"Malformed frontmatter in {filepath}")
        return {}, content

    yaml_str = parts[1].strip()
    body = parts[2].strip()

    try:
        frontmatter = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError as e:
        logger.warning(f"YAML parse error in {filepath}: {e}")
        return {}, content

    return frontmatter, body


def read_brief(path: str) -> str:
    """Read and return the body text of a brief.md file (strip frontmatter)."""
    _, body = read_yaml_frontmatter(path)
    return body


def get_phase_preamble(phase: str, contract_path: str = None) -> str:
    """Return the phase-specific instructions from the contract file.

    The contract file at {MULTI_AI_DIR}/contract.md contains sections like:
    ### SCOPING, ### EXECUTING, ### REVIEWING, ### DECISION

    Returns the full contract body with phase-specific section highlighted.
    """
    if contract_path is None:
        contract_path = os.path.join(MULTI_AI_DIR, "contract.md")

    contract_body = read_brief(contract_path)
    return contract_body


def get_phase_specific_instruction(phase: str, contract_path: str = None) -> str:
    """Extract the phase-specific instruction block from the contract."""
    if contract_path is None:
        contract_path = os.path.join(MULTI_AI_DIR, "contract.md")

    _, body = read_yaml_frontmatter(contract_path)

    # Try to extract section for the given phase
    phase_upper = phase.upper()
    pattern = rf"###\s+{phase_upper}\s*\n(.*?)(?=\n###\s|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def merge_context(
    global_brief: str,
    session_brief: str,
    transcript: List[Dict],
    summary: str,
    contract: str,
    phase: str,
    message: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Tuple[List[Dict], int, int]:
    """Assemble the message array for the API call.

    Args:
        global_brief: Body text of the global brief.
        session_brief: Body text of the session brief.
        transcript: List of exchange dicts with 'user' and 'assistant' messages.
        summary: Text of the thread summary (if compacted).
        contract: Body of the collaboration contract.
        phase: Current phase name.
        message: The current message to send.

    Returns:
        (messages, total_tokens, remaining_budget)
        messages: List of dicts ready for ProviderAdapter.call()
    """
    messages = []

    # System message: contract
    phase_instruction = get_phase_specific_instruction(phase)
    contract_text = contract
    if phase_instruction:
        contract_text += f"\n\n## Current Phase\n{phase}\n\n## Phase Instructions\n{phase_instruction}"
    messages.append({"role": "system", "content": contract_text.strip()})

    # User message: global brief
    if global_brief:
        messages.append({"role": "user", "content": f"## Global Brief\n\n{global_brief}"})

    # User message: session brief
    if session_brief:
        messages.append({"role": "user", "content": f"## Session Brief\n\n{session_brief}"})

    # User message: thread summary
    if summary:
        messages.append({"role": "user", "content": f"## Thread Summary\n\n{summary}"})

    # Transcript turns as user/assistant pairs
    for exchange in transcript:
        if exchange.get("user"):
            messages.append({"role": "user", "content": exchange["user"]})
        if exchange.get("assistant"):
            messages.append({"role": "assistant", "content": exchange["assistant"]})

    # Current message
    messages.append({"role": "user", "content": message})

    # Count tokens
    total_tokens = count_tokens("\n\n".join(m.get("content", "") for m in messages))

    # Check budget
    if total_tokens <= token_budget:
        return messages, total_tokens, token_budget - total_tokens

    # If over budget but within 110%, drop oldest transcript turns
    if total_tokens <= int(token_budget * 1.1):
        # Find the transcript turn indices (user/assistant pairs before the current message)
        # Messages structure: [0]=system, [1]=global brief, [2]=session brief, [3]=summary
        # then transcript pairs, then last is current message
        base_count = 1  # system
        if global_brief:
            base_count += 1
        if session_brief:
            base_count += 1
        if summary:
            base_count += 1

        transcript_pairs = len(transcript)
        while total_tokens > token_budget and transcript_pairs > 0:
            # Remove the oldest transcript pair (2 messages)
            # Find their indices
            idx = base_count  # first transcript pair starts here
            messages.pop(idx)  # remove user
            if idx < len(messages) and messages[idx].get("role") == "assistant":
                messages.pop(idx)  # remove corresponding assistant
            transcript_pairs -= 1
            total_tokens = count_tokens(
                "\n\n".join(m.get("content", "") for m in messages)
            )

        if total_tokens <= token_budget:
            return messages, total_tokens, token_budget - total_tokens

    # Still over budget, refuse
    raise BudgetExceededError(
        f"Context exceeds token budget ({total_tokens} > {token_budget}). "
        f"Run `aim compact <session>` to reduce thread length."
    )


def parse_metadata_block(response_text: str) -> Tuple[dict, str]:
    """Extract the YAML metadata block from the start of the response.

    Expected format:
    ---
    next_action: null
    phase_suggestion: null
    confidence: 9
    ---
    <actual response text>

    Returns (metadata_dict, cleaned_response_text).
    """
    text = response_text.lstrip()
    if not text.startswith("---"):
        return {}, response_text

    # Split on ---, handling at most 3 parts
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, response_text

    yaml_str = parts[1].strip()
    cleaned = parts[2].strip()

    try:
        metadata = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        logger.warning("Failed to parse YAML metadata block in response")
        return {}, response_text

    # Validate expected fields
    if not isinstance(metadata, dict):
        return {}, response_text

    # Validate confidence
    confidence = metadata.get("confidence")
    if confidence is not None:
        try:
            confidence = int(confidence)
            if confidence < 1 or confidence > 10:
                confidence = None
            metadata["confidence"] = confidence
        except (ValueError, TypeError):
            metadata["confidence"] = None

    return metadata, cleaned


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken, with fallback.

    Falls back to rough character-based estimate if tiktoken is not available.
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except (ImportError, Exception):
        # Rough estimate: ~4 chars per token
        return len(text) // 4


def enforce_monthly_budget(
    accumulated: float,
    cap: float = DEFAULT_MONTHLY_CAP,
    current_call_estimate: float = 0.0,
) -> Tuple[bool, str]:
    """Check whether the current call would exceed the monthly cap.

    Returns (allowed, message).
    """
    env_cap = os.environ.get("AIM_COST_MONTHLY_CAP")
    if env_cap is not None:
        try:
            cap = float(env_cap)
        except (ValueError, TypeError):
            pass

    if cap <= 0:
        return True, "ok"

    if accumulated + current_call_estimate < cap:
        return True, "ok"

    if accumulated + current_call_estimate >= cap and accumulated < cap:
        return True, "warn"

    return False, (
        f"Monthly cap ${cap:.2f} reached. "
        f"Set AIM_COST_MONTHLY_CAP to a higher value or override with --force."
    )


def check_brief_age(brief_path: str = None) -> Tuple[bool, str]:
    """Check if the global brief is too old (>7 days since last update).

    Returns (is_healthy, message).
    """
    if brief_path is None:
        brief_path = os.path.join(MULTI_AI_DIR, "brief.md")

    from datetime import datetime, date, timezone
    frontmatter, _ = read_yaml_frontmatter(brief_path)
    updated = frontmatter.get("updated", "")

    if not updated:
        return False, "Global brief: 'updated' date not set."

    try:
        if isinstance(updated, date) and not isinstance(updated, datetime):
            # YAML date object -> convert to datetime
            updated_date = datetime(updated.year, updated.month, updated.day, tzinfo=timezone.utc)
        elif isinstance(updated, datetime):
            updated_date = updated
            if updated_date.tzinfo is None:
                updated_date = updated_date.replace(tzinfo=timezone.utc)
        else:
            # String
            updated_date = datetime.fromisoformat(str(updated))
            if updated_date.tzinfo is None:
                updated_date = updated_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - updated_date
        days = delta.days
        if days > 7:
            return False, f"Global brief last updated {days} days ago ({updated}). Update it."
        return True, f"Global brief updated {days} day(s) ago ({updated})."
    except (ValueError, TypeError):
        return False, f"Global brief has invalid 'updated' date: {updated}"


def read_transcript(transcript_path: str) -> List[Dict]:
    """Read transcript.jsonl and return list of exchange dicts."""
    if not os.path.exists(transcript_path):
        return []

    exchanges = []
    try:
        with open(transcript_path, "r") as f:
            lines = f.readlines()

        # Group into user/assistant pairs
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                i += 1
                continue

            if entry.get("role") == "user":
                exchange = {"user": entry.get("content", "")}
                # Look for next assistant line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        try:
                            next_entry = json.loads(next_line)
                            if next_entry.get("role") == "assistant":
                                exchange["assistant"] = next_entry.get("content", "")
                                i += 1  # skip the assistant line
                        except json.JSONDecodeError:
                            pass
                exchanges.append(exchange)
            i += 1
    except (IOError, OSError) as e:
        logger.warning(f"Cannot read transcript {transcript_path}: {e}")

    return exchanges
