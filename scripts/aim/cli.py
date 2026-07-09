"""CLI routing and output formatting for the aim CLI.

Parses command-line arguments, routes to the appropriate handler function,
formats output for terminal display.
"""

import argparse
import os
import sys
import time
import json
import subprocess
import signal
from datetime import datetime, timezone
from typing import Optional

import yaml

from .errors import (
    AimCliError, SessionNotFoundError, SessionExistsError,
    LockActiveError, ProviderAuthError, ProviderRateLimitError,
    ProviderOverloadError, ProviderContextLengthError,
    BudgetExceededError, SecretFoundError, NoAPIKeyError,
    ProviderError,
)
from .protocol import (
    MULTI_AI_DIR,
    read_yaml_frontmatter,
    read_brief,
    merge_context,
    parse_metadata_block,
    count_tokens,
    enforce_monthly_budget,
    check_brief_age,
    read_transcript,
    DEFAULT_TOKEN_BUDGET,
)
from .session import (
    new_session,
    list_sessions,
    show_session,
    compact_session,
    session_path,
    update_session_meta,
    update_session_state,
)
from .exchange import (
    acquire_lock,
    release_lock,
    log_exchange,
    write_decision,
    scan_secrets,
    read_session_index,
    read_lock,
)
from .adapters import get_adapter, list_available_providers


# ─────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────

def _supports_color():
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def print_error(msg):
    if _supports_color():
        print(f"\033[91mERROR:\033[0m {msg}", file=sys.stderr)
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def print_success(msg):
    if _supports_color():
        print(f"\033[92m✓\033[0m {msg}")
    else:
        print(f"✓ {msg}")


def print_warning(msg):
    if _supports_color():
        print(f"\033[93mWARNING:\033[0m {msg}")
    else:
        print(f"WARNING: {msg}")


def print_info(msg):
    print(msg)


# ─────────────────────────────────────────────
# Argparse builder
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aim",
        description="Multi-AI Collaboration Interface CLI — let any AI talk to any other AI.",
        epilog="See https://github.com/michael/aim for full documentation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # aim new
    p_new = subparsers.add_parser("new", help="Create a new collaboration session")
    p_new.add_argument("name", help="Session name (alphanumeric + hyphens)")
    p_new.add_argument("--goal", "-g", help="Session goal description")
    p_new.add_argument("--provider", default="openai",
                       choices=list_available_providers(),
                       help="Default AI provider")
    p_new.add_argument("--model", help="Default model for the provider")
    p_new.add_argument("--no-external", action="store_true",
                       help="Mark session as no-external (secrets-safe)")

    # aim list
    p_list = subparsers.add_parser("list", help="List sessions")
    p_list.add_argument("--all", "-a", action="store_true",
                        help="Show all sessions including completed")
    p_list.add_argument("--phase", help="Filter by phase")

    # aim show
    p_show = subparsers.add_parser("show", help="Show session details")
    p_show.add_argument("session", help="Session name")

    # aim tail
    p_tail = subparsers.add_parser("tail", help="Show recent exchanges")
    p_tail.add_argument("session", help="Session name")
    p_tail.add_argument("-n", type=int, default=20,
                        help="Number of lines (default: 20)")

    # aim watch
    p_watch = subparsers.add_parser("watch", help="Follow session exchange log (tail -f)")
    p_watch.add_argument("session", help="Session name")

    # aim send
    p_send = subparsers.add_parser("send", help="Send a message to the target AI in this session")
    p_send.add_argument("session", help="Session name")
    p_send.add_argument("message", nargs="?",
                        help="Message text (reads from stdin if omitted)")
    p_send.add_argument("--provider", help="Provider override (default: from session meta.yaml)")
    p_send.add_argument("--model", help="Model override (default: from session meta.yaml)")
    p_send.add_argument("--compact", action="store_true",
                        help="Compact thread before sending")
    p_send.add_argument("--dry-run", action="store_true",
                        help="Show what would be sent, don't send")
    p_send.add_argument("--no-brief", action="store_true",
                        help="Skip global brief injection")
    p_send.add_argument("--no-session-brief", action="store_true",
                        help="Skip session brief injection")
    p_send.add_argument("--force", action="store_true",
                        help="Override budget/cost guardrails")
    p_send.add_argument("--timeout", type=int, default=60,
                        help="API timeout in seconds")

    # aim compact
    p_compact = subparsers.add_parser("compact", help="Compact thread transcript into summary.md")
    p_compact.add_argument("session", help="Session name")
    p_compact.add_argument("--force", action="store_true",
                           help="Skip confirmation")

    # aim decide
    p_decide = subparsers.add_parser("decide", help="Record Michael's decision")
    p_decide.add_argument("session", help="Session name")
    p_decide.add_argument("--answer", "-a", required=True,
                          help="The decision made")
    p_decide.add_argument("--question", "-q",
                          help="The question being decided")
    p_decide.add_argument("--rationale", "-r",
                          help="Rationale for the decision")
    p_decide.add_argument("--requested-by", default="michael",
                          choices=["claude", "openai", "gemini", "michael"],
                          help="Who requested the decision")
    p_decide.add_argument("--impact",
                          help="Impact of this decision")
    p_decide.add_argument("--alternative", action="append", nargs=2,
                          metavar=("KEY", "VALUE"),
                          help="Alternative option (can be specified multiple times)")

    # aim status
    p_status = subparsers.add_parser("status", help="System status and health check")
    p_status.add_argument("--verbose", "-v", action="store_true",
                          help="Detailed output")

    # aim costs
    p_costs = subparsers.add_parser("costs", help="Show cost breakdown")
    p_costs.add_argument("--reset", action="store_true",
                         help="Reset accumulated costs (requires confirmation)")

    # aim repair
    p_repair = subparsers.add_parser("repair", help="Fix a corrupted session or lock")
    p_repair.add_argument("session", help="Session name")
    p_repair.add_argument("--force", action="store_true",
                          help="Skip confirmation prompts")

    return parser


# ─────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────

def handle_new(args):
    """Create a new session."""
    try:
        result = new_session(
            name=args.name,
            goal=args.goal,
            provider=args.provider,
            model=args.model,
            no_external=args.no_external,
        )
        print_success(f"Session '{args.name}' created. Phase: scoping.")
        print_info(f"  Provider: {result.get('default_provider', 'openai')}")
        print_info(f"  Model: {result.get('default_model', 'gpt-4o')}")
        if args.goal:
            print_info(f"  Goal: {args.goal}")
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)
    except SessionExistsError as e:
        print_error(str(e))
        sys.exit(1)


def handle_list(args):
    """List sessions."""
    sessions = list_sessions(
        active_only=not args.all,
        phase_filter=args.phase,
    )

    if not sessions:
        if args.all:
            print_info("No sessions found. Create one with `aim new <name>`.")
        else:
            print_info("No active sessions. Create one with `aim new <name>`.")
        return

    # Print header
    print_info(f"{'Session':<25} {'Phase':<15} {'Exchanges':<10} {'Cost':<12} {'Last updated':<20}")
    print_info("-" * 82)

    for sess in sessions:
        name = sess["name"]
        phase = sess["phase"]
        ex_count = sess["exchange_count"]
        cost = f"${sess['cost_accumulated']:.4f}"
        updated = sess["last_updated"][:19] if sess["last_updated"] else "-"
        print_info(f"{name:<25} {phase:<15} {ex_count:<10} {cost:<12} {updated:<20}")


def handle_show(args):
    """Show session details."""
    try:
        sess = show_session(args.session)
    except SessionNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    print_info(f"Session: {sess['name']}")
    print_info(f"Goal: {sess['goal'] or 'Not set'}")
    print_info(f"Phase: {sess['phase']}")
    print_info(f"Provider: {sess['default_provider']} ({sess['default_model']})")
    print_info(f"Created: {sess['created'][:19] if sess['created'] else '-'}")
    print_info(f"Exchanges: {sess['exchange_count']}")
    print_info(f"Cost: ${sess['cost_accumulated']:.4f}")
    print_info(f"Decisions: {sess['decision_count']}")
    if sess["no_external"]:
        print_warning("No-external mode (API calls blocked)")
    if sess["locked"]:
        lock = sess["lock_info"]
        print_warning(f"Locked by PID {lock.get('pid', '?')} since {lock.get('acquired', '?')}")

    if sess["decisions"]:
        print_info("")
        print_info("Decisions:")
        for d in sess["decisions"]:
            print_info(f"  - {d}")

    if sess["recent_exchanges"]:
        print_info("")
        print_info("Recent exchanges:")
        for ex in sess["recent_exchanges"]:
            heading = ex.get("heading", "")
            msg = ex.get("message", "")[:80]
            resp = ex.get("response", "")[:80]
            if heading:
                print_info(f"  {heading[:60]}")
            if msg:
                print_info(f"    → {msg}")
            if resp:
                print_info(f"    ← {resp}")
    else:
        print_info("")
        print_info(f"No exchanges yet. Send a message with `aim send {args.session} \"your message\"`.")


def handle_tail(args):
    """Show recent exchange log lines."""
    sess_path = session_path(args.session)
    log_path = os.path.join(sess_path, "exchange-log.md")

    if not os.path.isdir(sess_path):
        print_error(f"Session '{args.session}' not found.")
        sys.exit(1)

    if not os.path.exists(log_path):
        print_info("Exchange log is empty.")
        return

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        print_error(f"Cannot read exchange log: {e}")
        sys.exit(1)

    if not lines:
        print_info("Exchange log is empty.")
        return

    tail_lines = lines[-args.n:]
    for line in tail_lines:
        print(line, end="")


def handle_watch(args):
    """Follow the exchange log like tail -f."""
    sess_path = session_path(args.session)
    log_path = os.path.join(sess_path, "exchange-log.md")

    if not os.path.isdir(sess_path):
        print_error(f"Session '{args.session}' not found.")
        sys.exit(1)

    if not os.path.exists(log_path):
        print_info("Exchange log is empty. Send a message first with `aim send`.")
        return

    # Use tail -f via subprocess
    try:
        proc = subprocess.Popen(
            ["tail", "-f", "-n", "10", log_path],
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )
        # Forward signals
        def signal_handler(sig, frame):
            proc.terminate()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        proc.wait()
    except FileNotFoundError:
        # tail not available, use polling fallback
        print_info("tail command not available. Using polling mode...")
        try:
            last_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
            while True:
                time.sleep(1)
                if os.path.exists(log_path):
                    current_size = os.path.getsize(log_path)
                    if current_size > last_size:
                        with open(log_path, "r") as f:
                            f.seek(last_size)
                            new_data = f.read()
                            print(new_data, end="")
                            sys.stdout.flush()
                        last_size = current_size
        except KeyboardInterrupt:
            pass


def handle_send(args):
    """Send a message to the target AI."""
    sess_path = session_path(args.session)

    if not os.path.isdir(sess_path):
        print_error(f"Session '{args.session}' not found.")
        sys.exit(1)

    # Read meta.yaml
    meta_path = os.path.join(sess_path, "meta.yaml")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}

    # Check no-external flag
    if meta.get("no_external", False) and not args.force:
        print_error(
            f"Session '{args.session}' is marked as no-external "
            f"(meta.yaml: no_external: true). "
            f"Cannot send data to external API. Remove the flag to enable sends."
        )
        sys.exit(1)

    # Get provider and model
    provider_name = args.provider or meta.get("default_provider", "openai")
    model = args.model or meta.get("default_model", "")

    # Read message from args or stdin
    if args.message:
        message = args.message
    else:
        if not sys.stdin.isatty():
            message = sys.stdin.read().strip()
        else:
            print_error("No message provided. Provide a message argument or pipe input via stdin.")
            sys.exit(1)

    if not message:
        print_error("Message cannot be empty.")
        sys.exit(1)

    # Read global brief
    global_brief_path = os.path.join(MULTI_AI_DIR, "brief.md")
    global_brief = ""
    if not args.no_brief:
        global_brief = read_brief(global_brief_path)

    # Check brief age
    brief_healthy, brief_msg = check_brief_age(global_brief_path)
    if not brief_healthy and not args.force:
        print_warning(brief_msg)

    # Read session brief
    session_brief = ""
    if not args.no_session_brief:
        session_brief_path = os.path.join(sess_path, "brief.md")
        session_brief = read_brief(session_brief_path)

    # Read contract
    contract_path = os.path.join(MULTI_AI_DIR, "contract.md")
    contract = read_brief(contract_path)

    # Read state for phase
    state_path = os.path.join(sess_path, "state.yaml")
    phase = "scoping"
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = yaml.safe_load(f) or {}
            phase = state.get("phase", "scoping")

    # Read transcript
    transcript_path = os.path.join(sess_path, "transcript.jsonl")
    transcript = read_transcript(transcript_path)

    # Read summary
    summary_path = os.path.join(sess_path, "summary.md")
    summary = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = f.read()

    # Compact if requested
    if args.compact:
        compact_result = compact_session(args.session)
        print_info(compact_result["message"])
        # Re-read transcript after compaction
        transcript = read_transcript(transcript_path)
        summary_path = os.path.join(sess_path, "summary.md")
        summary = ""
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = f.read()

    # Merge context
    try:
        messages, total_tokens, remaining_budget = merge_context(
            global_brief=global_brief,
            session_brief=session_brief,
            transcript=transcript,
            summary=summary,
            contract=contract,
            phase=phase,
            message=message,
            token_budget=DEFAULT_TOKEN_BUDGET,
        )
    except BudgetExceededError as e:
        if args.force:
            print_warning(f"Token budget exceeded: {e}")
            # Try without transcript
            messages, total_tokens, remaining_budget = merge_context(
                global_brief=global_brief,
                session_brief=session_brief,
                transcript=[],
                summary=summary,
                contract=contract,
                phase=phase,
                message=message,
                token_budget=DEFAULT_TOKEN_BUDGET,
            )
        else:
            print_error(str(e))
            sys.exit(1)

    # Estimate cost (rough: use gpt-4o rates)
    estimated_cost = (total_tokens * 2.50 / 1_000_000) + (1000 * 10.00 / 1_000_000)

    # Scrub messages for display (hide content)
    display_messages = []
    for m in messages:
        content_preview = m["content"][:80] + "..." if len(m["content"]) > 80 else m["content"]
        display_messages.append({"role": m["role"], "content": content_preview})

    # Print context summary
    print_info(f"[Sending to {provider_name}...]")
    print_info(f"Context: {total_tokens} tokens | Phase: {phase}")
    print_info(f"Method: api ({model})")

    # Secret scanning
    secret_matches = scan_secrets(message)
    if secret_matches and not args.force:
        raise SecretFoundError(secret_matches)

    # Monthly budget check
    index_path = os.path.join(MULTI_AI_DIR, "sessions", "index.yaml")
    index = read_session_index(index_path)
    total_cost = sum(
        s.get("cost_accumulated", 0.0)
        for s in index.get("sessions", {}).values()
    )
    budget_ok, budget_msg = enforce_monthly_budget(
        total_cost, current_call_estimate=estimated_cost
    )
    if not budget_ok:
        if args.force:
            print_warning(f"Cost cap exceeded: {budget_msg}")
        else:
            print_error(budget_msg)
            sys.exit(1)
    elif budget_msg == "warn":
        print_warning("Approaching monthly cost cap.")

    # Dry run
    if args.dry_run:
        print_info("")
        print_info("[DRY RUN] Would send:")
        print_info("─" * 50)
        for m in display_messages:
            role = m["role"].upper()
            content = m["content"]
            print_info(f"  {role}: {content}")
        print_info("─" * 50)
        print_info(f"Total: ~{total_tokens} tokens | Estimated cost: ${estimated_cost:.4f}")
        print_info("No response was sent. Use `aim send` without --dry-run to send.")
        return

    # Acquire lock
    try:
        acquire_lock(sess_path)
    except LockActiveError as e:
        print_error(str(e))
        sys.exit(1)

    lock_acquired = True

    try:
        # Get adapter and send
        adapter = get_adapter(provider_name)
        response = adapter.call(
            messages=messages,
            model=model if model else None,
            timeout=args.timeout,
        )

        # Parse response metadata
        metadata, cleaned_response = parse_metadata_block(response.content)
        response_text = cleaned_response or response.content

        # Build exchange dict
        now = datetime.now(timezone.utc)
        exchange = {
            "ts": now.isoformat(),
            "from": "claude",
            "to": provider_name,
            "phase": phase,
            "message": message,
            "context_injected": {
                "global_brief_version": "latest",
                "session_brief": args.session,
                "thread_turns": len(transcript),
                "tokens_total": total_tokens,
            },
            "method": "api",
            "model": response.model,
            "response_ts": now.isoformat(),
            "response_text": response_text,
            "response_id": response.response_id,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost": response.cost,
            "finish_reason": response.finish_reason,
            "metadata_parsed": metadata,
        }

        # Log exchange
        global_log_dir = os.path.join(MULTI_AI_DIR, "exchange-log")
        log_exchange(sess_path, global_log_dir, exchange)

        # Update session metadata
        new_exchange_count = meta.get("exchange_count", 0) + 1
        new_cost = meta.get("cost_accumulated", 0.0) + response.cost
        update_session_meta(args.session, {
            "exchange_count": new_exchange_count,
            "cost_accumulated": new_cost,
            "last_exchange": now.isoformat(),
        })

        # Update index cost
        index_sessions = index.get("sessions", {})
        if args.session in index_sessions:
            index_sessions[args.session]["exchange_count"] = new_exchange_count
            index_sessions[args.session]["cost_accumulated"] = new_cost
            index_sessions[args.session]["last_updated"] = now.isoformat()
            from .exchange import update_session_index
            update_session_index(index_path, args.session, index_sessions[args.session])

        # Print response
        print_info("")
        print_info(f"Response received ({response.elapsed_ms}ms):")
        print_info(response_text)

        # Print metadata
        print_info("")
        elapsed = response.elapsed_ms
        print_info(
            f"Tokens: {response.tokens_in} in / {response.tokens_out} out | "
            f"Cost: ${response.cost:.4f} | "
            f"Model: {response.model} | "
            f"Time: {elapsed}ms"
        )

        if metadata:
            next_action = metadata.get("next_action")
            phase_suggestion = metadata.get("phase_suggestion")
            confidence = metadata.get("confidence")
            if next_action:
                print_info(f"Suggested next action: {next_action}")
            if phase_suggestion:
                print_info(f"Phase suggestion: {phase_suggestion}")
            if confidence:
                print_info(f"Confidence: {confidence}/10")

    except (NoAPIKeyError, ProviderAuthError) as e:
        print_error(str(e))
        sys.exit(2)
    except ProviderRateLimitError as e:
        print_error(str(e))
        sys.exit(2)
    except ProviderContextLengthError as e:
        print_error(str(e))
        sys.exit(1)
    except ProviderOverloadError as e:
        print_error(str(e))
        sys.exit(2)
    except ProviderError as e:
        print_error(str(e))
        # Write failed entry to exchange log
        now = datetime.now(timezone.utc).isoformat()
        error_exchange = {
            "ts": now,
            "from": "claude",
            "to": provider_name,
            "phase": phase,
            "message": message,
            "error": str(e),
            "method": "api",
            "model": model,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0.0,
        }
        global_log_dir = os.path.join(MULTI_AI_DIR, "exchange-log")
        log_exchange(sess_path, global_log_dir, error_exchange)
        sys.exit(2)
    except SecretFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nInterrupted.")
        sys.exit(130)
    finally:
        # Always release lock
        if lock_acquired:
            release_lock(sess_path)


def handle_compact(args):
    """Compact a session's thread."""
    try:
        sess_path = session_path(args.session)
        try:
            acquire_lock(sess_path)
        except LockActiveError as e:
            print_error(str(e))
            sys.exit(1)
        try:
            result = compact_session(args.session)
            print_info(result["message"])
        finally:
            release_lock(sess_path)
    except SessionNotFoundError as e:
        print_error(str(e))
        sys.exit(1)


def handle_decide(args):
    """Record a decision."""
    sess_path = session_path(args.session)
    if not os.path.isdir(sess_path):
        print_error(f"Session '{args.session}' not found.")
        sys.exit(1)

    question = args.question or args.answer

    # Build alternatives
    alternatives = {}
    if args.alternative:
        for key, value in args.alternative:
            alternatives[key] = value

    decision = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "requested_by": args.requested_by,
        "session": args.session,
        "question": question,
        "alternatives": alternatives if alternatives else None,
        "decision": args.answer,
        "rationale": args.rationale or "",
        "impact": args.impact or "",
        "superseded_by": None,
    }

    try:
        filename = write_decision(sess_path, decision)
        print_success(f"Decision recorded: {question}")
        print_info(f"File: decisions/{filename}")
    except Exception as e:
        print_error(f"Failed to write decision: {e}")
        sys.exit(1)

    # Update state if in decision phase
    state_path = os.path.join(sess_path, "state.yaml")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = yaml.safe_load(f) or {}
        if state.get("phase") == "decision":
            # Return to the phase that originated the decision request
            transition_log = state.get("transition_log", [])
            return_phase = "executing"  # default
            if transition_log:
                # Find the most recent transition INTO decision
                for t in reversed(transition_log):
                    if t.get("to") == "decision":
                        return_phase = t.get("from", "executing")
                        break
            update_session_state(args.session, {
                "phase": return_phase,
                "transition_rationale": f"Decision made: {args.answer}",
                "transition_by": "michael",
            })
            print_info(f"Phase transitioned: decision -> {return_phase}")


def handle_status(args):
    """Show system status and health check."""
    print_info("Multi-AI Collaboration System Status")
    print_info("─" * 50)

    # Check API connectivity
    providers_ok = {}
    for pname in list_available_providers():
        try:
            adapter = get_adapter(pname)
            # Just getting the adapter validates the key exists
            providers_ok[pname] = "Connected (key present)"
        except NoAPIKeyError:
            providers_ok[pname] = "Not configured (no API key)"
        except Exception as e:
            providers_ok[pname] = f"Error: {e}"

    for pname, status in providers_ok.items():
        marker = "✓" if "Connected" in status else "✗"
        if "Not configured" in status:
            marker = "○"
        color = "\033[92m" if "Connected" in status else ("\033[93m" if "Not configured" in status else "\033[91m")
        reset = "\033[0m" if _supports_color() else ""
        print_info(f"  {pname}: {color}{marker} {status}{reset}")

    # Brief health
    brief_path = os.path.join(MULTI_AI_DIR, "brief.md")
    brief_frontmatter, brief_body = read_yaml_frontmatter(brief_path)
    if brief_frontmatter:
        updated = brief_frontmatter.get("updated", "unknown")
        version = brief_frontmatter.get("version", "?")
        token_budget = brief_frontmatter.get("token_budget", 2500)
        brief_tokens = count_tokens(brief_body)
        brief_healthy, brief_msg = check_brief_age(brief_path)
        marker = "✓" if brief_healthy else "!"
        print_info(f"  Brief: v{version}, updated {updated} ({brief_tokens}/{token_budget} tokens) {marker}")

    # Sessions
    sessions = list_sessions(active_only=False)
    active_count = len([s for s in sessions if s["phase"] != "complete"])
    total_count = len(sessions)
    print_info(f"  Sessions: {active_count} active, {total_count} total")

    # Costs
    index_path = os.path.join(MULTI_AI_DIR, "sessions", "index.yaml")
    index = read_session_index(index_path)
    total_cost = sum(
        s.get("cost_accumulated", 0.0)
        for s in index.get("sessions", {}).values()
    )
    monthly_cap = os.environ.get("AIM_COST_MONTHLY_CAP", "100.0")
    cap_used = 0.0
    try:
        cap_used = (total_cost / float(monthly_cap)) * 100 if float(monthly_cap) > 0 else 0
    except (ValueError, ZeroDivisionError):
        pass
    print_info(f"  Total cost: ${total_cost:.4f} (monthly)")
    print_info(f"  Cap: ${monthly_cap} (used: {cap_used:.1f}%)")

    # Config
    multi_ai_dir_set = "✓" if os.path.isdir(MULTI_AI_DIR) else "✗"
    print_info(f"  MULTI_AI_DIR: {multi_ai_dir_set}")


def handle_costs(args):
    """Show cost breakdown."""
    if args.reset:
        print_warning("Cost reset not yet implemented (would require confirmation).")
        return

    index_path = os.path.join(MULTI_AI_DIR, "sessions", "index.yaml")
    index = read_session_index(index_path)
    sessions = index.get("sessions", {})

    if not sessions:
        print_info("No sessions found.")
        return

    total = 0.0
    print_info(f"{'Session':<30} {'Exchanges':<12} {'Cost':<12}")
    print_info("-" * 54)

    for name, data in sorted(sessions.items()):
        cost = data.get("cost_accumulated", 0.0)
        ex_count = data.get("exchange_count", 0)
        print_info(f"{name:<30} {ex_count:<12} ${cost:<10.4f}")
        total += cost

    print_info("-" * 54)
    print_info(f"{'TOTAL':<30} {'':<12} ${total:<10.4f}")
    print_info("")
    print_info(f"Monthly cap: ${os.environ.get('AIM_COST_MONTHLY_CAP', '100.0')}")


def handle_repair(args):
    """Fix a corrupted session."""
    sess_path = session_path(args.session)

    if not os.path.isdir(sess_path):
        print_error(f"Session '{args.session}' not found.")
        sys.exit(1)

    actions = []

    # Check lock file
    lock_data = read_lock(sess_path)
    if lock_data:
        from datetime import timezone
        acquired = lock_data.get("acquired", "")
        try:
            dt = datetime.fromisoformat(acquired)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - dt).total_seconds()
            if age > 600 or args.force:
                os.remove(os.path.join(sess_path, "lock"))
                actions.append(f"Stale lock removed (age: {age:.0f}s)")
            else:
                actions.append(f"Lock held by PID {lock_data.get('pid', '?')} ({age:.0f}s old) — not breaking")
        except (ValueError, TypeError):
            if args.force:
                os.remove(os.path.join(sess_path, "lock"))
                actions.append("Corrupt lock removed")
    else:
        actions.append("Lock: OK (not locked)")

    # Check state.yaml
    state_path = os.path.join(sess_path, "state.yaml")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = yaml.safe_load(f) or {}
            phase = state.get("phase", "")
            if phase not in ["idle", "scoping", "executing", "reviewing", "decision", "complete"]:
                state["phase"] = "scoping"
                with open(state_path, "w") as f:
                    yaml.dump(state, f)
                actions.append(f"State phase reset from '{phase}' to 'scoping'")
            else:
                actions.append(f"State: OK (phase={phase})")
        except Exception as e:
            actions.append(f"State file corrupted: {e}")
    else:
        actions.append("State file missing — cannot repair automatically")

    # Check exchange log
    log_path = os.path.join(sess_path, "exchange-log.md")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            content = f.read()
        if content.strip() and not content.startswith("# Exchange"):
            actions.append("Exchange log may be corrupted — first line missing header")
        else:
            actions.append("Exchange log: OK")
    else:
        actions.append("Exchange log: missing")

    # Print results
    print_info(f"Repair actions for '{args.session}':")
    for action in actions:
        print_info(f"  • {action}")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Ensure the multi-ai directory structure exists
    os.makedirs(os.path.join(MULTI_AI_DIR, "sessions"), exist_ok=True)
    os.makedirs(os.path.join(MULTI_AI_DIR, "exchange-log"), exist_ok=True)

    try:
        if args.command == "new":
            handle_new(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "show":
            handle_show(args)
        elif args.command == "tail":
            handle_tail(args)
        elif args.command == "watch":
            handle_watch(args)
        elif args.command == "send":
            handle_send(args)
        elif args.command == "compact":
            handle_compact(args)
        elif args.command == "decide":
            handle_decide(args)
        elif args.command == "status":
            handle_status(args)
        elif args.command == "costs":
            handle_costs(args)
        elif args.command == "repair":
            handle_repair(args)
    except SessionNotFoundError as e:
        print_error(str(e))
        return 1
    except LockActiveError as e:
        print_error(str(e))
        return 1
    except ProviderAuthError as e:
        print_error(str(e))
        return 2
    except ProviderRateLimitError as e:
        print_error(str(e))
        return 2
    except ProviderOverloadError as e:
        print_error(str(e))
        return 2
    except BudgetExceededError as e:
        print_info(str(e))
        return 0
    except SecretFoundError as e:
        print_error(str(e))
        return 1
    except AimCliError as e:
        print_error(str(e))
        return 1
    except KeyboardInterrupt:
        print_info("\nInterrupted.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
