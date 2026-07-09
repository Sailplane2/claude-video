#!/usr/bin/env python3
"""Call ChatGPT via OpenAI API. stdlib-only — no pip install needed.

Usage:
    echo "your prompt" | python3 chatgpt_api.py [--model gpt-4o] [--system "..."]
    python3 chatgpt_api.py --prompt "your prompt" [--model gpt-4o] [--system "..."]

Environment:
    OPENAI_API_KEY — loaded from env or ~/.config/watch/.env
    Default model: gpt-4o
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gpt-4o"
API_URL = "https://api.openai.com/v1/chat/completions"


def load_api_key():
    """Load OPENAI_API_KEY from environment or ~/.config/watch/.env."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    env_file = Path.home() / ".config/watch/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "OPENAI_API_KEY":
                    return v.strip().strip('"').strip("'")

    print("ERROR: OPENAI_API_KEY not found.", file=sys.stderr)
    print("Set it in environment or ~/.config/watch/.env", file=sys.stderr)
    print("Run: python3 ~/claude-video/scripts/update_api_key.py openai", file=sys.stderr)
    sys.exit(1)


def call_chatgpt(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Send a single-turn chat completion and return the response text."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })

    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        print(f"ERROR: OpenAI API returned HTTP {e.code}", file=sys.stderr)
        print(error_body, file=sys.stderr)
        sys.exit(2)
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach OpenAI API — {e.reason}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"ERROR: Unexpected API response: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description="Call ChatGPT via OpenAI API"
    )
    parser.add_argument(
        "--prompt", "-p",
        help="The message to send to ChatGPT (reads from stdin if omitted)",
    )
    parser.add_argument(
        "--system", "-s",
        default="",
        help="System prompt to set context for the conversation",
    )
    parser.add_argument(
        "--system-file", "-S",
        default="",
        help="Path to a file whose contents will be used as the system prompt",
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL,
        help=f"Model to use (default: {MODEL})",
    )
    args = parser.parse_args()

    # Resolve system prompt: --system-file overrides --system
    system = args.system
    if args.system_file:
        sf = Path(args.system_file)
        if not sf.exists():
            print(f"ERROR: System file not found: {args.system_file}", file=sys.stderr)
            sys.exit(1)
        system = sf.read_text()
    if args.prompt:
        prompt = args.prompt
    else:
        if sys.stdin.isatty():
            print("ERROR: No prompt provided. Pipe input or use --prompt.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("ERROR: Empty prompt.", file=sys.stderr)
        sys.exit(1)

    global api_key
    api_key = load_api_key()

    response = call_chatgpt(prompt, system=system, model=args.model)
    print(response)


if __name__ == "__main__":
    main()
