#!/usr/bin/env python3
"""
Secure API key updater — masked entry, writes directly to config.
Claude never sees the key. Run after rotating any key.

Usage:
    python3 ~/claude-video/scripts/update_api_key.py deepseek
    python3 ~/claude-video/scripts/update_api_key.py gemini
    python3 ~/claude-video/scripts/update_api_key.py openai
    python3 ~/claude-video/scripts/update_api_key.py groq
    python3 ~/claude-video/scripts/update_api_key.py obsidian
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Key Registry ─────────────────────────────────────────────────────
# Each key knows its config file, variable name, and line pattern.

KEYS = {
    "deepseek": {
        "label": "DeepSeek API Key",
        "file": Path.home() / ".zshrc",
        "var": "DEEPSEEK_API_KEY",
        "pattern": 'export DEEPSEEK_API_KEY=',
        "how": "replace_line",
    },
    "gemini": {
        "label": "Gemini API Key",
        "file": Path.home() / ".config/watch/.env",
        "var": "GEMINI_API_KEY",
        "pattern": "GEMINI_API_KEY=",
        "how": "replace_line",
    },
    "openai": {
        "label": "OpenAI API Key",
        "file": Path.home() / ".config/watch/.env",
        "var": "OPENAI_API_KEY",
        "pattern": "OPENAI_API_KEY=",
        "how": "replace_line",
    },
    "groq": {
        "label": "Groq API Key",
        "file": Path.home() / ".config/watch/.env",
        "var": "GROQ_API_KEY",
        "pattern": "GROQ_API_KEY=",
        "how": "replace_line",
    },
    "obsidian": {
        "label": "Obsidian Local REST API Key",
        "file": Path.home() / ".claude.json",
        "var": "OBSIDIAN_API_KEY",
        "pattern": '"OBSIDIAN_API_KEY":',
        "how": "claude_json",
    },
}


def show_password_dialog(label):
    """Native macOS password dialog. Key never appears in terminal or stdout."""
    script = f'''
    set theKey to text returned of (display dialog "Enter new {label}" ¬
        default answer "" ¬
        with hidden answer ¬
        buttons {{"Cancel", "Save"}} ¬
        default button "Save" ¬
        with icon note)
    return theKey
    '''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print("Cancelled.")
        sys.exit(1)
    key = result.stdout.strip()
    if not key:
        print("No key entered. Aborting.")
        sys.exit(1)
    return key


def confirm(label, last4):
    """Show confirmation with only last 4 characters visible."""
    script = f'''
    display dialog "Save this key to {label}?\\n\\nKey ending in: ****{last4}" ¬
        buttons {{"Cancel", "Save"}} ¬
        default button "Save" ¬
        with icon caution
    '''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def replace_line_in_file(filepath, pattern, new_line):
    """Replace a line matching pattern. Create it if missing."""
    lines = filepath.read_text().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(pattern.strip()):
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    filepath.write_text("\n".join(lines) + "\n")


def update_deepseek(filepath, key):
    replace_line_in_file(filepath, 'export DEEPSEEK_API_KEY=', f'export DEEPSEEK_API_KEY={key}')


def update_env_var(filepath, var_name, key):
    replace_line_in_file(filepath, f'{var_name}=', f'{var_name}={key}')


def update_claude_json(filepath, key):
    import json
    data = json.loads(filepath.read_text())
    data["mcpServers"]["obsidian"]["env"]["OBSIDIAN_API_KEY"] = key
    filepath.write_text(json.dumps(data, indent=2) + "\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in KEYS:
        print("Usage: python3 update_api_key.py <key-name>")
        print(f"Available keys: {', '.join(KEYS)}")
        sys.exit(1)

    name = sys.argv[1]
    cfg = KEYS[name]

    # Step 1: Secure dialog — key never visible
    key = show_password_dialog(cfg["label"])

    # Step 2: Confirm with last 4 only
    last4 = key[-4:] if len(key) >= 4 else "????"
    if not confirm(cfg["label"], last4):
        print("Cancelled.")
        sys.exit(1)

    # Step 3: Write directly to config file
    if cfg["how"] == "replace_line":
        if name == "deepseek":
            update_deepseek(cfg["file"], key)
        else:
            update_env_var(cfg["file"], cfg["var"], key)
    elif cfg["how"] == "claude_json":
        update_claude_json(cfg["file"], key)

    # Step 4: Confirm — only shows last 4
    print(f"✓ {cfg['label']} updated (ends in ****{last4})")
    print(f"  Written to: {cfg['file']}")

    # Step 5: Reminder if needed
    if name == "deepseek":
        print("  Run: source ~/.zshrc")


if __name__ == "__main__":
    main()
