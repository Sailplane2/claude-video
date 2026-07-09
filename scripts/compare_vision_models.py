#!/usr/bin/env python3
"""Compare frame descriptions across vision models for /watch-save pipeline."""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
# Load API keys from ~/.config/watch/.env
ENV_FILE = Path.home() / ".config/watch/.env"
API_KEYS = {}
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            API_KEYS[k.strip()] = v.strip().strip('"').strip("'")

OPENAI_KEY = API_KEYS.get("OPENAI_API_KEY", "")
GEMINI_KEY = API_KEYS.get("GEMINI_API_KEY", "")
# DeepSeek key from environment (loaded via .zshrc export)
import os
DEEPSEEK_KEY = API_KEYS.get("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

# oMLX auth — typically "local" when running locally
OMLX_KEY = os.environ.get("OMLX_API_KEY", "local")

# ── Helpers ─────────────────────────────────────────────────────────

def b64_encode(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openai(image_path, model="gpt-4o-mini", max_tokens=200):
    """Call OpenAI vision API for frame description."""
    import urllib.request

    b64 = b64_encode(image_path)
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video frame in 2-3 sentences. "
                 "What is shown? What is the setting? Any people, text on screen, "
                 "or notable visual elements? Be concise and factual."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json"
        }
    )

    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    elapsed = time.time() - t0

    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return text.strip(), elapsed, tokens


def call_gemini(image_path, model="gemini-2.5-flash", max_tokens=2048):
    """Call Google Gemini Flash vision API — free tier (1,500 req/day)."""
    import urllib.request

    b64 = b64_encode(image_path)
    body = json.dumps({
        "contents": [{
            "parts": [
                {"text": "Describe this video frame in 2-3 sentences. "
                 "What is shown? What is the setting? Any people, text on screen, "
                 "or notable visual elements? Be concise and factual."},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64}}
            ]
        }],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    elapsed = time.time() - t0

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    tokens = data.get("usageMetadata", {}).get("totalTokenCount", "?")
    return text.strip(), elapsed, tokens


def call_omlx(image_path, model="gemma4", max_tokens=200):
    """Call local oMLX server for frame description."""
    import urllib.request

    # oMLX runs at localhost:8000 with OpenAI-compatible API
    b64 = b64_encode(image_path)
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video frame in 2-3 sentences. "
                 "What is shown? What is the setting? Any people, text on screen, "
                 "or notable visual elements? Be concise and factual."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OMLX_KEY}"
        }
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        elapsed = time.time() - t0
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", "?")
        return text.strip(), elapsed, tokens
    except Exception as e:
        return f"[ERROR] {e}", time.time() - t0, 0


def call_deepseek(image_path):
    """DeepSeek cannot process images — verify rejection."""
    import urllib.request

    b64 = b64_encode(image_path)
    body = json.dumps({
        "model": "deepseek-v4-pro",
        "max_tokens": 200,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/anthropic/v1/messages",
        data=body,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
            "Content-Type": "application/json",
            "x-api-key": DEEPSEEK_KEY if DEEPSEEK_KEY else "",
        "anthropic-version": "2023-06-01"
        }
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        elapsed = time.time() - t0
        return f"[UNEXPECTED SUCCESS] {json.dumps(data)[:200]}", elapsed, 0
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        body_text = e.read().decode()[:300]
        return f"[REJECTED] HTTP {e.code}: {body_text}", elapsed, 0


# ── Main ────────────────────────────────────────────────────────────

def compare(frame_path):
    frame = Path(frame_path)
    if not frame.exists():
        print(f"Frame not found: {frame_path}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"FRAME: {frame.name} ({frame.stat().st_size:,} bytes)")
    print(f"{'='*70}")

    # 1. Gemini Flash (free tier)
    print(f"\n── Gemini 2.5 Flash ──")
    if GEMINI_KEY.strip():
        try:
            desc, elapsed, tokens = call_gemini(frame)
            print(f"  Time: {elapsed:.1f}s | Tokens: {tokens} | Cost: $0.00")
            print(f"  {desc}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print("  [SKIP] No GEMINI_API_KEY in ~/.config/watch/.env")

    # 2. GPT-4o-mini (fallback)
    print(f"\n── GPT-4o-mini ($0.15/M input) ──")
    if OPENAI_KEY.strip():
        try:
            desc, elapsed, tokens = call_openai(frame)
            cost = (tokens / 1_000_000) * 0.15
            print(f"  Time: {elapsed:.1f}s | Tokens: {tokens} | Cost: ${cost:.6f}")
            print(f"  {desc}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print("  [SKIP] No OPENAI_API_KEY")

    # 3. Local model via oMLX
    print(f"\n── Local (oMLX) ──")
    try:
        desc, elapsed, tokens = call_omlx(frame)
        print(f"  Time: {elapsed:.1f}s | Tokens: {tokens} | Cost: $0.00")
        print(f"  {desc}")
    except Exception as e:
        print(f"  [SKIP] oMLX not running or not available: {e}")

    # 3. DeepSeek (expected to fail — verify)
    print(f"\n── DeepSeek (text-only — expected rejection) ──")
    if DEEPSEEK_KEY.strip():
        try:
            desc, elapsed, _ = call_deepseek(frame)
            print(f"  Time: {elapsed:.1f}s")
            print(f"  {desc}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print("  [SKIP] No DEEPSEEK_API_KEY in environment")

    print(f"\n{'='*70}")
    print("CONCLUSION: Use the cheapest model with acceptable quality.")
    print("If local model descriptions are >80% as good as GPT-4o-mini, use local.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 compare_vision_models.py <frame.jpg>")
        print("")
        print("Compares frame descriptions from:")
        print("  1. GPT-4o-mini (OpenAI API)")
        print("  2. Local model via oMLX (free)")
        print("  3. DeepSeek (text-only — should reject)")
        print("")
        print("API keys loaded from ~/.config/watch/.env")
        print("Requires oMLX running: omlx serve --port 8000")
        sys.exit(1)

    compare(sys.argv[1])
