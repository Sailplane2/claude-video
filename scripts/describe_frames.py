#!/usr/bin/env python3
"""Describe video frames using Gemini 2.5 Flash — parallel requests.

Usage:
    python3 describe_frames.py <frames-dir> [--out-dir <path>]

API key loaded from GEMINI_API_KEY in ~/.config/watch/.env.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPT = (
    "Describe this video frame in 2-3 sentences. "
    "What is shown? What is the setting? Any people, text on screen, "
    "or notable visual elements? Be concise and factual."
)
MODEL = "gemini-2.5-flash"
MAX_TOKENS = 2048


def load_env():
    env_file = Path.home() / ".config/watch/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    return v.strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found in ~/.config/watch/.env")


def b64_encode(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_one(frame_path, api_key):
    b64 = b64_encode(frame_path)
    body = json.dumps({
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64}}
            ]
        }],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS}
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, None
    except Exception as e:
        return None, str(e)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 describe_frames.py <frames-dir> [--out-dir <path>]", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(sys.argv[1]).resolve()
    out_dir = frames_dir
    if "--out-dir" in sys.argv:
        idx = sys.argv.index("--out-dir")
        if idx + 1 < len(sys.argv):
            out_dir = Path(sys.argv[idx + 1]).resolve()

    if not frames_dir.is_dir():
        print(f"Frames directory not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    api_key = load_env()

    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No .jpg or .png frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Describing {len(frames)} frames in parallel with {MODEL}…", file=sys.stderr)

    t0 = time.time()
    results = [None] * len(frames)

    with ThreadPoolExecutor(max_workers=80) as pool:
        fut_map = {pool.submit(describe_one, f, api_key): i for i, f in enumerate(frames)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            text, err = fut.result()
            ts = frames[i].stem.split("_t")[-1].replace("-", ":") if "_t" in frames[i].stem else ""
            results[i] = (frames[i].name, ts, text, err)
            name = frames[i].name
            if err:
                sys.stderr.write(f"  x {name}: {err}\n")
            else:
                sys.stderr.write(f"  ✓ {name}\n")

    elapsed = time.time() - t0

    out_path = out_dir / "frame_descriptions.txt"
    lines = []
    for name, ts, text, err in results:
        header = name
        if ts:
            header += f" (t={ts})"
        lines.append(f"--- {header} ---")
        lines.append(text if text else f"[ERROR] {err}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nDone — {len(frames)} frames in {elapsed:.1f}s → {out_path}", file=sys.stderr)
    print(out_path)


if __name__ == "__main__":
    main()
