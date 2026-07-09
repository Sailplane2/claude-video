#!/usr/bin/env python3
"""Read text from a screenshot using Gemini 2.5 Flash."""
import json, sys, base64, urllib.request
from pathlib import Path

def load_key():
    env = Path.home() / ".config/watch/.env"
    for line in env.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found")

img = Path(sys.argv[1])
b64 = base64.b64encode(img.read_bytes()).decode()
key = load_key()

body = json.dumps({
    "contents": [{"parts": [
        {"text": "Read all text visible in this screenshot. Transcribe every line exactly, including terminal output, error messages, and UI text."},
        {"inlineData": {"mimeType": "image/png", "data": b64}}
    ]}],
    "generationConfig": {"maxOutputTokens": 2048}
}).encode()

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read())
print(data["candidates"][0]["content"]["parts"][0]["text"])
