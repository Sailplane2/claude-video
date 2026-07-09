#!/usr/bin/env python3
"""analyze_audio.py — Extract music structure from audio for /style-recon.

Experimental, non-blocking. If librosa is unavailable or analysis fails,
exit 0 gracefully — do not break the research pipeline.

Usage:
    python3 analyze_audio.py --audio audio.mp3 --out-dir /path/to/output
    python3 analyze_audio.py --audio audio.mp3 --out-dir /path/to/output \\
        --video-duration 184.7 --frame-count 80
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract music structure for style recon")
    parser.add_argument("--audio", required=True, help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument("--out-dir", required=True, help="Directory to write audio_analysis.json")
    parser.add_argument("--video-duration", type=float, default=None,
                        help="Video duration in seconds (for frame alignment)")
    parser.add_argument("--frame-count", type=int, default=None,
                        help="Number of extracted frames (for frame alignment)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)

    if not audio_path.exists():
        print(f"[analyze_audio] audio file not found: {audio_path} — skipping", file=sys.stderr)
        sys.exit(0)

    # ---- try to import librosa ------------------------------------------------
    try:
        import librosa
        import numpy as np
    except ImportError:
        print("[analyze_audio] librosa not installed — skipping audio analysis", file=sys.stderr)
        sys.exit(0)

    # ---- load audio -----------------------------------------------------------
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"[analyze_audio] loaded: {duration:.1f}s, sr={sr}, samples={len(y)}")
    except Exception as e:
        print(f"[analyze_audio] failed to load audio: {e} — skipping", file=sys.stderr)
        sys.exit(0)

    if duration < 1.0:
        print("[analyze_audio] audio too short (<1s) — skipping", file=sys.stderr)
        sys.exit(0)

    result = {
        "source": str(audio_path),
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
    }

    # ---- tempo ----------------------------------------------------------------
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        # librosa 0.11 may return tempo as an array
        tempo_val = float(np.atleast_1d(tempo).flatten()[0])
        result["tempo_bpm"] = round(tempo_val, 1)
        result["beats"] = [round(float(b), 3) for b in beats]
        result["beat_count"] = len(beats)
        if beats.size > 1:
            result["mean_beat_interval"] = round(
                float(np.mean(np.diff(beats))), 3
            )
    except Exception as e:
        print(f"[analyze_audio] tempo estimation failed: {e}", file=sys.stderr)

    # ---- onset detection ------------------------------------------------------
    try:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units="time", backtrack=True
        )
        result["onsets"] = [round(float(o), 3) for o in onset_frames]
        result["onset_count"] = len(onset_frames)
    except Exception as e:
        print(f"[analyze_audio] onset detection failed: {e}", file=sys.stderr)

    # ---- energy curve (RMS over 100ms windows) --------------------------------
    try:
        hop_length = int(sr * 0.1)  # 100ms windows
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )
        energy_curve = [
            {"t": round(float(t), 2), "rms": round(float(e), 4)}
            for t, e in zip(rms_times, rms)
        ]
        result["energy_curve"] = energy_curve

        # Downsample to ~10 points/sec for prompt efficiency
        # The full energy_curve is kept for programmatic use;
        # energy_summary is a coarser version for AI prompts
        ds_factor = max(1, len(energy_curve) // 120)  # ~120 points max
        result["energy_summary"] = energy_curve[::ds_factor]

        # Detect energy peaks (local maxima above 75th percentile)
        rms_values = np.array([e["rms"] for e in energy_curve])
        if len(rms_values) > 5:
            rms_median = float(np.median(rms_values))
            rms_75th = float(np.percentile(rms_values, 75))
            result["rms_median"] = round(rms_median, 4)
            result["rms_75th"] = round(rms_75th, 4)

            # Peaks: local maxima above 75th percentile, separated by ≥1s
            peaks = []
            for i in range(2, len(rms_values) - 2):
                window = rms_values[i - 2 : i + 3]
                if rms_values[i] == max(window) and rms_values[i] >= rms_75th:
                    t = energy_curve[i]["t"]
                    if not peaks or t - peaks[-1]["t"] >= 1.0:
                        peaks.append({
                            "t": t,
                            "rms": round(float(rms_values[i]), 4),
                        })
            result["energy_peaks"] = peaks

            # Drops: local minima below 25th percentile, separated by ≥1s
            rms_25th = float(np.percentile(rms_values, 25))
            result["rms_25th"] = round(rms_25th, 4)
            drops = []
            for i in range(2, len(rms_values) - 2):
                window = rms_values[i - 2 : i + 3]
                if rms_values[i] == min(window) and rms_values[i] <= rms_25th:
                    t = energy_curve[i]["t"]
                    if not drops or t - drops[-1]["t"] >= 1.0:
                        drops.append({
                            "t": t,
                            "rms": round(float(rms_values[i]), 4),
                        })
            result["energy_drops"] = drops
    except Exception as e:
        print(f"[analyze_audio] energy analysis failed: {e}", file=sys.stderr)

    # ---- spectral centroid (brightness proxy) ---------------------------------
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_times = librosa.frames_to_time(
            np.arange(len(cent)), sr=sr
        )
        # Downsample
        ds = max(1, len(cent) // 60)
        result["spectral_brightness"] = [
            {"t": round(float(t), 2), "hz": round(float(c), 1)}
            for t, c in zip(cent_times[::ds], cent[::ds])
        ]
    except Exception as e:
        print(f"[analyze_audio] spectral analysis failed: {e}", file=sys.stderr)

    # ---- structural segmentation (energy-curve-based) -------------------------
    # Uses the RMS energy curve to find section boundaries. Avoids librosa's
    # agglomerative clustering (slow on full audio, unstable across versions).
    try:
        if "energy_curve" in result:
            ec = result["energy_curve"]
            rms_vals = np.array([e["rms"] for e in ec])
            times = [e["t"] for e in ec]
            rms_med = float(np.median(rms_vals)) if len(rms_vals) > 0 else 0.01

            # Track energy state transitions that persist ≥2s
            boundaries = [0.0]
            current_state = "medium"
            state_start_idx = 0

            for i in range(len(rms_vals)):
                r = rms_vals[i]
                if r > rms_med * 1.4:
                    new_state = "high"
                elif r < rms_med * 0.5:
                    new_state = "low"
                else:
                    new_state = "medium"

                if new_state != current_state:
                    # Check if this state persists ≥2s
                    state_duration = times[i] - times[state_start_idx] if i > state_start_idx else 0
                    if state_duration >= 2.0:
                        # Confirm new state persists at least 2s forward
                        future_count = min(20, len(rms_vals) - i)  # ~2s at 100ms windows
                        future_vals = rms_vals[i:i + future_count]
                        if future_count >= 10:
                            if new_state == "high":
                                persists = np.mean(future_vals) > rms_med * 1.2
                            elif new_state == "low":
                                persists = np.mean(future_vals) < rms_med * 0.7
                            else:
                                persists = True
                            if persists:
                                t = times[i]
                                if t - boundaries[-1] >= 8.0:  # min 8s gap
                                    boundaries.append(round(t, 2))
                                    state_start_idx = i
                                    current_state = new_state

            boundaries.append(round(duration, 2))

            # Build labeled sections
            sections = []
            for i in range(len(boundaries) - 1):
                start_t = boundaries[i]
                end_t = boundaries[i + 1]
                if end_t - start_t < 4.0:
                    continue

                section_rms = [e["rms"] for e in ec if start_t <= e["t"] <= end_t]
                mean_rms = float(np.mean(section_rms)) if section_rms else 0.0
                ratio = mean_rms / rms_med if rms_med > 0 else 1.0

                if ratio < 0.5:
                    energy_label = "low_energy"
                elif ratio < 0.85:
                    energy_label = "medium_energy"
                elif ratio < 1.3:
                    energy_label = "high_energy"
                else:
                    energy_label = "peak_energy"

                # Detect rising energy (build)
                if len(section_rms) > 5:
                    first_third = np.mean(section_rms[: len(section_rms) // 3])
                    last_third = np.mean(section_rms[-len(section_rms) // 3 :])
                    if last_third > first_third * 1.5:
                        energy_label = "build"

                sections.append({
                    "start": start_t,
                    "end": end_t,
                    "duration": round(end_t - start_t, 2),
                    "mean_rms": round(mean_rms, 4),
                    "label": energy_label,
                })

            result["sections"] = sections
    except Exception as e:
        print(f"[analyze_audio] segmentation failed: {e}", file=sys.stderr)

    # ---- cut alignment candidates (if frame info provided) --------------------
    if args.video_duration and args.frame_count and args.frame_count > 0:
        try:
            # Approximate frame timestamps (linear interpolation)
            frame_times = [
                i * args.video_duration / args.frame_count
                for i in range(args.frame_count)
            ]

            # Collect significant audio events
            events = []
            for peak in result.get("energy_peaks", []):
                events.append({"t": peak["t"], "type": "energy_peak"})
            for drop in result.get("energy_drops", []):
                events.append({"t": drop["t"], "type": "energy_drop"})
            for section in result.get("sections", []):
                if section["label"] in ("build", "high_energy", "peak_energy"):
                    events.append({"t": section["start"], "type": f"section_start_{section['label']}"})

            # Find nearest frame for each event
            candidates = []
            for event in events:
                nearest_idx = min(
                    range(len(frame_times)),
                    key=lambda i: abs(frame_times[i] - event["t"]),
                )
                nearest_time = frame_times[nearest_idx]
                delta = abs(nearest_time - event["t"])
                candidates.append({
                    "audio_time": round(event["t"], 2),
                    "nearest_frame_time": round(nearest_time, 2),
                    "nearest_frame_number": nearest_idx + 1,  # 1-indexed
                    "delta_seconds": round(delta, 2),
                    "event_type": event["type"],
                })

            # Deduplicate and sort by delta (closest alignment first)
            seen = set()
            unique = []
            for c in sorted(candidates, key=lambda x: x["delta_seconds"]):
                key = (c["nearest_frame_number"], c["event_type"])
                if key not in seen:
                    seen.add(key)
                    unique.append(c)

            result["cut_alignment_candidates"] = unique[:30]  # Cap at 30
        except Exception as e:
            print(f"[analyze_audio] alignment computation failed: {e}", file=sys.stderr)

    # ---- write output ---------------------------------------------------------
    out_path = out_dir / "audio_analysis.json"
    try:
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[analyze_audio] wrote {out_path}")
        print(f"[analyze_audio] summary: tempo={result.get('tempo_bpm', '?')}bpm, "
              f"beats={result.get('beat_count', '?')}, "
              f"onsets={result.get('onset_count', '?')}, "
              f"peaks={len(result.get('energy_peaks', []))}, "
              f"drops={len(result.get('energy_drops', []))}, "
              f"sections={len(result.get('sections', []))}"
              + (f", alignment_candidates={len(result.get('cut_alignment_candidates', []))}"
                 if args.frame_count else ""))
    except Exception as e:
        print(f"[analyze_audio] failed to write output: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
