use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

// ── Output types (match Python engine schemas) ──────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub index: usize,
    pub in_seconds: f64,
    pub out_seconds: f64,
    pub duration_seconds: f64,
    pub motion_intensity: f64,
    pub is_scene_boundary: bool,
    pub is_motion_boundary: bool,
}

#[derive(Debug, Serialize)]
pub struct SegmentResult {
    pub clip_path: String,
    pub duration_seconds: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub total_segments: usize,
    pub segments: Vec<Segment>,
}

// ── Histogram helpers ───────────────────────────────────────────────

const HIST_BINS: usize = 64;

fn compute_histogram(data: &[u8], stride: usize, width: usize, height: usize) -> [f64; HIST_BINS] {
    let mut hist = [0.0f64; HIST_BINS];
    let total = (width * height) as f64;
    if total == 0.0 {
        return hist;
    }
    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x * 3;
            let r = data[idx] as f64;
            let g = data[idx + 1] as f64;
            let b = data[idx + 2] as f64;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            let bin = ((lum / 255.0) * (HIST_BINS as f64 - 1.0)) as usize;
            hist[bin.min(HIST_BINS - 1)] += 1.0;
        }
    }
    for v in hist.iter_mut() {
        *v /= total;
    }
    hist
}

fn histogram_difference(a: &[f64; HIST_BINS], b: &[f64; HIST_BINS]) -> f64 {
    let mut diff = 0.0;
    for i in 0..HIST_BINS {
        diff += (a[i] - b[i]).abs();
    }
    diff / 2.0 // normalised 0..1
}

// ── Core segmentation ───────────────────────────────────────────────

pub fn segment_video(
    path: &Path,
    sample_interval_secs: f64,
    scene_threshold: f64,
    motion_threshold: f64,
    min_segment_secs: f64,
    max_segment_secs: f64,
) -> Result<SegmentResult> {
    let mut input = ffmpeg_next::format::input(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;

    // Extract all metadata inside a block so video_stream is dropped before the loop
    let (stream_idx, duration_secs, width, height, fps, time_base_f64, sample_stride, mut decoder, mut scaler) = {
        let video_stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("No video stream found")?;

        let stream_idx = video_stream.index();
        let duration_secs = video_stream.duration() as f64
            * f64::from(video_stream.time_base());
        let fps = f64::from(video_stream.rate());
        let time_base_f64 = f64::from(video_stream.time_base());

        let mut decoder =
            ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters())?
                .decoder()
                .video()
                .context("Failed to create video decoder")?;

        let width = decoder.width();
        let height = decoder.height();

        let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )?;

        let sample_stride = (sample_interval_secs * time_base_f64.recip()) as i64;
        let sample_stride = sample_stride.max(1);

        Ok::<_, anyhow::Error>((stream_idx, duration_secs, width, height, fps, time_base_f64, sample_stride, decoder, scaler))
    }?; // video_stream dropped here — borrow on input released

    let mut sampled_histograms: Vec<(f64, [f64; HIST_BINS])> = Vec::new();
    let mut frame_count: i64 = 0;

    for (stream, packet) in input.packets() {
        if stream.index() != stream_idx {
            continue;
        }
        if frame_count % sample_stride == 0 {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::frame::Video::empty();
            if decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb = ffmpeg_next::frame::Video::empty();
                scaler.run(&decoded, &mut rgb)?;

                let timestamp = decoded.timestamp().map_or(0.0, |t| t as f64 * time_base_f64);

                let hist = compute_histogram(
                    rgb.data(0),
                    rgb.stride(0),
                    width as usize,
                    height as usize,
                );
                sampled_histograms.push((timestamp, hist));
            }
        } else {
            // Still need to feed the decoder even when skipping
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg_next::frame::Video::empty();
            let _ = decoder.receive_frame(&mut decoded);
        }
        frame_count += 1;
    }

    // Flush decoder
    decoder.send_eof()?;
    loop {
        let mut decoded = ffmpeg_next::frame::Video::empty();
        if decoder.receive_frame(&mut decoded).is_err() {
            break;
        }
        if frame_count % sample_stride == 0 {
            let mut rgb = ffmpeg_next::frame::Video::empty();
            scaler.run(&decoded, &mut rgb)?;
            let timestamp = decoded.timestamp().map_or(0.0, |t| t as f64 * time_base_f64);
            let hist = compute_histogram(
                rgb.data(0),
                rgb.stride(0),
                width as usize,
                height as usize,
            );
            sampled_histograms.push((timestamp, hist));
        }
        frame_count += 1;
    }

    // Detect scene boundaries from histogram differences
    let mut boundary_times: Vec<f64> = vec![0.0];
    let mut motion_intensities: Vec<f64> = Vec::new();

    for i in 1..sampled_histograms.len() {
        let (_t0, ref h0) = sampled_histograms[i - 1];
        let (t1, ref h1) = sampled_histograms[i];
        let diff = histogram_difference(h0, h1);
        motion_intensities.push(diff);
        if diff > scene_threshold {
            boundary_times.push(t1);
        }
    }

    boundary_times.push(duration_secs);
    boundary_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    boundary_times.dedup_by(|a, b| (*a - *b).abs() < 1.0);

    // Enforce min/max segment duration
    let mut merged: Vec<f64> = vec![boundary_times[0]];
    for &t in &boundary_times[1..] {
        let last = *merged.last().unwrap();
        let candidate_dur = t - last;
        if candidate_dur < min_segment_secs && t != *boundary_times.last().unwrap() {
            continue; // skip — too short, merge with next
        }
        if candidate_dur > max_segment_secs {
            // Split into equal pieces
            let n = (candidate_dur / max_segment_secs).ceil() as usize;
            for k in 1..n {
                merged.push(last + (candidate_dur * k as f64 / n as f64));
            }
        }
        merged.push(t);
    }
    merged.dedup_by(|a, b| (*a - *b).abs() < 0.5);

    // Build segments
    let mut segments = Vec::new();
    for i in 0..merged.len() - 1 {
        let in_sec = merged[i];
        let out_sec = merged[i + 1];
        let seg_motion = motion_intensities
            .iter()
            .filter(|&&m| m > 0.0)
            .fold(0.0f64, |a, &b| a + b)
            / motion_intensities.len().max(1) as f64;

        segments.push(Segment {
            index: i,
            in_seconds: in_sec,
            out_seconds: out_sec,
            duration_seconds: out_sec - in_sec,
            motion_intensity: seg_motion.min(1.0),
            is_scene_boundary: i == 0
                || boundary_times.iter().any(|&t| (t - in_sec).abs() < 0.5),
            is_motion_boundary: seg_motion > motion_threshold,
        });
    }

    Ok(SegmentResult {
        clip_path: path.display().to_string(),
        duration_seconds: duration_secs,
        width,
        height,
        fps,
        total_segments: segments.len(),
        segments,
    })
}
