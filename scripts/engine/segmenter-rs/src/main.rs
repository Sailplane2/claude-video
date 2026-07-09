use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "segmenter")]
#[command(about = "Fast video segmentation using scene detection and motion analysis")]
struct Args {
    /// Path to video file
    #[arg(short, long)]
    input: PathBuf,

    /// Sample interval in seconds (default: 2.0)
    #[arg(short, long, default_value = "2.0")]
    interval: f64,

    /// Scene change threshold 0.0-1.0 (default: 0.35)
    #[arg(short = 't', long, default_value = "0.35")]
    scene_threshold: f64,

    /// Motion boundary threshold 0.0-1.0 (default: 0.30)
    #[arg(short = 'm', long, default_value = "0.30")]
    motion_threshold: f64,

    /// Minimum segment duration in seconds (default: 3.0)
    #[arg(short = 'n', long, default_value = "3.0")]
    min_segment: f64,

    /// Maximum segment duration in seconds (default: 60.0)
    #[arg(short = 'x', long, default_value = "60.0")]
    max_segment: f64,

    /// Pretty-print JSON output
    #[arg(short, long)]
    pretty: bool,
}

fn main() {
    let args = Args::parse();

    let result = match segmenter::segment_video(
        &args.input,
        args.interval,
        args.scene_threshold,
        args.motion_threshold,
        args.min_segment,
        args.max_segment,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {:#}", e);
            std::process::exit(1);
        }
    };

    if args.pretty {
        serde_json::to_writer_pretty(std::io::stdout(), &result).unwrap();
    } else {
        serde_json::to_writer(std::io::stdout(), &result).unwrap();
    }
    println!();
}
