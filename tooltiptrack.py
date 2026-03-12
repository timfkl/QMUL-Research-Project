#!/usr/bin/env python3
"""
ToolTipTrack - Command-line tool for surgical video tooltip tracking.
Uses Lucas-Kanade optical flow with Shi-Tomasi feature detection.
"""

import argparse
from src.tracker import track_lucas_kanade_optical_flow, calculate_statistics, plot_trajectory, export_review_zip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToolTipTrack: Track surgical tool tips in video using Lucas-Kanade optical flow"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--max-seconds", type=int, default=30, help="Maximum seconds to process (default: 30)")
    parser.add_argument("--mode", choices=["flow", "plot", "export", "full"], default="full", 
                        help="Processing mode: flow (track only), plot (visualize), export (save results), full (all)")
    parser.add_argument("--output", default="tracking_review.zip", help="Output ZIP filename (default: tracking_review.zip)")
    parser.add_argument("--no-keyframes", action="store_true", help="Skip keyframe display during tracking")
    
    args = parser.parse_args()
    
    try:
        # Track optical flow
        if args.mode in ["flow", "full"]:
            print(f"🎬 Tracking: {args.video}")
            tracked_frames, centers, frame_count, fps = track_lucas_kanade_optical_flow(
                args.video,
                max_seconds=args.max_seconds,
                show_progress=True,
                show_keyframes=not args.no_keyframes
            )
            stats = calculate_statistics(centers)
        
        # Plot trajectory
        if args.mode in ["plot", "full"]:
            print("📈 Plotting trajectory...")
            plot_trajectory(centers)
        
        # Export results
        if args.mode in ["export", "full"]:
            print("📦 Exporting results...")
            export_review_zip(centers, tracked_frames, frame_count, fps, args.video, args.output)
        
        print("✅ Complete!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
