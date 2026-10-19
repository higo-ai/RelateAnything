"""
VidVRD RelateAnything End-to-End Pipeline Runner
================================================
Unified single-entry CLI runner for video visual relationship detection.

Usage:
    python run.py --video data/videos/video1.mp4
    python run.py --video data/videos/video2.mp4 --device cuda
    python run.py --video path/to/your_video.mp4 --conf 0.35
"""

import os
import sys
import argparse
import subprocess
import json


def print_banner():
    banner = """
================================================================================
          VidVRD RelateAnything: End-to-End Video Relation Detection
================================================================================
    """
    print(banner)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VidVRD RelateAnything End-to-End Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input surveillance video MP4 (e.g. data/videos/video1.mp4)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence threshold (default: 0.35)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device: 'auto', 'cuda', or 'cpu' (default: auto)")
    parser.add_argument("--output_subdir", type=str, default=None, help="Custom output subdirectory under data/output/")
    parser.add_argument("--full_video", action="store_true", default=True, help="Process full video from 0.0s to duration (default: True)")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file does not exist: '{args.video}'")
        sys.exit(1)

    python_exe = sys.executable
    pipeline_script = os.path.join(os.path.dirname(__file__), "pipeline_relate_anything.py")

    cmd = [
        python_exe, pipeline_script,
        "--video", args.video,
        "--mode", "roi",
        "--conf", str(args.conf),
        "--device", args.device,
        "--full_video"
    ]
    if args.output_subdir:
        cmd.extend(["--output_subdir", args.output_subdir])

    print(f"[RUNNER] Launching pipeline execution:")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join("data", "output", "full_video", video_basename) if not args.output_subdir else os.path.join("data", "output", args.output_subdir, video_basename)
    json_path = os.path.join(out_dir, f"{video_basename}_relations.json")
    mp4_path = os.path.join(out_dir, f"{video_basename}_visualizer.mp4")

    print("\n" + "=" * 80)
    print("                      EXECUTION SUMMARY & RESULTS")
    print("=" * 80)
    print(f"  Target Video:     {args.video}")
    print(f"  Output JSON:      {json_path}")
    print(f"  Visualizer MP4:   {mp4_path}")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tracks = data.get("tracks", data.get("entities", {}))
            relations = data.get("relations", [])
            print(f"  Total Tracks:     {len(tracks)}")
            print(f"  Total Relations:  {len(relations)}")
            if relations:
                print("\n  Confirmed Visual Relationships:")
                print(f"  {'#':<3} | {'Subject':<18} | {'Predicate':<12} | {'Object':<18} | {'Conf':<6} | {'Time Interval':<16}")
                print("  " + "-" * 76)
                for idx, r in enumerate(relations, 1):
                    sub_str = f"{r['subject_id']} ({r.get('subject_class', '')})"
                    obj_str = f"{r['object_id']} ({r.get('object_class', '')})"
                    t_range = r.get("time_range", {})
                    t_str = f"{t_range.get('start_sec', 0.0):.1f}s -> {t_range.get('end_sec', 0.0):.1f}s"
                    print(f"  {idx:<3} | {sub_str:<18} | {r['predicate']:<12} | {obj_str:<18} | {r.get('confidence', 0.0):<6.2f} | {t_str:<16}")
        except Exception as e:
            print(f"  [WARN] Could not parse relations JSON summary: {e}")

    print("\n" + "=" * 80)
    print("Run completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
