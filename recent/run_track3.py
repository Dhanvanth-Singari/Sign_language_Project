import argparse
from pathlib import Path
import subprocess
import sys
import os

def run(cmd):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=True)
    return proc.returncode

def main(video, frames_dir, target_img, word_name, fps=25):
    base = Path("backend/track3_frame_generation")
    word_frames = base/"base_action_frames"/word_name
    pose_out = base/"pose"/word_name
    final_out = base/"final_frames"/word_name
    target_path = base/"target_images"/Path(target_img).name

    # ensure dirs
    word_frames.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)
    (base/"target_images").mkdir(parents=True, exist_ok=True)

    if video:
        # extract frames from video
        run([sys.executable, str(base/"tools"/"extract_video_frames.py"),
             "--video", video,
             "--out_dir", str(word_frames),
             "--fps", str(fps)])

    # if user provided frames_dir, copy/symlink is skipped; assume files already in place

    # 1) Extract poses
    run([sys.executable, str(base/"dreampose_like"/"extract_poses.py"),
         "--frames_dir", str(word_frames),
         "--out_dir", str(pose_out)])

    # 2) Inference via ControlNet OpenPose img2img
    prompt = f"a realistic person performing ISL sign for '{word_name}', plain background, high detail"
    run([sys.executable, str(base/"dreampose_like"/"infer_controlnet_img2img.py"),
         "--pose_dir", str(pose_out/"pose_png"),
         "--target_img", str(target_path),
         "--out_dir", str(final_out),
         "--prompt", prompt])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end Track 3 runner (baseline)")
    ap.add_argument("--video", default=None, help="Optional: source video to extract base action frames from" )
    ap.add_argument("--frames_dir", default=None, help="Optional: use an existing frames dir instead of --video" )
    ap.add_argument("--target_img", required=True, help="Target image path (will be copied into target_images/)" )
    ap.add_argument("--word_name", required=True, help="Word label (used for folder names)" )
    ap.add_argument("--fps", type=float, default=25 )
    args = ap.parse_args()

    # If frames_dir is provided, we assume user already placed frames under base_action_frames/word_name
    if args.frames_dir and args.video:
        raise SystemExit("Provide either --video or --frames_dir, not both.")
    main(args.video, args.frames_dir, args.target_img, args.word_name, args.fps)
