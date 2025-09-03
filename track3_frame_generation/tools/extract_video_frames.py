import os
import cv2
import argparse
from pathlib import Path

def extract(video_path: str, out_dir: str, fps: float=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1
    if fps and fps > 0:
        step = max(int(round(src_fps / fps)), 1)
    idx = 0
    out_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{out_idx:04d}.jpg"), frame)
            out_idx += 1
        idx += 1
    cap.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input .mp4 path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=None, help="Target FPS (optional)" )
    args = ap.parse_args()
    extract(args.video, args.out_dir, args.fps)
