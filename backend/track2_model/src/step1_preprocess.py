import os, sys, math, argparse, glob
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_videos(input_dir):
    """
    Expects structure: data/raw/<WORD>/<SAMPLE>.mp4  (30 samples per word)
    Returns list of (word, sample_stem, video_path)
    """
    vids = []
    for word in sorted(os.listdir(input_dir)):
        wdir = os.path.join(input_dir, word)
        if not os.path.isdir(wdir): 
            continue
        for vp in sorted(glob.glob(os.path.join(wdir, "*.mp4"))):
            sample = os.path.splitext(os.path.basename(vp))[0]
            vids.append((word, sample, vp))
    return vids

def choose_indices(total_frames, target=60):
    """
    Uniformly sample exactly 'target' indices in [0, total_frames-1].
    If total_frames < target, indices will repeat to pad.
    """
    if total_frames <= 0:
        return []
    idx = np.linspace(0, total_frames - 1, num=target)
    idx = np.rint(idx).astype(int).tolist()
    # Safety clamp
    idx = [min(max(i, 0), total_frames-1) for i in idx]
    return idx

def draw_holistic(frame_bgr, results):
    annotated = frame_bgr.copy()
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    # Hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Face (contours for clarity)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles
                .get_default_face_mesh_contours_style())
    return annotated

def extract_pose_frames(video_path, out_dir, target_frames=60, resize=256, jpg_quality=95):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Try to get total frames quickly
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # Fallback: read and count if unknown
    if total <= 0:
        frames_buf = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames_buf.append(f)
        total = len(frames_buf)
        indices = choose_indices(total, target_frames)
        ensure_dir(out_dir)
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            for i, fi in enumerate(indices):
                frame = frames_buf[fi]
                if resize:
                    frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated = draw_holistic(frame, results)
                cv2.imwrite(os.path.join(out_dir, f"frame_{i:03d}.jpg"),
                            annotated, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        return

    # Normal path: we know total frames
    indices = choose_indices(total, target_frames)
    pick_set = set(indices)
    ensure_dir(out_dir)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        cur = 0
        next_pick_idx = 0
        targets = indices  # preserve order with repeats
        # We read sequentially and process when cur matches next target
        while cur < total:
            ret, frame = cap.read()
            if not ret:
                break
            # process as many targets equal to cur (handles repeats)
            while next_pick_idx < len(targets) and targets[next_pick_idx] == cur:
                proc = frame
                if resize:
                    proc = cv2.resize(proc, (resize, resize), interpolation=cv2.INTER_AREA)
                results = holistic.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                annotated = draw_holistic(proc, results)
                cv2.imwrite(os.path.join(out_dir, f"frame_{next_pick_idx:03d}.jpg"),
                            annotated, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
                next_pick_idx += 1
            cur += 1

    cap.release()

def main():
    ap = argparse.ArgumentParser(description="Extract 60 holistic-annotated frames per video.")
    ap.add_argument("--input_dir",  type=str, default="backend/track2_model/data/raw",
                    help="Folder with videos grouped by word: raw/<WORD>/*.mp4")
    ap.add_argument("--output_dir", type=str, default="backend/track2_model/data/processed",
                    help="Output frames: processed/<WORD>/<SAMPLE>/frame_XXX.jpg")
    ap.add_argument("--frames",     type=int, default=60, help="Frames per video")
    ap.add_argument("--size",       type=int, default=256, help="Output frame size (square)")
    ap.add_argument("--quality",    type=int, default=95, help="JPEG quality 1-100")
    args = ap.parse_args()

    videos = list_videos(args.input_dir)
    if not videos:
        print(f"No videos found in {args.input_dir}. Expected raw/<WORD>/*.mp4", file=sys.stderr)
        sys.exit(1)

    for word, sample, vpath in tqdm(videos, desc="Processing videos"):
        out_dir = os.path.join(args.output_dir, word, sample)
        extract_pose_frames(vpath, out_dir, target_frames=args.frames,
                            resize=args.size, jpg_quality=args.quality)

    print("✅ Done. Pose frames saved under:", args.output_dir)

if __name__ == "__main__":
    main()
