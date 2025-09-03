import os, sys, argparse, glob
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------- utils ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def list_videos(input_dir):
    """
    Expects structure: data/raw/<WORD>/<SAMPLE>.mp4
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
    If total_frames < target, indices will repeat.
    """
    if total_frames <= 0:
        return []
    idx = np.linspace(0, total_frames - 1, num=target)
    idx = np.rint(idx).astype(int).tolist()
    return [min(max(i, 0), total_frames - 1) for i in idx]

# ---------- drawing ----------
def draw_holistic(frame_bgr, results):
    annotated = frame_bgr.copy()

    # Custom specs (smaller size, thinner lines)
    pose_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
    hand_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
    face_spec = mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)

    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_spec, connection_drawing_spec=pose_spec)

    # Hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_spec)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_spec)

    # Face
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=face_spec, connection_drawing_spec=face_spec)

    return annotated

# ---------- processing ----------
def crop_hand(frame, landmarks, margin=20, crop_size=128):
    """Crop region around hand landmarks, return cropped+resized frame."""
    if not landmarks:
        return None
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        return cv2.resize(crop, (crop_size, crop_size))
    return None

def extract_pose_frames(video_path, out_dir, target_frames=60, body_size=256,
                        hand_size=128, jpg_quality=95):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = choose_indices(total, target_frames)

    # Create subfolders
    body_dir = ensure_dir(os.path.join(out_dir, "body"))
    lhand_dir = ensure_dir(os.path.join(out_dir, "left_hand"))
    rhand_dir = ensure_dir(os.path.join(out_dir, "right_hand"))

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        cur = 0
        next_pick_idx = 0
        while cur < total:
            ret, frame = cap.read()
            if not ret:
                break

            while next_pick_idx < len(indices) and indices[next_pick_idx] == cur:
                # --- Body frame ---
                proc_body = cv2.resize(frame, (body_size, body_size))
                results = holistic.process(cv2.cvtColor(proc_body, cv2.COLOR_BGR2RGB))
                annotated_body = draw_holistic(proc_body, results)
                cv2.imwrite(os.path.join(body_dir, f"{next_pick_idx:03d}.jpg"),
                            annotated_body, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

                # --- Left hand ---
                if results.left_hand_landmarks:
                    lcrop = crop_hand(proc_body, results.left_hand_landmarks, crop_size=hand_size)
                    if lcrop is not None:
                        # run holistic again on cropped hand for drawing
                        res_l = holistic.process(cv2.cvtColor(lcrop, cv2.COLOR_BGR2RGB))
                        ann_l = draw_holistic(lcrop, res_l)
                        cv2.imwrite(os.path.join(lhand_dir, f"{next_pick_idx:03d}.jpg"),
                                    ann_l, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

                # --- Right hand ---
                if results.right_hand_landmarks:
                    rcrop = crop_hand(proc_body, results.right_hand_landmarks, crop_size=hand_size)
                    if rcrop is not None:
                        res_r = holistic.process(cv2.cvtColor(rcrop, cv2.COLOR_BGR2RGB))
                        ann_r = draw_holistic(rcrop, res_r)
                        cv2.imwrite(os.path.join(rhand_dir, f"{next_pick_idx:03d}.jpg"),
                                    ann_r, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

                next_pick_idx += 1
            cur += 1

    cap.release()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Extract body+hand holistic frames per video.")
    ap.add_argument("--input_dir",  type=str, default="backend/track2_model/data/raw",
                    help="Folder with videos grouped by word: raw/<WORD>/*.mp4")
    ap.add_argument("--output_dir", type=str, default="backend/track2_model/data/processed",
                    help="Output frames: processed/<WORD>/<SAMPLE>/{body,left_hand,right_hand}")
    ap.add_argument("--frames",     type=int, default=60, help="Frames per video")
    ap.add_argument("--body_size",  type=int, default=256, help="Resize for full body frames")
    ap.add_argument("--hand_size",  type=int, default=128, help="Resize for cropped hands")
    ap.add_argument("--quality",    type=int, default=95, help="JPEG quality 1-100")
    args = ap.parse_args()

    videos = list_videos(args.input_dir)
    if not videos:
        print(f"No videos found in {args.input_dir}. Expected raw/<WORD>/*.mp4", file=sys.stderr)
        sys.exit(1)

    for word, sample, vpath in tqdm(videos, desc="Processing videos"):
        out_dir = os.path.join(args.output_dir, word, sample)
        extract_pose_frames(vpath, out_dir, target_frames=args.frames,
                            body_size=args.body_size, hand_size=args.hand_size,
                            jpg_quality=args.quality)

    print("✅ Done. Frames saved under:", args.output_dir)

if __name__ == "__main__":
    main()
