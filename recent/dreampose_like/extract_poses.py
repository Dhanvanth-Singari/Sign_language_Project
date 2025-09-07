import os
import argparse
from pathlib import Path
import json
import mediapipe as mp
import cv2
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def extract_keypoints(image_bgr):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        # Collect pose + hand landmarks
        def lm_to_list(lms):
            if not lms:
                return []
            return [{"x":lm.x, "y":lm.y, "z":getattr(lm,'z',0.0), "v":getattr(lm,'visibility',1.0)} for lm in lms.landmark]
        data = {
            "pose": lm_to_list(results.pose_landmarks),
            "left_hand": lm_to_list(results.left_hand_landmarks),
            "right_hand": lm_to_list(results.right_hand_landmarks),
            "face": lm_to_list(results.face_landmarks)  # optional
        }
        return data, results

def draw_skeleton(image_bgr, results):
    image = image_bgr.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image

def main(frames_dir, out_dir):
    frames = sorted([p for p in Path(frames_dir).iterdir()
                     if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    json_dir = Path(out_dir) / "pose_json"
    png_dir = Path(out_dir) / "pose_png"
    json_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(tqdm(frames, desc="Extracting poses"), start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        data, results = extract_keypoints(img)
        # save json
        with open(json_dir / f"{i:04d}.json", "w") as f:
            json.dump(data, f)
        # save preview skeleton image (for ControlNet OpenPose you may also use controlnet-aux detector)
        skel = draw_skeleton(img, results)
        cv2.imwrite(str(png_dir / f"{i:04d}.png"), skel)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Directory of base action frames for a word" )
    ap.add_argument("--out_dir", required=True, help="Output dir (pose_json + pose_png)" )
    args = ap.parse_args()
    main(args.frames_dir, args.out_dir)
