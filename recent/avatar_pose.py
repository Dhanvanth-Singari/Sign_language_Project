import os
import cv2
import json

# Paths
AVATAR_PATH = "avatar.png"
KEYPOINTS_FILE = "pose_keypoints.json"   # single JSON file
OUTPUT_DIR = "avatar_output"

# Skeleton connections (MoveNet 17 points)
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]

# Ensure output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load avatar
avatar = cv2.imread(AVATAR_PATH)
if avatar is None:
    raise FileNotFoundError(f"Avatar image not found: {AVATAR_PATH}")
h, w, _ = avatar.shape

# Load all keypoints
with open(KEYPOINTS_FILE, "r") as f:
    all_keypoints = json.load(f)

def draw_pose(img, keypoints):
    """Draw skeleton on avatar using normalized coords."""
    for kp_name, (y, x, conf) in keypoints.items():
        if conf > 0.2:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    for a, b in SKELETON:
        if a in keypoints and b in keypoints:
            y1, x1, c1 = keypoints[a]
            y2, x2, c2 = keypoints[b]
            if c1 > 0.2 and c2 > 0.2:
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.line(img, p1, p2, (255, 0, 0), 2)

# Process each frame in the JSON
for frame_name, keypoints in all_keypoints.items():
    frame = avatar.copy()
    draw_pose(frame, keypoints)

    out_path = os.path.join(OUTPUT_DIR, f"avatar_{frame_name}")
    cv2.imwrite(out_path, frame)
    print(f"✅ Saved {out_path}")

print("🎉 All avatar animation frames saved in", OUTPUT_DIR)
