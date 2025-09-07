import os
import json
import cv2
import numpy as np

# Input folder (JSON files with landmarks)
pose_json_dir = "pose_debug"

# Output folder (stick figure images)
pose_png_dir = os.path.join("pose", "hello", "pose_png")
os.makedirs(pose_png_dir, exist_ok=True)

# Define body connections (edges between landmarks)
# Using MoveNet's 17 keypoints
BODY_CONNECTIONS = [
    (0, 1), (0, 2),       # Nose to eyes
    (1, 3), (2, 4),       # Eyes to ears
    (0, 5), (0, 6),       # Nose to shoulders
    (5, 7), (7, 9),       # Left arm
    (6, 8), (8, 10),      # Right arm
    (5, 6),               # Shoulders
    (5, 11), (6, 12),     # Torso sides
    (11, 12),             # Hips
    (11, 13), (13, 15),   # Left leg
    (12, 14), (14, 16)    # Right leg
]

IMG_SIZE = 256  # size of output pose image

def draw_pose(landmarks, output_path):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)  # black background
    
    # Draw keypoints
    for x, y, score in landmarks:
        if score > 0.3:  # only draw confident points
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    
    # Draw connections
    for p1, p2 in BODY_CONNECTIONS:
        if landmarks[p1][2] > 0.3 and landmarks[p2][2] > 0.3:
            x1, y1, _ = landmarks[p1]
            x2, y2, _ = landmarks[p2]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    
    cv2.imwrite(output_path, img)

# Process all JSON files
for file in sorted(os.listdir(pose_json_dir)):
    if file.endswith(".json"):
        with open(os.path.join(pose_json_dir, file), "r") as f:
            data = json.load(f)
        
        landmarks = data.get("landmarks", [])
        if not landmarks:
            continue
        
        output_file = os.path.join(pose_png_dir, file.replace(".json", ".png"))
        draw_pose(landmarks, output_file)

print(f"✅ Pose stick figures saved in {pose_png_dir}")
