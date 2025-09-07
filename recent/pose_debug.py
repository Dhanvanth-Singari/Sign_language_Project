import os
import cv2
import numpy as np
import tensorflow as tf
import json

frames_folder = r"D:\BTP-20250820T120438Z-1-001\track3\Sign_language_Project\track3_frame_generation\base_action_frames\hello"
tflite_model_path = "movenet_singlepose_lightning.tflite"
output_json_path = "pose_keypoints.json"
debug_out_folder = "pose_debug"

os.makedirs(debug_out_folder, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Pairs of joints to connect (COCO skeleton)
skeleton_pairs = [
    ("left_eye", "nose"), ("right_eye", "nose"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle")
]

all_frames_keypoints = {}

for file_name in sorted(os.listdir(frames_folder)):
    if not file_name.endswith(".jpg"):
        continue

    image_path = os.path.join(frames_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not load {image_path}")
        continue

    h, w, _ = image.shape

    # Resize input for MoveNet
    input_image = cv2.resize(image, (192, 192))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    keypoints = keypoints_with_scores[0, 0, :, :]  # (17, 3)
    frame_data = {}

    for i, name in enumerate(keypoint_names):
        y, x, confidence = keypoints[i]
        # scale back to original image size
        abs_x, abs_y = int(x * w), int(y * h)
        frame_data[name] = [float(x), float(y), float(confidence)]

        if confidence > 0.2:  # draw only reliable points
            cv2.circle(image, (abs_x, abs_y), 4, (0, 255, 0), -1)

    # Draw skeleton lines
    for p1, p2 in skeleton_pairs:
        if frame_data[p1][2] > 0.2 and frame_data[p2][2] > 0.2:
            x1, y1 = int(frame_data[p1][0] * w), int(frame_data[p1][1] * h)
            x2, y2 = int(frame_data[p2][0] * w), int(frame_data[p2][1] * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save debug frame
    debug_path = os.path.join(debug_out_folder, file_name)
    cv2.imwrite(debug_path, image)
    print(f"✅ Saved debug frame {debug_path}")

    all_frames_keypoints[file_name] = frame_data

# Save JSON
with open(output_json_path, "w") as f:
    json.dump(all_frames_keypoints, f, indent=2)

print(f"\n🎉 Done! Saved JSON + debug frames to {debug_out_folder}")
