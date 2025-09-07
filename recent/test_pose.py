import os
import cv2
import numpy as np
import tensorflow as tf
import json

# Paths
frames_folder = r"D:\BTP-20250820T120438Z-1-001\track3\Sign_language_Project\track3_frame_generation\base_action_frames\hello"
tflite_model_path = "movenet_singlepose_lightning.tflite"  # ensure this file is in same folder
output_json_path = "pose_keypoints.json"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# COCO keypoint order mapping
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

all_frames_keypoints = {}

# Process each frame
for file_name in sorted(os.listdir(frames_folder)):
    if not file_name.endswith(".jpg"):
        continue

    image_path = os.path.join(frames_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not load {image_path}")
        continue

    # Resize and normalize
    input_image = cv2.resize(image, (192, 192))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Extract keypoints
    keypoints = keypoints_with_scores[0, 0, :, :]  # (17, 3)
    frame_data = {}

    for i, name in enumerate(keypoint_names):
        y, x, confidence = keypoints[i]
        frame_data[name] = [float(x), float(y), float(confidence)]  # store x,y,conf

    # Save for this frame
    all_frames_keypoints[file_name] = frame_data
    print(f"✅ Processed {file_name}")

# Save all to JSON
with open(output_json_path, "w") as f:
    json.dump(all_frames_keypoints, f, indent=2)

print(f"\n🎉 Saved keypoints for {len(all_frames_keypoints)} frames to {output_json_path}")
