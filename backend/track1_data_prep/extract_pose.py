import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
from scipy import interpolate

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def interpolate_missing_landmarks(landmarks_sequence, frame_indices):
    """
    Interpolate missing landmarks for better temporal consistency
    """
    if len(landmarks_sequence) < 2:
        return landmarks_sequence
    
    # Convert to numpy array for easier processing
    valid_frames = []
    valid_landmarks = []
    
    for i, landmarks in enumerate(landmarks_sequence):
        if landmarks is not None:
            valid_frames.append(frame_indices[i])
            # Convert landmarks to array
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            valid_landmarks.append(landmark_array)
    
    if len(valid_landmarks) < 2:
        return landmarks_sequence
    
    # Interpolate missing frames
    valid_frames = np.array(valid_frames)
    valid_landmarks = np.array(valid_landmarks)
    
    interpolated_sequence = []
    for target_frame in frame_indices:
        if target_frame in valid_frames:
            # Use existing landmark
            idx = np.where(valid_frames == target_frame)[0][0]
            interpolated_sequence.append(valid_landmarks[idx])
        else:
            # Interpolate
            if target_frame < valid_frames[0]:
                interpolated_sequence.append(valid_landmarks[0])
            elif target_frame > valid_frames[-1]:
                interpolated_sequence.append(valid_landmarks[-1])
            else:
                # Linear interpolation
                f = interpolate.interp1d(valid_frames, valid_landmarks, axis=0, kind='linear')
                interpolated_landmark = f(target_frame)
                interpolated_sequence.append(interpolated_landmark)
    
    return interpolated_sequence

def extract_frames_60fps(video_path, output_root):
    """
    Enhanced frame extraction with 60fps output and better detection logic
    """
    os.makedirs(output_root, exist_ok=True)
    body_path = os.path.join(output_root, "body_frames")
    left_hand_path = os.path.join(output_root, "left_hand_frames")
    right_hand_path = os.path.join(output_root, "right_hand_frames")

    os.makedirs(body_path, exist_ok=True)
    os.makedirs(left_hand_path, exist_ok=True)
    os.makedirs(right_hand_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original FPS: {original_fps}, Total frames: {total_frames}")
    
    # Target 60fps - calculate frame duplication factor
    target_fps = 60
    frame_multiplier = target_fps / original_fps
    
    # Storage for temporal consistency
    pose_history = []
    left_hand_history = []
    right_hand_history = []
    frame_data = []
    
    frame_count = 0
    
    # Enhanced MediaPipe settings for better detection
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,  # Higher complexity for better accuracy
        smooth_landmarks=True,  # Enable smoothing
        enable_segmentation=False,  # Disable if not needed for speed
        smooth_segmentation=False,
        refine_face_landmarks=False,  # Disable if not needed for speed
        min_detection_confidence=0.5,  # Lower threshold for better detection
        min_tracking_confidence=0.5   # Lower threshold for better tracking
    ) as holistic:
        
        # First pass: collect all frame data
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            frame_data.append({
                'frame': frame.copy(),
                'pose': results.pose_landmarks,
                'left_hand': results.left_hand_landmarks,
                'right_hand': results.right_hand_landmarks,
                'pose_confidence': getattr(results.pose_landmarks, 'landmark', None),
                'left_confidence': getattr(results.left_hand_landmarks, 'landmark', None),
                'right_confidence': getattr(results.right_hand_landmarks, 'landmark', None)
            })
            
            frame_count += 1

    cap.release()
    
    # Post-process for better consistency and interpolation
    output_frame_count = 0
    
    for i in range(len(frame_data)):
        current_data = frame_data[i]
        
        # Calculate how many frames to generate for this input frame
        frames_to_generate = int(frame_multiplier)
        if i == len(frame_data) - 1:  # Last frame
            frames_to_generate = max(1, frames_to_generate)
        
        for sub_frame in range(frames_to_generate):
            frame = current_data['frame']
            
            # BODY PROCESSING with enhanced logic
            if current_data['pose'] is not None:
                body_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    body_frame, 
                    current_data['pose'], 
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                cv2.imwrite(os.path.join(body_path, f"frame_{output_frame_count:05d}.png"), body_frame)
            else:
                # Try to use previous frame if available for temporal consistency
                if i > 0 and frame_data[i-1]['pose'] is not None:
                    body_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        body_frame, 
                        frame_data[i-1]['pose'], 
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Lighter for interpolated
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )
                    cv2.imwrite(os.path.join(body_path, f"frame_{output_frame_count:05d}.png"), body_frame)

            # LEFT HAND PROCESSING with enhanced logic
            if current_data['left_hand'] is not None:
                left_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    left_frame, 
                    current_data['left_hand'], 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                cv2.imwrite(os.path.join(left_hand_path, f"frame_{output_frame_count:05d}.png"), left_frame)
            else:
                # Look for nearby frames with left hand detection
                for offset in [-2, -1, 1, 2]:
                    check_idx = i + offset
                    if 0 <= check_idx < len(frame_data) and frame_data[check_idx]['left_hand'] is not None:
                        left_frame = frame.copy()
                        mp_drawing.draw_landmarks(
                            left_frame, 
                            frame_data[check_idx]['left_hand'], 
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),  # Lighter for interpolated
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                        )
                        cv2.imwrite(os.path.join(left_hand_path, f"frame_{output_frame_count:05d}.png"), left_frame)
                        break

            # RIGHT HAND PROCESSING with enhanced logic
            if current_data['right_hand'] is not None:
                right_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    right_frame, 
                    current_data['right_hand'], 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )
                cv2.imwrite(os.path.join(right_hand_path, f"frame_{output_frame_count:05d}.png"), right_frame)
            else:
                # Look for nearby frames with right hand detection
                for offset in [-2, -1, 1, 2]:
                    check_idx = i + offset
                    if 0 <= check_idx < len(frame_data) and frame_data[check_idx]['right_hand'] is not None:
                        right_frame = frame.copy()
                        mp_drawing.draw_landmarks(
                            right_frame, 
                            frame_data[check_idx]['right_hand'], 
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),  # Lighter for interpolated
                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
                        )
                        cv2.imwrite(os.path.join(right_hand_path, f"frame_{output_frame_count:05d}.png"), right_frame)
                        break

            output_frame_count += 1
    
    print(f"Generated {output_frame_count} frames at 60fps from {frame_count} original frames")

# Alternative simpler approach - frame duplication
def extract_frames_60fps_simple(video_path, output_root):
    """
    Simpler approach: duplicate frames to achieve 60fps
    """
    os.makedirs(output_root, exist_ok=True)
    body_path = os.path.join(output_root, "body_frames")
    left_hand_path = os.path.join(output_root, "left_hand_frames")
    right_hand_path = os.path.join(output_root, "right_hand_frames")

    os.makedirs(body_path, exist_ok=True)
    os.makedirs(left_hand_path, exist_ok=True)
    os.makedirs(right_hand_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate duplication factor (2x for 30fps -> 60fps)
    duplication_factor = int(60 / original_fps)
    
    frame_count = 0
    output_frame_count = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.3,  # Lower for better detection
        min_tracking_confidence=0.3
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Generate multiple frames for 60fps
            for dup in range(duplication_factor):
                # BODY
                if results.pose_landmarks:
                    body_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        body_frame, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    cv2.imwrite(os.path.join(body_path, f"frame_{output_frame_count:05d}.png"), body_frame)

                # LEFT HAND
                if results.left_hand_landmarks:
                    left_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        left_frame, 
                        results.left_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    cv2.imwrite(os.path.join(left_hand_path, f"frame_{output_frame_count:05d}.png"), left_frame)

                # RIGHT HAND
                if results.right_hand_landmarks:
                    right_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        right_frame, 
                        results.right_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                    cv2.imwrite(os.path.join(right_hand_path, f"frame_{output_frame_count:05d}.png"), right_frame)

                output_frame_count += 1
            
            frame_count += 1

    cap.release()
    print(f"Original frames: {frame_count}, Output frames: {output_frame_count}")

# Process all cropped videos
cropped_folder = "track1/cropped_videos/BLUETOOTH"
output_folder = "track1/frames/BLUETOOTH"

# Choose your approach:
# 1. Advanced approach with interpolation and temporal consistency
# 2. Simple approach with frame duplication

for video in tqdm(os.listdir(cropped_folder)):
    if video.endswith(".mp4"):
        video_name = video.split(".")[0]
        print(f"\nProcessing {video_name}...")
        
        # Use either approach:
        
        # RECOMMENDED: Simple but effective
        extract_frames_60fps_simple(
            os.path.join(cropped_folder, video),
            os.path.join(output_folder, video_name)
        )
        
        # OR: Advanced with interpolation (slower but potentially better quality)
        # extract_frames_60fps(
        #     os.path.join(cropped_folder, video),
        #     os.path.join(output_folder, video_name)
        # )
