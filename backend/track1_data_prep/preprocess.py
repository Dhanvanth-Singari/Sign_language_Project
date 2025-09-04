import os
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def get_person_bbox(frame, pose_model):
    results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in results.pose_landmarks.landmark]
    ys = [lm.y * h for lm in results.pose_landmarks.landmark]
    
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def get_video_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(static_image_mode=False) as pose:
        x_min_total, y_min_total = 1e9, 1e9
        x_max_total, y_max_total = 0, 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = get_person_bbox(frame, pose)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                x_min_total = min(x_min_total, x_min)
                y_min_total = min(y_min_total, y_min)
                x_max_total = max(x_max_total, x_max)
                y_max_total = max(y_max_total, y_max)
    
    cap.release()
    if x_max_total == 0 and y_max_total == 0:
        return None
    return x_min_total, y_min_total, x_max_total, y_max_total

def crop_person_from_video(input_path, output_path, output_size=512, padding_factor=0.3, min_padding=50):
    """
    Crop person from video with better framing control
    
    Args:
        padding_factor: Percentage of person size to add as padding (0.3 = 30% larger frame)
        min_padding: Minimum padding in pixels
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get original video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    bbox = get_video_bbox(input_path)
    if not bbox:
        print(f"No person detected in {input_path}")
        return
    
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate person dimensions
    person_width = x_max - x_min
    person_height = y_max - y_min
    
    # Smart padding based on person size and factor
    padding_x = max(int(person_width * padding_factor), min_padding)
    padding_y = max(int(person_height * padding_factor), min_padding)
    
    # Apply padding
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(original_width, x_max + padding_x)
    y_max = min(original_height, y_max + padding_y)
    
    # Make it square - but use the larger dimension to preserve more context
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)
    
    # Center the square crop
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    
    # Calculate square boundaries
    half_side = side // 2
    x_min_square = max(0, cx - half_side)
    x_max_square = min(original_width, cx + half_side)
    y_min_square = max(0, cy - half_side)
    y_max_square = min(original_height, cy + half_side)
    
    # Adjust if we hit boundaries
    actual_width = x_max_square - x_min_square
    actual_height = y_max_square - y_min_square
    final_side = min(actual_width, actual_height)
    
    # Recenter with actual achievable size
    x_min_final = max(0, cx - final_side // 2)
    x_max_final = min(original_width, x_min_final + final_side)
    y_min_final = max(0, cy - final_side // 2)  
    y_max_final = min(original_height, y_min_final + final_side)
    
    # High-quality codec settings
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (output_size, output_size))
    
    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        cropped = frame[y_min_final:y_max_final, x_min_final:x_max_final]
        
        # High-quality resize
        resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
        out.write(resized)
        
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed: {os.path.basename(input_path)}")
    print(f"  Original: {original_width}x{original_height}")
    print(f"  Cropped area: {final_side}x{final_side}")
    print(f"  Output: {output_size}x{output_size}")
    print(f"  Frames: {frame_count}")

def crop_videos_in_folder(input_folder, output_folder, output_size=512, padding_factor=0.3):
    """
    Process all videos with adjustable framing
    
    Args:
        padding_factor: How much context to include around person
                       0.2 = tight crop (20% padding)
                       0.3 = medium crop (30% padding) - RECOMMENDED
                       0.5 = loose crop (50% padding) 
                       0.8 = very loose crop (80% padding)
    """
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"\nProcessing {file_name}...")
            crop_person_from_video(input_path, output_path, output_size, padding_factor)

# Usage examples:
input_folder = "track1/raw_videos/BLUETOOTH"
output_folder = "track1/cropped_videos/BLUETOOTH"

# Different framing options:

# 1. RECOMMENDED: Medium framing (30% padding around person)
crop_videos_in_folder(input_folder, output_folder, output_size=512, padding_factor=0.3)

# 2. Looser framing (50% padding) - more context, less zoomed
# crop_videos_in_folder(input_folder, output_folder, output_size=512, padding_factor=0.5)

# 3. Very loose framing (80% padding) - lots of background context  
# crop_videos_in_folder(input_folder, output_folder, output_size=512, padding_factor=0.8)
