# Track 3 — Target-Image Animation via Pose (DreamPose-style, practical baseline)

This module animates a **target image** using **pose sequences** from base action frames (Track 2 output).  
It uses a *DreamPose-style* approach with a practical baseline built on **Stable Diffusion + ControlNet(OpenPose)**.
The baseline preserves target identity by running **img2img** with the target image while **ControlNet** enforces pose.

> Paper reference: *DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion (arXiv:2304.06025)*

## Folder Layout
```
backend/track3_frame_generation/
  ├─ target_images/           # put your target/avatar image(s) here (e.g., target.png)
  ├─ base_action_frames/      # frames (PNG/JPG) for each word sequence from Track 2
  ├─ pose/                    # extracted skeletons (JSON + preview PNGs)
  ├─ final_frames/            # generated output frames for each word
  ├─ tools/
  │   └─ extract_video_frames.py
  ├─ dreampose_like/
  │   ├─ extract_poses.py
  │   ├─ infer_controlnet_img2img.py
  │   └─ utils.py
  └─ run_track3.py
```

## Quickstart

1) **Install deps** (Python 3.10+ recommended):
```bash
pip install -r backend/track3_frame_generation/requirements.txt
```

2) **Prepare inputs**
- Put your **target image** at: `backend/track3_frame_generation/target_images/target.png`
- Put your **base action frames** (sequence for a word) into a folder, e.g.  
  `backend/track3_frame_generation/base_action_frames/word_hello/` with files `0001.jpg, 0002.jpg, ...`  
  (If you only have a video, first extract frames: see `tools/extract_video_frames.py`)

3) **Extract poses from base action frames**
```bash
python backend/track3_frame_generation/dreampose_like/extract_poses.py   --frames_dir backend/track3_frame_generation/base_action_frames/word_hello   --out_dir    backend/track3_frame_generation/pose/word_hello
```

4) **Generate final frames (ControlNet OpenPose img2img)**
```bash
python backend/track3_frame_generation/dreampose_like/infer_controlnet_img2img.py   --pose_dir    backend/track3_frame_generation/pose/word_hello/pose_png   --target_img  backend/track3_frame_generation/target_images/target.png   --out_dir     backend/track3_frame_generation/final_frames/word_hello   --prompt      "a realistic person performing ISL sign for 'hello', plain background, high detail"
```

5) **(Optional) Stitch to video**
You can convert frames into a video using ffmpeg:
```bash
ffmpeg -y -framerate 25 -i backend/track3_frame_generation/final_frames/word_hello/%04d.png   -c:v libx264 -pix_fmt yuv420p backend/track3_frame_generation/final_frames/word_hello.mp4
```

## Notes
- **DreamPose two-phase fine-tuning** is approximated here by using ControlNet(OpenPose) with the target image as `init_image`. For subject-specific adaptation, fine-tune or use IP-Adapter FaceID (not included by default).
- Set `--guidance_scale`, `--strength`, and `--cfg_text` in `infer_controlnet_img2img.py` to balance identity vs. pose fidelity.
- For ISL, ensure **both hands** are visible and that your pose extractor runs `mediapipe.holistic` (provided).
