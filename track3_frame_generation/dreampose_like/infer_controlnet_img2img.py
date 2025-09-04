import os
import argparse
from PIL import Image
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


# ---------------------------
# Resize helper
# ---------------------------
def preprocess_images(pose_dir, target_img, size=(512, 512)):
    # Resize target avatar
    if os.path.exists(target_img):
        avatar = Image.open(target_img).convert("RGB").resize(size)
        avatar.save(target_img)

    # Resize pose images
    for fname in os.listdir(pose_dir):
        if fname.endswith(".png"):
            fpath = os.path.join(pose_dir, fname)
            img = Image.open(fpath).convert("RGB").resize(size)
            img.save(fpath)


# ---------------------------
# Build Pipeline
# ---------------------------
def build_pipelines(device, sd_model, controlnet_model, dtype):
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    sd_model,
    controlnet=controlnet,
    # torch_dtype=torch.float32
    torch_dtype=torch.float16
)

    pipe.to(device)
    return pipe


# ---------------------------
# Main
# ---------------------------
def main(pose_dir, target_img, out_dir, prompt,
         sd_model="runwayml/stable-diffusion-v1-5",
         controlnet_model="lllyasviel/sd-controlnet-openpose",
         device="cuda" if torch.cuda.is_available() else "cpu",
         dtype=torch.float16):

    # Step 1: Preprocess input images
    preprocess_images(pose_dir, target_img, size=(512, 512))

    # Step 2: Build pipeline
    pipe = build_pipelines(device, sd_model, controlnet_model, dtype)

    # Step 3: Load target avatar
    avatar = Image.open(target_img).convert("RGB")

    # Step 4: Process each pose image
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(pose_dir)):
        if not fname.endswith(".png"):
            continue

        pose_path = os.path.join(pose_dir, fname)
        pose_img = Image.open(pose_path).convert("RGB")

        result = pipe(
            prompt=prompt,
            image=avatar,
            control_image=pose_img,
            strength=0.8,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

        result.save(os.path.join(out_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory with pose images")
    parser.add_argument("--target_img", type=str, required=True, help="Path to target avatar image")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--prompt", type=str, default="A person performing the action", help="Prompt for generation")
    args = parser.parse_args()
    main(**vars(args))
