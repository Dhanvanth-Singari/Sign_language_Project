# backend/track2_model/src/main.py

import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image
import cv2
import os

from train_encoder import train_encoder
from train_diffusion import train_diffusion
from dataset_loader import SignLanguageDataset
from encoder_decoder import SignClassifier, extract_embeddings
from utils import accuracy_fn
from diffusion_model import build_diffusion_models

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Evaluation: Encoder
# ------------------------------
def evaluate_encoder(model_path, data_path, num_classes, batch_size=4):
    dataset = SignLanguageDataset(root_dir=data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SignClassifier(embed_dim=512, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    total_acc = 0

    with torch.no_grad():
        for frames, labels in loader:
            labels = labels.to(device)
            mid_idx = frames.shape[2] // 2
            mid_frames = frames[:, :, mid_idx, :, :]

            pil_imgs = [Image.fromarray((f.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for f in mid_frames]
            embeddings = extract_embeddings(pil_imgs).to(device)

            preds = model(embeddings)
            total_acc += accuracy_fn(preds, labels)

            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_acc = total_acc / len(loader)
    print(f"✅ Evaluation Accuracy: {avg_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.word_to_idx.keys()))
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix - Encoder")
    plt.show()


# ------------------------------
# Evaluation: Diffusion
# ------------------------------
def evaluate_diffusion(model_path, data_path, num_samples=4):
    dataset = SignLanguageDataset(root_dir=data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    unet, scheduler = build_diffusion_models()
    checkpoint = torch.load(model_path, map_location=device)
    unet.load_state_dict(checkpoint["model_state"])
    unet = unet.to(device)
    unet.eval()

    output_dir = "backend/track2_model/outputs/base_action_videos"
    os.makedirs(output_dir, exist_ok=True)

    fid = FrechetInceptionDistance(feature=64).to(device)

    all_psnr, all_ssim = [], []

    for i, (frames, labels) in enumerate(loader):
        if i >= num_samples:
            break

        frames = frames.to(device)
        mid_idx = frames.shape[2] // 2
        x0 = frames[:, :, mid_idx, :, :]   # (B, C, H, W)

        noise = torch.randn_like(x0)
        timesteps = torch.tensor([scheduler.num_train_timesteps-1], device=device).long()
        noisy = scheduler.add_noise(x0, noise, timesteps)

        generated_frames = []
        with torch.no_grad():
            for t in scheduler.timesteps[::50]:  # partial denoise
                pred = unet(noisy, t, encoder_hidden_states=None).sample
                noisy = scheduler.step(pred, t, noisy).prev_sample
                frame = (noisy[0].permute(1,2,0).cpu().clamp(0,1).numpy() * 255).astype("uint8")
                generated_frames.append(frame)

        # Original vs Generated final frame
        orig = (x0[0].permute(1,2,0).cpu().clamp(0,1).numpy() * 255).astype("uint8")
        gen = generated_frames[-1]

        # --- Metrics ---
        psnr_val = psnr(orig, gen, data_range=255)
        ssim_val = ssim(orig, gen, channel_axis=-1, data_range=255)
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)

        fid.update(torch.tensor(orig).permute(2,0,1).unsqueeze(0).byte().to(device), real=True)
        fid.update(torch.tensor(gen).permute(2,0,1).unsqueeze(0).byte().to(device), real=False)

        # Save video
        word = list(dataset.word_to_idx.keys())[labels.item()]
        save_path = os.path.join(output_dir, f"{word}_{i}.mp4")
        h, w, _ = generated_frames[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for f in generated_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

        print(f"🎥 Saved base action video: {save_path}")
        print(f"📊 Sample {i}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        axes[0].imshow(orig)
        axes[0].set_title("Original Frame")
        axes[0].axis("off")
        axes[1].imshow(gen)
        axes[1].set_title("Generated Frame")
        axes[1].axis("off")
        plt.show()

    # --- Final metrics ---
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    fid_score = fid.compute().item()

    print(f"\n✅ Diffusion Evaluation Complete")
    print(f"   Avg PSNR: {avg_psnr:.2f}")
    print(f"   Avg SSIM: {avg_ssim:.3f}")
    print(f"   FID Score: {fid_score:.2f}")


# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Track2 Model Training & Evaluation")
    parser.add_argument("--mode", type=str, choices=["encoder", "diffusion"], required=True)
    parser.add_argument("--data", type=str, default="backend/track2_model/data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation")

    args = parser.parse_args()

    if args.mode == "encoder":
        print("🚀 Training Encoder/Classifier...")
        train_encoder(data_path=args.data, num_classes=args.num_classes, epochs=args.epochs, batch_size=args.batch_size)

        if args.eval and args.checkpoint:
            evaluate_encoder(args.checkpoint, args.data, args.num_classes, args.batch_size)

    elif args.mode == "diffusion":
        print("🌌 Training Diffusion Model...")
        train_diffusion(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)

        if args.eval and args.checkpoint:
            evaluate_diffusion(args.checkpoint, args.data)


if __name__ == "__main__":
    main()
