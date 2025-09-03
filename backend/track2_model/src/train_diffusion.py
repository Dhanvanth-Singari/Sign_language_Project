# backend/track2_model/src/train_diffusion.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from dataset_loader import SignLanguageDataset
from diffusion_model import build_diffusion_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_diffusion(data_path, epochs=5, batch_size=2):
    dataset = SignLanguageDataset(root_dir=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet, scheduler = build_diffusion_models()
    unet = unet.to(device)
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        unet.train()

        for frames, _ in dataloader:
            frames = frames.to(device)
            mid_idx = frames.shape[2] // 2
            x0 = frames[:, :, mid_idx, :, :]

            noise = torch.randn_like(x0)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
            noisy = scheduler.add_noise(x0, noise, timesteps)

            noise_pred = unet(noisy, timesteps, encoder_hidden_states=None).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss/len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}] Loss={losses[-1]:.4f}")

        os.makedirs("backend/track2_model/outputs/checkpoints", exist_ok=True)
        torch.save({
            "model_state": unet.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, f"backend/track2_model/outputs/checkpoints/diffusion_epoch{epoch+1}.pt")

    plt.plot(losses, label="Diffusion Loss")
    plt.legend()
    plt.show()
