# backend/track2_model/src/train_encoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from encoder_decoder import extract_embeddings, SignClassifier
from dataset_loader import SignLanguageDataset
from utils import accuracy_fn
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_encoder(data_path, num_classes, epochs=10, batch_size=4):
    dataset = SignLanguageDataset(root_dir=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SignClassifier(embed_dim=512, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []

    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0, 0
        model.train()

        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)

            # Split body, left, right
            C = batch.shape[1] // 3
            body = batch[:, :C, :, :, :]
            left = batch[:, C:2*C, :, :, :]
            right = batch[:, 2*C:, :, :, :]

            mid_idx = body.shape[2] // 2
            body_mid = body[:, :, mid_idx, :, :]
            left_mid = left[:, :, mid_idx, :, :]
            right_mid = right[:, :, mid_idx, :, :]

            def to_pil(frames):
                return [Image.fromarray((f.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for f in frames]

            body_pil = to_pil(body_mid)
            left_pil = to_pil(left_mid)
            right_pil = to_pil(right_mid)

            body_embed = extract_embeddings(body_pil)
            left_embed = extract_embeddings(left_pil)
            right_embed = extract_embeddings(right_pil)

            preds = model(body_embed, left_embed, right_embed)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_fn(preds, labels)

        losses.append(epoch_loss / len(dataloader))
        accs.append(epoch_acc / len(dataloader))

        print(f"Epoch [{epoch+1}/{epochs}] Loss={losses[-1]:.4f} Acc={accs[-1]:.4f}")

        os.makedirs("backend/track2_model/outputs/checkpoints", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, f"backend/track2_model/outputs/checkpoints/encoder_epoch{epoch+1}.pt")

    plt.plot(losses, label="Loss")
    plt.plot(accs, label="Accuracy")
    plt.legend()
    plt.title("Encoder Training")
    plt.show()
