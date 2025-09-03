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

        for frames, labels in dataloader:
            labels = labels.to(device)

            mid_idx = frames.shape[2] // 2
            mid_frames = frames[:, :, mid_idx, :, :]

            pil_imgs = [Image.fromarray((f.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for f in mid_frames]
            embeddings = extract_embeddings(pil_imgs).to(device)

            preds = model(embeddings)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_fn(preds, labels)

        losses.append(epoch_loss/len(dataloader))
        accs.append(epoch_acc/len(dataloader))

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
