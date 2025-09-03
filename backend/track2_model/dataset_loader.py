# backend/track2_model/src/dataset_loader.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, frames_per_sample=60, image_size=(256, 256), transform=None):
        """
        Args:
            root_dir (str): Path to dataset containing words as folders
            frames_per_sample (int): Number of frames per video sample (default=60)
            image_size (tuple): Resize images to this size (H, W)
            transform: Optional torchvision transforms applied on tensor
        """
        self.root_dir = root_dir
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.transform = transform

        self.samples = []      # list of (frame_paths, label)
        self.word_to_idx = {}  # map word → integer class

        # Map each word folder to an index
        for idx, word in enumerate(sorted(os.listdir(root_dir))):
            word_path = os.path.join(root_dir, word)
            if not os.path.isdir(word_path):
                continue

            self.word_to_idx[word] = idx

            # Each person folder is one sample
            for person in sorted(os.listdir(word_path)):
                person_path = os.path.join(word_path, person)
                if not os.path.isdir(person_path):
                    continue

                frames = sorted(os.listdir(person_path))[:frames_per_sample]
                frame_paths = [os.path.join(person_path, f) for f in frames]

                if len(frame_paths) == frames_per_sample:
                    self.samples.append((frame_paths, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []

        for f in frame_paths:
            img = cv2.imread(f)
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # normalize [0,1]
            frames.append(img)

        frames = np.stack(frames)  # (T, H, W, C)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        frames = torch.tensor(frames, dtype=torch.float32)

        if self.transform:
            frames = self.transform(frames)

        return frames, label


# ---------------------------
# Example usage (self-test)
# ---------------------------
if __name__ == "__main__":
    data_path = "backend/track2_model/data/processed"

    dataset = SignLanguageDataset(root_dir=data_path)

    print("📂 Classes found:", dataset.word_to_idx)
    print("📦 Total samples:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for frames, labels in dataloader:
        print("✅ Batch loaded")
        print("Frames shape:", frames.shape)  # (B, 3, 60, 256, 256)
        print("Labels:", labels)              # (B,)
        break
