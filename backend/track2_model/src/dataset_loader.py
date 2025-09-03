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

        self.samples = []      # list of (body_paths, left_paths, right_paths, label)
        self.word_to_idx = {}  # map word → integer class

        # Map each word folder to an index
        for idx, word in enumerate(sorted(os.listdir(root_dir))):
            word_path = os.path.join(root_dir, word)
            if not os.path.isdir(word_path):
                continue

            self.word_to_idx[word] = idx

            # Each sample (person/session) folder
            for sample in sorted(os.listdir(word_path)):
                sample_path = os.path.join(word_path, sample)
                if not os.path.isdir(sample_path):
                    continue

                # Expect: body/, left/, right/
                body_path = os.path.join(sample_path, "body")
                left_path = os.path.join(sample_path, "left_hand")
                right_path = os.path.join(sample_path, "right_hand")

                if not os.path.isdir(body_path):
                    continue  # skip if body missing (mandatory)

                body_frames = sorted(os.listdir(body_path))[:frames_per_sample]
                left_frames = sorted(os.listdir(left_path))[:frames_per_sample] if os.path.isdir(left_path) else []
                right_frames = sorted(os.listdir(right_path))[:frames_per_sample] if os.path.isdir(right_path) else []

                body_paths = [os.path.join(body_path, f) for f in body_frames]

                # Only keep samples with full body frames
                if len(body_paths) == frames_per_sample:
                    self.samples.append((body_paths, left_frames, right_frames, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body_paths, left_paths, right_paths, label = self.samples[idx]

        def load_frames(paths, size):
            frames = []
            for f in paths:
                img = cv2.imread(f)
                img = cv2.resize(img, size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                frames.append(img)
            return frames

        # Load body frames (required)
        body_frames = load_frames(body_paths, self.image_size)

        # Load left/right if available, else fill with zeros
        if left_paths:
            left_frames = load_frames([os.path.join(os.path.dirname(body_paths[0]), "..", "left", os.path.basename(f)) for f in left_paths], (128, 128))
        else:
            left_frames = [np.zeros((128, 128, 3), dtype=np.float32) for _ in range(self.frames_per_sample)]

        if right_paths:
            right_frames = load_frames([os.path.join(os.path.dirname(body_paths[0]), "..", "right", os.path.basename(f)) for f in right_paths], (128, 128))
        else:
            right_frames = [np.zeros((128, 128, 3), dtype=np.float32) for _ in range(self.frames_per_sample)]

        # Convert to tensors
        body = np.stack(body_frames)  # (T, H, W, C)
        body = np.transpose(body, (3, 0, 1, 2))  # (C, T, H, W)

        left = np.stack(left_frames)
        left = np.transpose(left, (3, 0, 1, 2))

        right = np.stack(right_frames)
        right = np.transpose(right, (3, 0, 1, 2))

        body = torch.tensor(body, dtype=torch.float32)
        left = torch.tensor(left, dtype=torch.float32)
        right = torch.tensor(right, dtype=torch.float32)

        if self.transform:
            body = self.transform(body)

        return {"body": body, "left": left, "right": right}, label

