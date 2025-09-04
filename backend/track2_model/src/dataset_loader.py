# backend/track2_model/src/dataset_loader.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def pad_or_truncate(frames, target_len, size):
    """Pad with last frame or truncate to fixed length."""
    n = len(frames)
    if n == target_len:
        return frames
    elif n > target_len:
        return frames[:target_len]
    else:
        pad_frame = np.zeros_like(frames[-1]) if n > 0 else np.zeros(size, dtype=np.float32)
        frames.extend([pad_frame] * (target_len - n))
        return frames

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, frames_per_sample=16, image_size=(128, 128), transform=None):
        self.root_dir = root_dir
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        self.word_to_idx = {}

        for idx, word in enumerate(sorted(os.listdir(root_dir))):
            word_path = os.path.join(root_dir, word)
            if not os.path.isdir(word_path):
                continue
            self.word_to_idx[word] = idx
            for sample in sorted(os.listdir(word_path)):
                sample_path = os.path.join(word_path, sample)
                if not os.path.isdir(sample_path):
                    continue
                body_path = os.path.join(sample_path, "body")
                left_path = os.path.join(sample_path, "left_hand")
                right_path = os.path.join(sample_path, "right_hand")
                if not os.path.isdir(body_path):
                    continue
                self.samples.append((body_path, left_path, right_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body_path, left_path, right_path, label = self.samples[idx]

        def load_frames_from_dir(path, size):
            frames = []
            if path and os.path.isdir(path):
                for f in sorted(os.listdir(path)):
                    img = cv2.imread(os.path.join(path, f))
                    if img is None:
                        continue
                    img = cv2.resize(img, size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    frames.append(img)
            return frames

        body_frames = load_frames_from_dir(body_path, self.image_size)
        left_frames = load_frames_from_dir(left_path, (128, 128))
        right_frames = load_frames_from_dir(right_path, (128, 128))

        body_frames = pad_or_truncate(body_frames, self.frames_per_sample, (self.image_size[0], self.image_size[1], 3))
        left_frames = pad_or_truncate(left_frames, self.frames_per_sample, (128, 128, 3))
        right_frames = pad_or_truncate(right_frames, self.frames_per_sample, (128, 128, 3))

        body = torch.tensor(np.transpose(np.stack(body_frames), (3,0,1,2)), dtype=torch.float32)
        left = torch.tensor(np.transpose(np.stack(left_frames), (3,0,1,2)), dtype=torch.float32)
        right = torch.tensor(np.transpose(np.stack(right_frames), (3,0,1,2)), dtype=torch.float32)

        frames = torch.cat([body, left, right], dim=0)
        if self.transform:
            frames = self.transform(frames)

        return frames, label
