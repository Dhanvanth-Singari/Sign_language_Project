# backend/track2_model/src/encoder_decoder.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_embeddings(frames):
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings

class SignClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_classes=10):
        super(SignClassifier, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
