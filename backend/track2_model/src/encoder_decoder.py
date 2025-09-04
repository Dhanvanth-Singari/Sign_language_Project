# backend/track2_model/src/encoder_decoder.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP once globally
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_embeddings(frames):
    """
    Extract CLIP embeddings for a list of PIL images.
    Returns tensor of shape (B, 512)
    """
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings

class SignClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_classes=10):
        super(SignClassifier, self).__init__()
        self.fc = nn.Linear(embed_dim * 3, num_classes)

    def forward(self, body_emb, left_emb, right_emb):
        combined = torch.cat([body_emb, left_emb, right_emb], dim=1)
        return self.fc(combined)
