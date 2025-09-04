# backend/track2_model/src/main.py

import argparse
import torch
from torch.utils.data import DataLoader
from train_encoder import train_encoder
from dataset_loader import SignLanguageDataset
from encoder_decoder import SignClassifier, extract_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main():
    parser = argparse.ArgumentParser(description="Track2 Model Training & Evaluation")
    parser.add_argument("--mode", type=str, choices=["encoder", "diffusion"], default="encoder")
    parser.add_argument("--data", type=str, default="D:/Dhanvanth/SL/Sign_language_Project/Sign_language_Project/backend/track2_model/data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    if args.mode == "encoder":
        print("🚀 Training Encoder/Classifier...")
        train_encoder(data_path=args.data, num_classes=args.num_classes, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
