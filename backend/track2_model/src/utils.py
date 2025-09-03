# backend/track2_model/src/utils.py

import torch

def accuracy_fn(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()
