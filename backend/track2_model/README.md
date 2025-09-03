Got it 👍 I’ll keep it **brief and practical**, so your team can just follow and run.

---

# 🚀 How to Run (Track 2 – Step 1)

1. **Install dependencies** (only once):

   ```powershell
   pip install opencv-python mediapipe tqdm numpy
   ```

2. **Put raw videos** inside:

   ```
   backend/track2_model/data/raw/<WORD_NAME>/
   ```

   Example:

   ```
   backend/track2_model/data/raw/LOCK/person01.mp4
   backend/track2_model/data/raw/LOCK/person02.mp4
   ```

3. **Run preprocessing script**:

   ```powershell
   python backend\track2_model\src\step1_preprocess.py `
     --input_dir backend\track2_model\data\raw `
     --output_dir backend\track2_model\data\processed `
     --frames 60 --size 256 --quality 95
   ```

4. **Check output frames** in:

   ```
   backend/track2_model/data/processed/<WORD_NAME>/<PERSON_ID>/
   ```

---
5. **Test the dataset loader**
python backend/track2_model/src/dataset_loader.py

6. **Use in your training script**
from dataset_loader import SignLanguageDataset
from torch.utils.data import DataLoader

dataset = SignLanguageDataset("backend/track2_model/data/processed")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for frames, labels in dataloader:
    print(frames.shape)   # (B, 3, 60, 256, 256)
    print(labels)         # word indices
    break
7. **Data → Embeddings (CLIP) → Diffusion → Losses → Evaluation → Output.**

"""

---

# 🔹 Final Implementation Steps (Track 2)

---

## **Step 1: Data Preparation**

* Input: 30 samples × 60 frames per word.
* For each frame:

  1. Extract **holistic pose** (body, hands, face).
  2. Normalize full frame to **256×256**.
  3. Crop **left hand** and **right hand** regions (e.g., 128×128).
  4. Save:

     ```
     dataset/
       WORD_1/
         sample_01/
            full/
            left_hand/
            right_hand/
         sample_02/
         ...
     ```

---

## **Step 2: Feature Extraction (CLIP Video Encoder)**

* Use **CLIP Vision Encoder (ViT-B/32 or ViT-L/14)**.

* For each frame sequence:

  * Extract embeddings for **full frame**.
  * Extract embeddings for **left + right hands**.
  * Fuse embeddings (concatenate or weighted sum):

    $$
    E = W_f E_{full} + W_l E_{left} + W_r E_{right}
    $$

* Save fused embeddings as **motion representation**.

---

## **Step 3: Diffusion Model (Stable Diffusion Backbone)**

* Condition diffusion on **motion embeddings**.

* Pipeline:

  ```
  Noise → Diffusion U-Net
        + Conditioning (E)
        → Denoised Frames (Base Action Video)
  ```

* Training:

  * Forward process adds noise to ground-truth frames.
  * Reverse process denoises guided by motion embeddings.

---

## **Step 4: Loss Functions**

We balance **full-frame + hand-priority losses**:

1. **Full Reconstruction Loss (MSE):**

$$
L_{recon}^{full} = \|x - \hat{x}\|^2
$$

2. **Hand-Weighted Loss:**

$$
L_{recon}^{hand} = \|M \odot (x - \hat{x})\|^2
$$

3. **Perceptual Loss (VGG/DINOv2):**

$$
L_{perc} = \|\phi(x) - \phi(\hat{x})\|^2
$$

4. **Embedding Consistency (CLIP):**

$$
L_{emb} = 1 - \cos(E_{real}, E_{gen})
$$

5. **Total Loss:**

$$
L_{total} = \lambda_1 L_{recon}^{full} + \lambda_2 L_{recon}^{hand} + \lambda_3 L_{perc} + \lambda_4 L_{emb}
$$

---

## **Step 5: Training Setup**

* **Split**: 70% train / 15% validation / 15% test.
* **Optimizer**: AdamW (learning rate 1e-4).
* **Batch size**: 8–16 sequences.
* **Scheduler**: cosine decay.
* **Epochs**: 50–100 depending on dataset size.

---

## **Step 6: Evaluation Metrics**

* **PSNR**: frame sharpness.
* **SSIM**: structure similarity.
* **FID**: realism of generated frames.
* **CLIP Similarity**: embedding alignment.
* **Hand Keypoint Distance (HKD)**: precision of finger articulation.

---

## **Step 7: Output**

* For each word:

  * One **Base Action Video** (60 frames).
  * Stored in:

    ```
    base_action_videos/
       WORD_1.mp4
       WORD_2.mp4
    ```

---

"""
8. 