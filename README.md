# Description
SGT-Net implementation for skin lesion segmentation using CNN, Transformer, and spectral learning. Achieves high Dice (0.91+) on ISIC datasets with fuzzy logic-based probabilistic outputs.

# SGT-Net: Skin Lesion Segmentation using Spectral Generalized Transformer

## 📌 Overview
This project implements **SGT-Net (Spectral Generalized Transformer Network)** for automated skin lesion segmentation from dermoscopic images.

The model combines:
- Convolutional Neural Networks (CNN) for local feature extraction
- Transformer architecture for global context understanding
- Spectral learning (FFT-based) for noise reduction and generalization

This hybrid approach enables accurate and robust segmentation, even in challenging medical images.

---

## 🎯 Problem Statement
Given a dermoscopic image, the goal is to generate a **binary segmentation mask** that identifies the lesion region at the pixel level.

Challenges include:
- Irregular lesion shapes
- Low contrast boundaries
- Noise and artifacts (hair, lighting variations)
- Poor generalization across datasets

---

## 🧠 Methodology
The architecture consists of three main components:

### 1. Dual Encoder
- **CNN (ResNet/Res2Net):** captures spatial and local features  
- **Transformer (DeiT):** captures global dependencies using self-attention  

### 2. Spectral Adaptive Block (SAB)
- Applies **Fast Fourier Transform (FFT)**  
- Filters features in frequency domain  
- Suppresses noise and enhances important structures  

### 3. Decoder with Reverse Attention
- Refines segmentation iteratively  
- Focuses on missed and boundary regions  
- Uses skip connections for detail preservation  

Final output is a **probability map (0–1)** → thresholded to binary mask.

---

## 📊 Results

Model performance after **10 epochs**:

| Metric | Value |
|-------|------|
| Validation Loss | 0.1883 |
| Dice Score | 0.9132 |
| IoU Score | 0.8431 |

✔ High Dice score indicates strong overlap with ground truth  
✔ Efficient convergence due to pretrained encoders  


---

## 🧪 Output Visualization

Each prediction generates:
- Original Image  
- Probability Map (fuzzy output)  
- Binary Mask  
- Overlay Visualization  

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- OpenCV  
- Albumentations  
- NumPy  

---

## 📂 Project Structure
├── train.py # Training pipeline
├── inference.py # Prediction script
├── models/
│ └── Pra_FATNet_local.py # SGT-Net architecture
├── dataset/
├── checkpoints/
└── README.md


---

## ⚙️ Training Details

- Image Size: 384x384  
- Batch Size: 4  
- Optimizer: AdamW  
- Loss Function: BCE + Dice  
- Epochs: 10  
- Scheduler: Cosine Annealing  

---

## 🧩 Soft Computing Perspective

This project applies key soft computing concepts:

- **Artificial Neural Networks:** deep learning backbone  
- **Fuzzy Logic:** sigmoid output as membership function (0–1)  
- **Defuzzification:** thresholding to binary mask  

---

## 🚀 Future Work
- Genetic Algorithm for hyperparameter tuning  
- Wavelet-based spectral learning  
- Model optimization for real-time deployment  
- Extension to other medical imaging tasks  

---

## 📚 References
- SGT-Net Paper  
- U-Net (2015)  
- PraNet (2021)  
- DeiT Transformer  
- ISIC Dataset  

---
