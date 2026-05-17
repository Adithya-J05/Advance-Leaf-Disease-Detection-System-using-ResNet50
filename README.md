<p align="center">
  <h1 align="center">🍃 Advance Leaf Disease Detection System using ResNet50</h1>
  <p align="center">
    A deep learning–powered system for detecting and classifying <b>tomato leaf diseases</b> using a fine-tuned <b>ResNet50</b> model, built with <b>PyTorch</b> and accelerated with <b>CUDA</b>.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/ResNet50-Transfer%20Learning-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Disease Classes](#-disease-classes)
- [Project Architecture](#-project-architecture)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Results & Metrics](#-results--metrics)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Plant diseases are a significant threat to global food security. Early and accurate detection of leaf diseases can help farmers take timely action and prevent crop loss.

This project implements an **end-to-end deep learning pipeline** that:

1. **Preprocesses** tomato leaf images (resize, normalize, augment)
2. **Trains** a ResNet50 model using **transfer learning** for multi-class classification
3. **Evaluates** performance with precision, recall, F1-score, and confusion matrices
4. **Predicts** disease from new leaf images with confidence scores
5. **Explains** predictions using **LIME** (Local Interpretable Model-agnostic Explanations)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Transfer Learning** | Fine-tuned ResNet50 pre-trained on ImageNet |
| ⚡ **GPU Acceleration** | Full CUDA support for fast training (~15ms/image inference on RTX 3050) |
| 📊 **Comprehensive EDA** | Class distribution analysis, sample visualization |
| 🔬 **LIME Explainability** | Interpretable predictions showing which regions influence the model |
| 📈 **Rich Visualizations** | Training curves, confusion matrices, prediction confidence charts |
| 📱 **Android App** | Companion Android application for on-device leaf scanning |
| 🏥 **11 Disease Classes** | Covers 10 tomato diseases + healthy leaves |

---

## 🦠 Disease Classes

The model classifies tomato leaves into **11 categories**:

| # | Class | Description |
|---|---|---|
| 1 | `Bacterial_spot` | Bacterial infection causing dark spots |
| 2 | `Early_blight` | Fungal disease with concentric ring patterns |
| 3 | `Late_blight` | Water mold infection causing rapid decay |
| 4 | `Leaf_Mold` | Fungal growth on leaf surfaces |
| 5 | `Septoria_leaf_spot` | Small circular spots with dark borders |
| 6 | `Spider_mites` | Pest damage from two-spotted spider mites |
| 7 | `Target_Spot` | Concentric target-like lesions |
| 8 | `Tomato_Yellow_Leaf_Curl_Virus` | Viral infection causing leaf curling |
| 9 | `Tomato_mosaic_virus` | Viral mosaic patterns on leaves |
| 10 | `healthy` | No disease detected |
| 11 | `powdery_mildew` | White powdery fungal coating |

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Raw Leaf Images                     │
│              (Train: ~25,848 | Valid: ~6,683)        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                  │
│    • Resize to 224×224                               │
│    • Normalize ([0.5, 0.5, 0.5])                     │
│    • Data Augmentation                               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            ResNet50 (Transfer Learning)              │
│    • Pre-trained on ImageNet                         │
│    • Custom FC: 2048 → 512 → 11 classes              │
│    • Optimizer: Adam | Loss: CrossEntropy            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Evaluation & Prediction                 │
│    • Accuracy, Precision, Recall, F1-Score           │
│    • Confusion Matrix Visualization                  │
│    • LIME Explainability                             │
│    • Single-image Prediction with Confidence         │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

The system uses **ResNet50** with a modified fully connected head:

```python
model = models.resnet50(pretrained=True)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Custom classification head
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 11),       # 11 disease classes
    nn.Softmax(dim=1)
)
```

### Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | ResNet50 (ImageNet pre-trained) |
| **Input Size** | 224 × 224 × 3 |
| **Batch Size** | 32 |
| **Epochs** | 25 |
| **Optimizer** | Adam (lr = 0.001) |
| **Loss Function** | CrossEntropyLoss |
| **Regularization** | Dropout (0.5), BatchNorm |

---

## 🏋️ Training Pipeline

```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Steps:**
1. Forward pass — compute predictions
2. Compute loss — compare with ground truth
3. Backward pass — calculate gradients
4. Update weights — optimizer step
5. Track metrics — accuracy, loss per epoch

---

## 📊 Results & Metrics

| Metric | Value |
|---|---|
| **Validation Accuracy** | ~95% |
| **Inference Speed** | ~15ms per image (RTX 3050) |
| **Loss Convergence** | Rapid reduction via transfer learning |

### Visualizations Generated:
- ✅ Training vs Validation Accuracy Curves
- ✅ Training vs Validation Loss Curves
- ✅ Confusion Matrix (11×11)
- ✅ Per-class Precision, Recall, F1-Score
- ✅ LIME Explanation Maps
- ✅ Sample Predictions with Confidence Bars

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Adithya-J05/Advance-Leaf-Disease-Detection-System-using-ResNet50.git
cd Advance-Leaf-Disease-Detection-System-using-ResNet50

# Install dependencies
pip install torch torchvision matplotlib numpy opencv-python tqdm lime Pillow
```

### Quick Prediction

```python
from Model_Training.predict import predict_image

# Predict on a single leaf image
label, confidence, probabilities = predict_image("path/to/leaf_image.jpg")
print(f"Disease: {label} | Confidence: {confidence*100:.2f}%")
```

---

## 📁 Project Structure

```
Advance-Leaf-Disease-Detection-System-using-ResNet50/
│
├── Model_Training/
│   ├── model_training.ipynb        # Main training notebook (end-to-end pipeline)
│   ├── dataload.py                 # Custom PyTorch Dataset class
│   ├── preprocessing.py            # Image preprocessing & normalization
│   ├── predict.py                  # Single-image prediction with visualization
│   ├── plotaccuracy.py             # Training/validation accuracy plots
│   ├── Confusion_Matrix_Visualization.py  # Confusion matrix generation
│   ├── visualization.py            # Data visualization utilities
│   ├── Lime.py                     # LIME explainability for model predictions
│   ├── EDA.py                      # Exploratory Data Analysis
│   └── readme.md                   # Detailed model training documentation
│
├── model/
│   └── best_model.pth              # Trained ResNet50 model weights (~94 MB)
│
├── .gitignore
└── README.md                       # This file
```

---

## 🛠 Technologies Used

<table>
<tr>
<td><b>Category</b></td>
<td><b>Technology</b></td>
</tr>
<tr><td>Deep Learning Framework</td><td>PyTorch</td></tr>
<tr><td>Model Architecture</td><td>ResNet50 (Transfer Learning)</td></tr>
<tr><td>GPU Acceleration</td><td>NVIDIA CUDA</td></tr>
<tr><td>Image Processing</td><td>OpenCV, Pillow, torchvision</td></tr>
<tr><td>Visualization</td><td>Matplotlib</td></tr>
<tr><td>Explainability</td><td>LIME</td></tr>
<tr><td>Language</td><td>Python 3.x</td></tr>
<tr><td>Mobile App</td><td>Android (companion app)</td></tr>
</table>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>⭐ If you found this project useful, please give it a star!</b>
</p>
