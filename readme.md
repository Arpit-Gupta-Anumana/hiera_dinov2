
# Adapting Foundation Models for Ultrasound Segmentation
### A PyTorch Implementation of DINOv2 + Hiera for the CAMUS Dataset

This repository contains the code for a deep learning model that combines the **DINOv2** and **Hiera** vision foundation models for real-time semantic segmentation of cardiac ultrasound images from the CAMUS dataset. The primary goal is to accurately segment the Left Ventricle (LV) and Left Atrium (LA).

The architecture and methodology are based on the principles of adapting large, pretrained models for specialized downstream tasks, as outlined in recent research.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Technical Approach](#technical-approach)
3.  [Model Architecture](#model-architecture)
4.  [Project Structure](#project-structure)
5.  [Setup & Installation](#setup--installation)
6.  [Usage](#usage)
7.  [Results](#results)
8.  [Citation](#citation)

## Introduction

Semantic segmentation of medical images, particularly ultrasound, presents unique challenges due to noise, artifacts, and anatomical variability. This project tackles this challenge by leveraging two powerful, pretrained Vision Foundation Models:

*   **Hiera**: A hierarchical vision transformer that excels at capturing multi-scale structural and spatial relationships within an image. It serves as the architectural backbone of our encoder.
*   **DINOv2**: A self-supervised model renowned for learning rich, fine-grained texture and local pattern features. It acts as a powerful feature enhancer.

The core innovation is the **interleaving** of DINOv2's detailed features into each stage of the Hiera backbone. This creates an enriched, hybrid feature representation that is then passed to a U-Net-style decoder to generate a precise, pixel-perfect segmentation mask.

## Technical Approach

*   **Hybrid Encoder**: Utilizes frozen, pretrained backbones of Hiera and DINOv2 to extract a powerful combination of hierarchical and texture-based features.
*   **Transfer Learning**: Only a small number of newly added layers (an adapter for DINOv2 and the decoder) are trained. This makes the training process highly data-efficient and computationally feasible.
*   **U-Net Style Decoder**: A standard expansive-path decoder progressively upsamples the feature maps, using skip connections from the encoder to recover spatial precision and produce a high-resolution output mask.
*   **Rich Data Augmentation**: Employs the `albumentations` library for a robust set of spatial and pixel-level augmentations, making the model resilient to variations in image orientation and quality.
*   **Specialized Loss Function**: Uses the `DiceCELoss` from the MONAI library, a compound loss function that combines the benefits of both Dice Loss (for handling class imbalance and overlap) and Cross-Entropy Loss (for pixel-wise accuracy).

## Model Architecture

The model follows a classic encoder-decoder structure with the novel interleaved feature injection.

```
Input Image (B, 3, 224, 224)
 │
 │      ┌───────────────────────────┐      ┌──────────────────────────┐
 └─────►│     Hiera Encoder         ├─────►│ Hiera Features (4 stages)│
        │    (Frozen Backbone)      │      └──────────────────────────┘
        └───────────────────────────┘                │ (1)
                                                     │
        ┌───────────────────────────┐      ┌──────────────────────────┐
        │     DINOv2 Extractor      ├─────►│ DINOv2 Feature (16x16)   │
        │    (Frozen Backbone)      │      └──────────────────────────┘
        └───────────────────────────┘                │
                  │ (Project to 144 channels)        │ (2)
                  ▼                                  │
        ┌───────────────────────────┐                │
        │ Projected DINOv2 Feature  │◄───────────────┘
        └───────────────────────────┘

ENCODER FEATURES (Interleaved)
 │
 ├─ Stage 1: Hiera Feat 1 (56x56) + Interp. DINOv2 Feat -> Skip Connection 1
 ├─ Stage 2: Hiera Feat 2 (28x28) + Interp. DINOv2 Feat -> Skip Connection 2
 ├─ Stage 3: Hiera Feat 3 (14x14) + Interp. DINOv2 Feat -> Skip Connection 3
 └─ Stage 4: Hiera Feat 4 (7x7)   + Interp. DINOv2 Feat -> To Decoder Bottom
                                                               │
                                                     ┌─────────▼─────────┐
                                                     │  DECODER (U-Net)  │
                                                     │ (Trainable)       │
                                                     └─────────▲─────────┘
                                                               │
                                                     ┌─────────┴─────────┐
                                                     │ Segmentation Head │
                                                     │ (Trainable 1x1 Conv)│
                                                     └─────────▼─────────┘
                                                               │
                                                     Final Mask (B, 3, 224, 224)
```

## Project Structure

```
HIERA-DINOV2/
├── checkpoints/                # Saved model weights will be stored here
├── data/
│   └── CAMUS_public/           # Root directory for the CAMUS dataset
│       ├── database_nifti/
│       └── database_split/
├── hiera/                      # Cloned official Hiera repository
├── src/
│   ├── datasets/
│   │   └── ultrasound_dataset.py # Defines the PyTorch Dataset for CAMUS
│   ├── models/
│   │   └── ultrasound_segmenter.py # Defines the Hiera+DINOv2 model architecture
│   ├── utils.py                # Helper functions (e.g., dice_score metric)
│   └── train.py                # Main script to run the training process
├── .gitignore
└── README.md
```

## Setup & Installation

Follow these steps to set up the environment and run the project.

**1. Prerequisites**
*   Python 3.11+
*   Git

**2. Clone the Repository**
```bash
git clone <your-repo-url>
cd HIERA-DINOV2
```

**3. Set Up Python Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

**4. Install Dependencies**
It is recommended to create a `requirements.txt` file with the following content and install via pip.

*(File: `requirements.txt`)*
```
torch
torchvision
torchaudio
tqdm
nibabel
monai
albumentations
scikit-image
matplotlib
```
Install with:
```bash
pip install -r requirements.txt```

**5. Download Hiera Source Code**
The model architecture depends on the official Hiera code. Clone it into the project's root directory.
```bash
git clone https://github.com/facebookresearch/hiera.git
```

**6. Download the CAMUS Dataset**
*   Download the dataset from the official CAMUS challenge website: [https://www.creatis.insa-lyon.fr/Challenge/camus/](https://www.creatis.insa-lyon.fr/Challenge/camus/)
*   Extract the contents into the `data/` folder, such that the final path is `data/CAMUS_public/`.

## Usage

**1. Configure Paths**
Before running, open `src/train.py` and ensure the `CAMUS_ROOT_DIR` variable points to the correct location of your dataset.

```python
# In src/train.py
CAMUS_ROOT_DIR = "/path/to/your/project/data/CAMUS_public"
```

**2. Run Training**
To start the training process, run the `train.py` script as a module from the root project directory:

```bash
python -m src.train
```

*   The script will print progress to the console, including training loss, validation loss, and the average Dice Score for each epoch.
*   The best performing model (based on validation Dice score) will be saved to `checkpoints/best_model.pth.tar`.

**3. Run Model Test**
To verify the model architecture and ensure all components are working correctly before training, you can run the model's test script directly:

```bash
python src/models/ultrasound_segmenter.py
```
A successful run will print `Model test successful with correct output shape!`.

## Results

After training is complete, the best model checkpoint is available at `checkpoints/best_model.pth.tar`. This checkpoint contains the model's trained weights and can be loaded for inference on new, unseen ultrasound images.

During training, you should observe the following trends in the console output:
*   **Training and Validation Loss:** Should generally decrease over time.
*   **Average Dice Score:** Should generally increase over time, indicating better segmentation overlap.

Performance on the test set should be evaluated separately to gauge the model's final accuracy.

## Citation
If you use this work, please consider citing the original paper that inspired the architectural design (placeholder).

```
[Paper Citation Here]
```