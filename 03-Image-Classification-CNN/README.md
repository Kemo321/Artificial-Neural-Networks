# Project 03: 50-Class Image Classifier (Custom CNN)

## Task Description
The goal of this project is to develop an image classifier capable of assigning input images into one of 50 distinct categories (including various everyday objects, animals, and items). 

## Constraints
* **Strict Limitation:** The use of pre-trained architectures (e.g., ResNet, VGG, EfficientNet via Transfer Learning) was strictly forbidden. The task required designing, initializing, and training a **completely custom architecture from scratch**.

## Technical Approach
* **Architecture:** A deep Convolutional Neural Network (CNN) designed using PyTorch's `nn.Module`.
* **Key Components:**
  * Multiple convolutional blocks (`nn.Conv2d`) with increasing filter sizes (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256).
  * **Batch Normalization** (`nn.BatchNorm2d`) after splotch layers to stabilize training and accelerate convergence.
  * **Max Pooling** (`nn.MaxPool2d`) for spatial dimensionality reduction.
  * Fully connected classifier layers with **Dropout** (p=0.5) to robustly combat overfitting.
* **Data Augmentation:** Implemented `torchvision.transforms` (RandomHorizontalFlip, RandomRotation, and ColorJitter) to artificially increase training sample diversity.
* **Evaluation:** Average accuracy computed across all 50 target classes using the provided `helpers.py` framework.