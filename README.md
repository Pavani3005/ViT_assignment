# ViT_assignment
#q1
Vision Transformer (ViT) for CIFAR-10 Classification
This notebook implements a Vision Transformer model from scratch in PyTorch to classify images from the CIFAR-10 dataset.

Overview
The notebook covers the following steps:

Setup: Imports necessary libraries and sets up the device (GPU or CPU).
Configuration: Defines hyperparameters for the model and training process.
Data Preparation: Downloads, transforms, and loads the CIFAR-10 dataset using PyTorch DataLoaders. Data augmentation is applied to the training set.
Model Architecture: Defines the Vision Transformer architecture, including:
PatchEmbedding: Converts input images into a sequence of flattened patches and projects them into a higher-dimensional space.
TransformerEncoder: Implements a standard Transformer encoder layer with multi-head attention and an MLP block.
VisionTransformer: Combines the patch embedding, learnable CLS token, positional embeddings, and a stack of Transformer encoders with a final classification head.
Training Setup: Defines the loss function (Cross-Entropy), optimizer (AdamW), and learning rate scheduler (Cosine Annealing).
Training & Evaluation Functions: Implements functions for training one epoch and evaluating the model on the test set.
Main Training Loop: Runs the training process for a specified number of epochs, saves the best model based on test accuracy, and prints training and evaluation metrics per epoch.
Requirements
PyTorch
Torchvision
Tqdm
These libraries are typically pre-installed in Google Colab environments.

Usage
Run all cells: Execute the cells sequentially from top to bottom.
Monitor Training: Observe the training progress and metrics printed in the output of the training loop cell. The best test accuracy achieved will be reported at the end.
Best Model: The best performing model weights will be saved to a file named best_vit_cifar10.pth.
Configuration
You can adjust the model and training hyperparameters in the "Configuration" cell (e.g., IMG_SIZE, PATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, etc.) to experiment with different settings.

Dataset
The notebook uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
