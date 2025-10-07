# ViT_assignment
Q1. 
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

Dataset
The notebook uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.



Q2. This project implements an image segmentation pipeline that combines the capabilities of GroundingDINO for text-prompted object detection and Segment Anything Model 2 (SAM 2) for precise instance segmentation.

## Pipeline Description

The workflow is as follows:

1. Load Image: An input image is loaded.
2. Accept Text Prompt: The user provides a text description of the object to be segmented.
3. Text-to-Box (GroundingDINO): GroundingDINO processes the image and the text prompt to detect objects matching the description and output bounding boxes with confidence scores.
4. Select Best Box: The bounding box with the highest confidence score is selected as the region of interest.
5. Box-to-Mask (SAM 2): The selected bounding box is used as a prompt for SAM 2, which generates a precise segmentation mask for the object within that box.
6. Display Results: The original image is displayed with the predicted bounding box from GroundingDINO and the final segmentation mask from SAM 2 overlaid.

This pipeline allows for zero-shot segmentation of objects based on natural language descriptions.

## Example Output

Here is an example demonstrating the pipeline's output for a laptop:
<img width="636" height="790" alt="image" src="https://github.com/user-attachments/assets/22694cfc-4c3d-4976-aeb6-e08b048b6ded" />

## Limitation

1. Detection Dependent: Accuracy relies on GroundingDINO's ability to detect the object from text.
2. Threshold & Prompt Sensitivity: Results are sensitive to detection threshold and prompt clarity.
3. Box Quality Impacts Mask: SAM 2's mask quality depends on GroundingDINO's bounding box accuracy.
