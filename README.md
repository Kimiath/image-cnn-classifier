# FashionMNIST Image Classification with Convolutional Neural Networks

This project builds and trains a Convolutional Neural Network (CNN) using PyTorch to classify images from the FashionMNIST dataset. The model predicts one of 10 clothing categories (e.g., T-shirt, sneaker, bag) using grayscale 28×28 images.

## Overview

The goal of this project is to explore deep learning techniques for image classification by:
- Implementing a custom CNN architecture from scratch  
- Training and evaluating the model on the FashionMNIST dataset  
- Using data augmentation to improve generalization  
- Experimenting with advanced methods such as transfer learning

The final model achieves strong performance on the test set and demonstrates how CNNs learn spatial patterns in image data.

## Model Architecture

The model is built using the PyTorch `nn.Sequential` API and includes:
- **3 convolutional layers** with ReLU activations  
- **Max-pooling** after the first two conv layers  
- **Flattening** followed by two fully connected layers  
- Final **Softmax output** for 10 clothing classes  

Additional techniques explored:
- Dropout (used in later variations)
- Data augmentation (flip, rotation, random resized crop)
- Transfer learning with ResNet18 

# Dataset
The project uses FashionMNIST, could be loaded automatically through torchvision.datasets:
	•	60,000 training images
	•	10,000 test images
	•	10 clothing categories

Images are normalized and transformed using ToTensor()!
