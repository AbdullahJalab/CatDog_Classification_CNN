# CatDog Classification with SimpleCNN

This project demonstrates how to classify images of cats and dogs using a Convolutional Neural Network (CNN) in PyTorch. The workflow is implemented in Google Colab for easy experimentation.

## Features
- Reads images directly from a ZIP archive without extracting.
- Detects and removes corrupted and duplicate images.
- Applies data augmentation and normalization.
- Implements a simple CNN with 3 convolutional layers and 2 fully connected layers.
- Tracks training loss and evaluates performance with Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Compatible with GPU (CUDA) for faster training.

## Dataset
The dataset should be structured as a ZIP file containing two folders:CatDog/
├── Cat/
└── Dog/
if  you want datasset send me an email.
هب 


## Usage
1. Open the project in Google Colab.
2. Mount your Google Drive and provide the path to the ZIP dataset.
3. Run the notebook cells to preprocess the images, train the CNN, and evaluate it.

## Results
- Training and evaluation metrics are printed.
- Confusion matrix and training loss are visualized.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- matplotlib
- seaborn
