


# =========================================
# General Imports
# =========================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# =========================================
# PyTorch Imports
# =========================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================
# Extra Libraries
# =========================================
from torchvision import transforms
from PIL import Image
import zipfile as zf
import pickle
from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive to access datasets

import cv2
import numpy as np

# =========================================
# Helper Function: Compute perceptual hash of an image
# Used for duplicate detection
# =========================================
def get_image_hash(img):
    small = cv2.resize(img, (8,8))                 # Resize to small 8x8 image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    avg = gray.mean()                               # Compute average pixel value
    hash_bits = (gray > avg).astype(int)           # Binary hash
    return ''.join(hash_bits.flatten().astype(str))

# =========================================
# Step 1: Extract image paths from ZIP
# =========================================
zip_path = '/content/drive/MyDrive/CatDog.zip'
valid_paths = []

with zf.ZipFile(zip_path, 'r') as zip_ref:
    filelist = zip_ref.namelist()
    for folder in ['CatDog/Cat','CatDog/Dog']:
        # Collect all jpg files under each folder
        img_files = [f for f in filelist if f.startswith(folder+'/') and f.endswith('.jpg')]
        valid_paths.extend(img_files)

# =========================================
# Step 2: Remove duplicate images using perceptual hashing
# =========================================
unique_paths = []
seen_hashes = set()

with zf.ZipFile(zip_path, 'r') as zip_ref:
    for img_path in valid_paths:
        try:
            img_array = np.frombuffer(zip_ref.read(img_path), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img_hash = get_image_hash(img)
            if img_hash not in seen_hashes:
                unique_paths.append(img_path)
                seen_hashes.add(img_hash)
        except Exception as e:
            print(f'Error processing image {img_path}: {e}')

print(f'Unique and valid images: {len(unique_paths)}')

# Save unique image paths for future use
safe_path = '/content/drive/MyDrive/unique_image_paths.pkl'
with open(safe_path, 'wb') as f:
    pickle.dump(unique_paths, f)
print("✅ Saved unique_images successfully!")

# =========================================
# Step 3: Visualize Class Distribution
# =========================================
from collections import Counter

classes = [path.split('/')[1] for path in unique_paths]  # Extract class name
class_counts = Counter(classes)
print(class_counts)

plt.figure(figsize=(6,4))
plt.bar(class_counts.keys(), class_counts.values(), color=['black','blue'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()

# =========================================
# Step 4: Data Transformations for Augmentation
# =========================================
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),        # Random crop for augmentation
    transforms.RandomHorizontalFlip(),        # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # Normalize to [-1,1]
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),             # Resize to fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# =========================================
# Step 5: Custom Dataset Class to read images directly from ZIP
# =========================================
class ZipImageDataset(Dataset):
    def __init__(self, zip_path, image_paths, transform=None):
        self.zip_path = zip_path
        self.image_paths = image_paths
        self.transform = transform

        # Open ZIP file once to avoid reopening on every getitem
        self.zip_ref = zf.ZipFile(self.zip_path, 'r')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Determine label: 0 = Cat, 1 = Dog
        label = 0 if "/Cat/" in img_path else 1
        
        # Read image from zip
        img_data = self.zip_ref.read(img_path)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================================
# Step 6: Train-Test Split
# =========================================
from sklearn.model_selection import train_test_split

train_paths, test_paths = train_test_split(
    unique_paths,
    test_size=0.2,
    random_state=42,
    stratify=[0 if "Cat" in p else 1 for p in unique_paths]  # Keep class balance
)

# =========================================
# Step 7: DataLoaders
# =========================================
train_dataset = ZipImageDataset(zip_path, train_paths, transform=train_transforms)
test_dataset  = ZipImageDataset(zip_path, test_paths,  transform=test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,     # Important for Colab to avoid crashes
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =========================================
# Step 8: Define CNN Model
# =========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, lr=1e-3):
        super(SimpleCNN, self).__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # --- Loss & Optimizer ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss_history = []

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # ================================
    # Training Function
    # ================================
    def train_model(self, dataloader, num_epochs):
        self.train()
        self.loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        return self.loss_history

    # ================================
    # Evaluation Function
    # ================================
    def evaluate(self, dataloader):
        self.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        return all_labels, all_preds

    # ================================
    # Plot Training Loss
    # ================================
    def plot_loss(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.loss_history, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss per Epoch")
        plt.grid()
        plt.show()

    # ================================
    # Plot Confusion Matrix
    # ================================
    def plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # ================================
    # Save Model Function
    # ================================
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved to:", path)

# =========================================
# Step 9: Initialize model and move to device (CPU/GPU)
# =========================================
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes)
model.to(device)
print('Model moved to:', device)

# =========================================
# Step 10: Train the model
# =========================================
epoch = 10
train_losses = model.train_model(train_loader, epoch)

# =========================================
# Step 11: Evaluate on Test Set
# =========================================
y_true, y_pred = model.evaluate(test_loader)
print("✅ Evaluation done!")

# Print metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))

