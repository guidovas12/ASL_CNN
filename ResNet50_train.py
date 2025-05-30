# sign_language_recognition.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data paths
train_dir = "split_dataset/train"
validation_dir = "split_dataset/validation"

# Data transformations (corrected for ResNet-50)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]),
    "validation": transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]),
}

# Load datasets
image_datasets = {
    "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
    "validation": datasets.ImageFolder(validation_dir, transform=data_transforms["validation"]),
}

# Data loaders
dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
    "validation": DataLoader(image_datasets["validation"], batch_size=32, shuffle=False),
}

# Number of classes
num_classes = len(image_datasets["train"].classes)
class_names = image_datasets["train"].classes
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Load pre-trained ResNet50 and modify final layer
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training parameters
num_epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets["train"])
    epoch_acc = running_corrects.double() / len(image_datasets["train"])
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item()) 

    print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["validation"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(image_datasets["validation"])
    val_acc = val_running_corrects.double() / len(image_datasets["validation"])
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    print(f"Validation Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

# Plot training and validation loss over epochs
plt.figure(figsize=(14, 6))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", color="blue")
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report on Validation Set:")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataloaders["train"].classes,
            yticklabels=dataloaders["train"].classes)
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), "sign_language_resnet50_model.pth")
print("\nTraining complete. Model saved as sign_language_resnet50_model.pth")
