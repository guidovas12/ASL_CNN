import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict

def main():
    print("\U0001F680 Starting training...")

    # ===== 1. Config =====
    train_dir = "split_dataset/train"
    val_dir = "split_dataset/validation"
    batch_size = 32
    num_epochs = 10
    num_classes = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F5A5️ Using device: {device}")

    # ===== 2. Transform (RGB, Resize, Normalize Only) =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ===== 3. Load and Sample N Per Class =====
    def sample_n_per_class(dataset, n):
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            class_indices[label].append(idx)

        selected = []
        for label, indices in class_indices.items():
            if len(indices) >= n:
                selected.extend(random.sample(indices, n))
            else:
                raise ValueError(f"Class {label} has only {len(indices)} images, need at least {n}")
        return Subset(dataset, selected)

    full_train_set = datasets.ImageFolder(train_dir, transform=transform)
    full_val_set = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(full_val_set, batch_size=batch_size, shuffle=False)

    print(f"\U0001F9EA Sampled {len(full_train_set)} training images")
    print(f"\U0001F9EA Sampled {len(full_val_set)} validation images")
    print(f"\U0001F8BE Classes: {full_train_set.classes}")

    # ===== 4. CNN Model =====
    class ASLCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(128 * 28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = ASLCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== 5. Training with Validation Tracking =====
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"\n\U0001F4DA Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    # ===== 6. Plot Loss and Accuracy =====
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ===== 7. Final Evaluation =====
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n\U0001F3AF Validation Accuracy: {acc:.2%}")
    print(classification_report(all_labels, all_preds, target_names=full_train_set.classes, digits=3))

    # ===== 8. Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_train_set.classes,
                yticklabels=full_train_set.classes)
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ===== 9. Save Model =====
    torch.save(model.state_dict(), "asl_cnn_rgb_224_per_class.pth")
    print("\U0001F4BE Model saved as 'asl_cnn_rgb_224_per_class.pth'")

if __name__ == "__main__":
    main()
