import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load and Balance Dataset
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")
df = pd.concat([train_df, test_df])

def fix_labels(labels):
    return np.where((labels >= 10) & (labels <= 24), labels - 1, labels)

df['label'] = fix_labels(df['label'].values)

grouped = df.groupby("label")
balanced = grouped.apply(lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

train_indices, test_indices = [], []
for label in balanced['label'].unique():
    class_indices = balanced[balanced['label'] == label].index
    train_idx, test_idx = train_test_split(class_indices, test_size=60, random_state=42)
    train_indices.extend(train_idx)
    test_indices.extend(test_idx)

train_data = balanced.loc[train_indices].reset_index(drop=True)
test_data = balanced.loc[test_indices].reset_index(drop=True)

# Dataset
class SignLanguageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.X = dataframe.drop("label", axis=1).values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        self.y = dataframe["label"].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).unsqueeze(0)
        return image, label

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = SignLanguageDataset(train_data, transform=transform)
test_dataset = SignLanguageDataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# CNN Model 
class CNN2D(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
            nn.Linear(128* 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2D().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = {'train_loss': [], 'train_acc': []}

for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(acc)
    print(f"Epoch {epoch+1}/30 - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")

# Evaluation 
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Results 
test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nTest Accuracy: {test_acc:.4f}")
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 31), history['train_loss'], marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, 31), history['train_acc'], marker='o')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()
# Save Model & Training History 
torch.save(model.state_dict(), "sign_language_2dcnn.pt")
with open("pytorch_training_history.json", "w") as f:
    json.dump(history, f)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# lassification Report
report = classification_report(y_true, y_pred, digits=4)
print("Classification Report:\n")
print(report)


