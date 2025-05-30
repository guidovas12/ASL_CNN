import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import string

# Dataset Loader 
class ASL3DDataset(Dataset):
    def __init__(self, X_path, y_path):
        assert os.path.exists(X_path), "File not found: {}".format(X_path)
        assert os.path.exists(y_path), "File not found: {}".format(y_path)
        self.X = torch.from_numpy(np.load(X_path).astype(np.float32)) # alreday normalised
        self.y = torch.from_numpy(np.load(y_path).astype(np.int64))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].permute(1, 0, 2, 3)  # [D, C, H, W] -> [C, D, H, W]
        return x, self.y[idx]

#2. 3D CNN Model
class ASL3DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 3. Training & Evaluation
def main():
    X_train_path = "D:/3dcnn/combined/X_train.npy"
    y_train_path = "D:/3dcnn/combined/y_train.npy"
    X_test_path  = "D:/3dcnn/combined/X_test.npy"
    y_test_path  = "D:/3dcnn/combined/y_test.npy"
    train_dataset = ASL3DDataset(X_train_path, y_train_path)
    test_dataset = ASL3DDataset(X_test_path, y_test_path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    print(" Train classes:", np.unique(y_train))
    print(" Test classes:", np.unique(y_test))
    num_classes = np.max(y_train) + 1
    model = ASL3DCNN(num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #  Training 
    start_time = time.time()
    losses = []
    for epoch in range(30):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print("Epoch {}/30 | Loss: {:.4f}".format(epoch+1, epoch_loss))
    elapsed_time = time.time() - start_time
    print("Training completed in {:.2f} seconds".format(elapsed_time))

    # Plot Loss Curve 
    plt.plot(range(1, 31), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=3)
    print(" Test Accuracy: {:.2%}".format(acc))
    print(report)

    plt.figure(figsize=(14, 10))
    labels = list(string.ascii_uppercase)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Accuracy: {:.2%})".format(acc))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "D:/3dcnn/combined/ssssssssasl_3dcnn.pth")

if __name__ == "__main__":
    main()
