import torch
from torch.utils.data import DataLoader, random_split
from dataset import WSIDataset
from model import MILModel
from sklearn.metrics import roc_auc_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = WSIDataset("normalized_dataset", "labels.csv")

# Split dataset (70-30)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1)

# Model
model = MILModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------- TRAIN --------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for patches, labels in train_loader:
        patches = patches.squeeze(0).to(device)
        labels = labels.squeeze(0).to(device)   

        optimizer.zero_grad()

        outputs = model(patches)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss {total_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), "model.pth")

# -------- EVALUATION --------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for patches, labels in test_loader:
        patches = patches.squeeze(0).to(device)
        labels = labels.squeeze(0)

        outputs = model(patches)
        probs = torch.sigmoid(outputs).cpu()

        all_preds.append(probs)
        all_labels.append(labels)

# Convert to numpy
all_preds = torch.stack(all_preds).numpy()
all_labels = torch.stack(all_labels).numpy()

print("\n--- AUROC Scores ---")

for i, name in enumerate(["IDH1R132H","ATRX","P53"]):
    try:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        print(f"{name}: {auc:.3f}")
    except:
        print(f"{name}: Not enough data")