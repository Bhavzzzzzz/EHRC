import torch
from torch.utils.data import DataLoader
from dataset import WSIDataset
from model import MILModel

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = WSIDataset("normalized_dataset", "labels.csv")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = MILModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for patches, labels in loader:
        patches = patches.squeeze(0).to(device)
        labels = labels.squeeze(0).to(device)

        optimizer.zero_grad()

        outputs = model(patches)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss {total_loss/len(loader)}")

torch.save(model.state_dict(), "model.pth")