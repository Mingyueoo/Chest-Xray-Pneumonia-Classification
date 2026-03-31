# @Version :1.0
# @Author  : Mingyue
# @File    : train.py
# @Time    : 30/03/2026 20:15
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, classes = get_dataloader("data/chest_xray/train")

model = SimpleCNN(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pth")