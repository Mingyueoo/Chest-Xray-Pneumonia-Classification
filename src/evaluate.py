# @Version :1.0
# @Author  : Mingyue
# @File    : evaluate.py
# @Time    : 30/03/2026 20:15
import torch
from dataset import get_dataloader
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader, _ = get_dataloader("data/chest_xray/test")

model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")