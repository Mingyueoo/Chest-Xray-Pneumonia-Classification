# @Version :1.0
# @Author  : Mingyue
# @File    : predict.py
# @Time    : 30/03/2026 22:11
import matplotlib.pyplot as plt
import torch
from dataset import get_dataloader
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader, classes = get_dataloader("data/chest_xray/test", batch_size=1)

model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

for i, (img, label) in enumerate(loader):
    img = img.to(device)
    pred = torch.argmax(model(img), dim=1).item()

    plt.imshow(img.cpu().squeeze().permute(1,2,0))
    plt.title(f"Pred: {classes[pred]}, True: {classes[label.item()]}")
    plt.show()

    if i == 5:
        break