import torch

from src.dataset import get_dataloader
from src.model import SimpleCNN

TEST_DIR = "data/chest_xray/test"
MODEL_PATH = "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    test_loader, classes = get_dataloader(TEST_DIR)

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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

    print(f"Classes: {classes}")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
