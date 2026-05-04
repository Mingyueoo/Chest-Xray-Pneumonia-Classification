import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloader
from src.model import SimpleCNN

TEST_DIR = "data/chest_xray/test"
MODEL_PATH = "model.pth"
NUM_EXAMPLES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    loader, classes = get_dataloader(TEST_DIR, batch_size=1, shuffle=True)

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    for i, (img, label) in enumerate(loader):
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).item()

        plt.imshow(img.cpu().squeeze().permute(1, 2, 0))
        plt.title(f"Pred: {classes[pred]}, True: {classes[label.item()]}")
        plt.axis("off")
        plt.show()

        if i + 1 >= NUM_EXAMPLES:
            break


if __name__ == "__main__":
    main()
