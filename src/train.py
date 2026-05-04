import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloader
from src.model import SimpleCNN

TRAIN_DIR = "data/chest_xray/train"
VAL_DIR = "data/chest_xray/val"
MODEL_PATH = "model.pth"
EPOCHS = 5
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy


def main():
    train_loader, classes = get_dataloader(TRAIN_DIR, shuffle=True)
    val_loader, _ = get_dataloader(VAL_DIR)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
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

        train_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy * 100:.2f}%"
        )

    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
