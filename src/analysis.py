from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.dataset import get_dataloader
from src.model import SimpleCNN

TEST_DIR = "data/chest_xray/test"
MODEL_PATH = "model.pth"
EXAMPLES_DIR = Path("examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    test_loader, classes = get_dataloader(TEST_DIR)

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probabilities[:, 1].cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, classes)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        digits=4,
    )
    print("Classification Report:")
    print(report)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"Sensitivity ({classes[1]} recall): {sensitivity:.4f}")
    print(f"Specificity ({classes[0]} recall): {specificity:.4f}")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    save_roc_curve(fpr, tpr, auc)
    print(f"ROC AUC: {auc:.4f}")


def save_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(EXAMPLES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def save_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(EXAMPLES_DIR / "roc_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
