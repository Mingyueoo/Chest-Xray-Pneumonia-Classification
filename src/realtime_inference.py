import cv2
import torch
from PIL import Image

from src.dataset import build_transform
from src.model import SimpleCNN

MODEL_PATH = "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = build_transform()
    classes = ["NORMAL", "PNEUMONIA"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        img = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).item()

        label = classes[pred]
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Realtime Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
