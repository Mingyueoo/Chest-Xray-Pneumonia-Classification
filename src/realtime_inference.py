# @Version :1.0
# @Author  : Mingyue
# @File    : realtime_inference.py
# @Time    : 30/03/2026 20:15
import cv2
import torch
from src.model import SimpleCNN
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)  # 打开摄像头
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(img), dim=1).item()
    label = "Tumor" if pred == 1 else "Normal"
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.imshow("Realtime Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()