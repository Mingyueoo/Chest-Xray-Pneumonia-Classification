# Chest-Xray-Pneumonia-Classification

**Chest X-ray Pneumonia Detection using Deep Learning (PyTorch)**

---

## 📌 Project Overview

This project demonstrates an end-to-end deep learning pipeline for medical image classification using chest X-ray images. The goal is to classify images into **NORMAL** and **PNEUMONIA** categories.

The project showcases practical skills in medical imaging, deep learning model development, and real-world dataset handling.

---

## 🧠 Key Features

- Medical image preprocessing pipeline
- CNN-based classification model (PyTorch)
- Training and evaluation on real-world dataset
- Model performance evaluation (accuracy)
- Visualization of predictions
- Modular and reproducible code structure

---

## 📊 Dataset
The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset, provided by Paul Mooney on Kaggle.

- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- **Content:** The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). 

### How to access the data:
1. Download the dataset from the Kaggle link above.
2. Extract the ZIP file.
3. Place the `chest_xray` folder into the `data/` directory of this project.

**Note:** The `data/` folder is included in `.gitignore` to avoid uploading large binary files to GitHub.

---
## ⚙️ Model Architecture

- Convolutional Neural Network (CNN)
- 2 Convolution layers + MaxPooling
- Fully connected classification layers

---

## 🚀 Training

```bash
python src/train.py
```
- Trained for 5 epochs
- Final training loss: ~0.05

### Result:
- Test Accuracy: 79.97%

---
## 🎥 Prediction & Visualization

The following gallery showcases a selection of random prediction results from the test set. These examples demonstrate the model's ability to distinguish between NORMAL lung structures and the opacities characteristic of PNEUMONIA.
```bash
python src/predict.py
```

Example outputs:

| Prediction | Ground Truth |
| ---------- | ------------ |
| PNEUMONIA  | NORMAL       |
| NORMAL     | NORMAL       |


(See `/examples` folder for sample outputs)

| Case 1 (Normal) | Case 2 (Normal) | Case 3 (Normal) |
| :---: | :---: | :---: |
| ![Fig1](examples/Figure_1.png) | ![Fig2](examples/Figure_2.png) | ![Fig3](examples/Figure_3.png) |
| **Pred: PNEUMONIA** | **Pred: PNEUMONIA** | **Pred: NORMAL** |
| (Label: NORMAL) | (Label: NORMAL) | (Label: NORMAL) |

| Case 4 (Pneumonia) | Case 5 (Normal) | Case 6 (Pneumonia) |
| :---: | :---: | :---: |
| ![Fig4](examples/Figure_4.png) | ![Fig5](examples/Figure_5.png) | ![Fig6](examples/Figure_6.png) |
| **Pred: PNEUMONIA** | **Pred: NORMAL** | **Pred: PNEUMONIA** |
| (Label: PNEUMONIA) | (Label: NORMAL) | (Label: NORMAL) |

---
## 📁 Project Structure
```text
MedImage-AI-Demo/
│
├─ data/
│   └─ chest_xray/
├─ examples/
├─ src/
│   ├─ dataset.py
│   ├─ model.py
│   ├─ train.py
│   ├─ evaluate.py
│   └─ predict.py
│
├─ model.pth
├─ requirements.txt
└─ README.md
```
---
## 🛠️ Technologies Used
- Python
- PyTorch
- OpenCV
- Matplotlib
- NumPy
---
## 💡 Key Learnings
- Built an end-to-end deep learning pipeline for medical image classification
- Worked with real-world medical imaging datasets
- Implemented model training, evaluation, and visualization
- Gained experience in CNN-based image analysis
---
## 📌 Future Improvements
- Use pre-trained models (ResNet, EfficientNet)
- Add data augmentation
- Improve model generalization
- Deploy as a web-based application
---
## 📝 License
This project is for educational and demonstration purposes.