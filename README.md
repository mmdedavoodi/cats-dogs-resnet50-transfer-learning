# 🐱🐶 Cats vs Dogs Classification with CNN Transfer Learning & Fine-Tuning

This repository demonstrates **image classification** of cats and dogs using **PyTorch**, leveraging **ResNet50** for **transfer learning** and **fine-tuning**. The project combines **feature extraction**, **SVM classification**, and **deep learning fine-tuning** for high-accuracy predictions.

---

## 📚 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
* [Training and Evaluation](#training-and-evaluation)
* [Results & Visualization](#results--visualization)
* [License](#license)

---

## 🏗 Project Overview

This project aims to classify images of **cats** and **dogs** by:

1. Loading and preprocessing the dataset 🖼️
2. Using **pretrained ResNet50** for feature extraction 🌟
3. Training an **SVM classifier** on extracted features 🧠
4. Building a **custom fine-tuned CNN** for end-to-end learning 🔥
5. Evaluating model performance with metrics and visualization 📊

---

## 📂 Dataset

We use the **Cats and Dogs dataset (filtered subset)**:

* Source: [Microsoft Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
* Structure:

```
cats_and_dogs_filtered/
├── train/
│   ├── cats/
│   └── dogs/
└── validation/
    ├── cats/
    └── dogs/
```

* Classes: `cats` 🐱 and `dogs` 🐶
* Split: 80% train, 20% validation

---

## ⚙ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib opencv-python tqdm gdown
```

3. Download the dataset (handled automatically in notebook):

```python
!gdown 1IcAf8TmM2T7HB_jvb-cKela94_ZIPuqP
!unzip cats_and_dogs_filtered.zip
```

---

## 🛠 Usage

1. **Load dataset** and apply transformations:

```python
train_transforms = t.Compose([
    t.ToTensor(),
    t.RandomHorizontalFlip(p=0.5),
    t.RandomRotation(degrees=20),
    t.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6),
    models.ResNet50_Weights.IMAGENET1K_V2.transforms()
])
```

2. **Create PyTorch Dataset & DataLoader**:

```python
DataSetTrain = DataSetCatDog(train_files, transform=train_transforms)
Dataloader_Train = torch.utils.data.DataLoader(DataSetTrain, batch_size=64, shuffle=True)
```

3. **Feature extraction with ResNet50**:

```python
features_train = model(image_batch)
```

4. **Train SVM classifier** (optional for fast testing):

```python
from sklearn.svm import SVC
model_svm = SVC(kernel="rbf")
model_svm.fit(features_train, labels_train)
```

5. **Fine-tune CNN model**:

```python
model = BackBoneCatDog()
model.freeze()  # train only classifier head first
train_model(model, Dataloader_Train, Dataloader_Test, optimizer, loss_function, device)
model.unfreeze()  # then fine-tune entire backbone
```

6. **Plot training curves**:

```python
draw_plot(train_loss, train_acc, test_loss, test_acc)
```

---

## 🏗 Model Architecture

* **Backbone**: ResNet50 pretrained on ImageNet
* **Custom Classifier**: Fully connected layer (2048 → 1)
* **Activation**: Sigmoid for binary classification
* Supports **frozen backbone** training and **full fine-tuning**

---

## 🖌 Data Preprocessing & Augmentation

To increase generalization and prevent overfitting:

* Random horizontal flips ↔️
* Random rotations 🔄
* Color jitter (brightness, contrast, saturation) 🌈
* Normalization using ResNet50 pretrained stats

---

## 📊 Training & Evaluation

* **Loss function**: `BCEWithLogitsLoss` ⚡
* **Optimizer**: Adam 💪
* **Batch size**: 64
* **Metrics**: Accuracy, Confusion Matrix, Classification Report

**Training phases:**

1. 🔒 **Frozen backbone**: train classifier head only
2. 🔓 **Unfrozen backbone**: fine-tune full network

**Example evaluation:**

```python
plt.imshow(DataSetTest[i][0].permute(1,2,0))
res = model(DataSetTest[i][0].to(device).unsqueeze(0)).sigmoid().round().item()
print(f"Predicted label: {res}")
```

---

## 📈 Results & Visualization

* High classification accuracy with both **SVM** and **fine-tuned CNN**
* Confusion matrix and training curves for insights
* Visual inspection of predictions on test images

---

## 📜 License

MIT License – free to use, modify, and distribute for personal or academic projects.

---

**Author:** Your Name
**Contact:** [your.email@example.com](mailto:your.email@example.com)

---

## 🌟 Optional Enhancements

* Add **GIFs of training**
* Include **example predictions on multiple test images**
* Add **GitHub badges** for PyTorch, Python, and license
