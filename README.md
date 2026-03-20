#  Brain Tumor MRI Classifier

A deep learning model that classifies brain MRI scans into four categories using **transfer learning with Xception**. Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.


---

##  Overview

Brain tumors are among the most critical conditions in medical imaging diagnosis. This project builds an end-to-end image classification pipeline to detect and categorize brain tumors from MRI scans using a pretrained convolutional neural network.

**Classes:**
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

##  Streamlit App

The app lets you upload any brain MRI image and get an instant prediction with confidence scores across all four classes.

**Features:**
- Upload a JPG/PNG MRI scan
- Predicted class with confidence percentage
- Visual confidence bar chart for all classes
- Brief description of each tumor type

**Run locally:**
```bash
streamlit run app.py
```

---

##  Dataset

**Source:** [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset is organized into `Training/` and `Testing/` directories, each with subfolders for each class.

```
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

> **Note:** The dataset is not included in this repository. Download it from Kaggle and update the paths in the notebook accordingly.

---

##  Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building & training |
| Xception | Pretrained base model (ImageNet) |
| Streamlit | Web app |
| NumPy / Pandas | Data handling |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Evaluation metrics |
| Pillow | Image loading |

---

##  Model Architecture

- **Base Model:** Xception (pretrained on ImageNet, top excluded, `pooling='max'`)
- **Custom Head:**
  - Flatten
  - Dropout (0.3)
  - Dense(128, ReLU)
  - Dropout
  - Dense(4, Softmax)
- **Input Shape:** 299 × 299 × 3
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metrics:** Accuracy, Precision, Recall

---

##  Pipeline

```
Raw MRI Images
     │
     ▼
Load & Build DataFrames (train / test)
     │
     ▼
Train / Validation / Test Split
  (test set split 50/50 into validation & test)
     │
     ▼
Data Augmentation (brightness jitter, rescale)
     │
     ▼
Xception Transfer Learning (frozen base)
     │
     ▼
Training (25 epochs, batch size 32)
     │
     ▼
Evaluation (accuracy, loss, confusion matrix, classification report)
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract it locally.

### 4. Update dataset paths

In the notebook, update these lines to point to your local dataset:

```python
tr_df = train_df('/path/to/brain-tumor-mri-dataset/Training')
ts_df = test_df('/path/to/brain-tumor-mri-dataset/Testing')
```

### 5. Train and save the model

Run all cells in the notebook, then save the model:

```python
model.save("brain_tumor_classifier.h5")
```

### 6. Run the app

```bash
python -m streamlit run app.py
```

---

##  Results

The model is evaluated on a held-out test set with:
- **Confusion Matrix** — to visualize per-class performance
- **Classification Report** — precision, recall, F1-score per class
- **Training curves** — accuracy and loss over epochs

---

##  Repository Structure

```
brain-tumor-mri-classifier/
├── brain-tumor-mri-classifier.ipynb   # Training notebook
├── app.py                             # Streamlit web app
├── brain_tumor_classifier.h5          # Saved model weights
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical diagnosis.

---

##  Acknowledgements

- Dataset by [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar) on Kaggle
- Xception architecture by François Chollet