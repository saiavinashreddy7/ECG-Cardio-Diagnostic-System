# 🩺 ECG Cardio Diagnostic System

A **Deep Learning + Machine Learning–based ECG Classification System** that automatically classifies **ECG images into 4 cardiac conditions**.

This project replicates a research-level medical pipeline using **Custom CNN**, **ML classifiers**, **feature extraction**, and **5-Fold Cross Validation** to achieve high accuracy.

## 📚 Classes in the Dataset
- **NP – Normal Person**
- **AH – Abnormal Heartbeat**
- **MI – Myocardial Infarction**
- **HMI – History of Myocardial Infarction**

## 📦 Dataset Overview

**Total Images:** **928**

| Class | Description | Count |
|-------|-------------|--------|
| NP | Normal Person | 284 |
| AH | Abnormal Heartbeat | 233 |
| MI | Myocardial Infarction | 239 |
| HMI | History of MI | 172 |

### 📁 Folder Structure

<img width="122" height="116" alt="image" src="https://github.com/user-attachments/assets/afb2366d-00eb-453f-8360-bf30dd475132" />

## 🛠️ Preprocessing Steps
Every ECG image undergoes:

- ✂️ **Cropping** – remove header/footer text  
- 📏 **Resize** → 227 × 227 × 3  
- 🎚️ **Normalization** → pixel / 255  
- 🔄 **Data Augmentation**  
  - Rotation  
  - Horizontal Flip  
  - Translation  
  - Zoom  

**Final Augmented Dataset:** ~ **4400+ images**

## 🔄 5-Fold Cross Validation

Dataset split per fold:

- **Training images:** 742  
- **Testing images:** 186  

This gives reliable accuracy and minimizes overfitting.

## 🧠 Model Architectures

### 1️⃣ Custom CNN (Deep Learning Model)

- Conv → LeakyReLU → BatchNorm → MaxPool  
- Conv → LeakyReLU → BatchNorm → MaxPool  
- Conv → LeakyReLU → BatchNorm → MaxPool  
- Dense branch  
- Feature branch  
- 🔗 Concatenation  
- 1×1 Convolution  
- Dense (512)  
- Dense (4) + **Softmax** output  

### 2️⃣ Machine Learning Classifiers (Using CNN Features)

- 🌲 **Random Forest**  
- 📘 **Gaussian NB**  
- 🔢 **KNN**  
- 📈 **SVM**  
- 🧩 **MLP**  

## 📊 Model Performance (5-Fold Average)

| Model | Accuracy |
|--------|-----------|
| ⭐ **Random Forest** | **93.10%** |
| KNN | 82.65% |
| GaussianNB | 76.83% |
| MLP | 48.93% |
| SVM | 30.60% |
| Custom CNN | ~38% |

## 🧰 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **OpenCV**
- **Albumentations**
- **NumPy**
- **Matplotlib / Seaborn**

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python train_cnn.py
   python extract_features.py
   python train_ml_models.py
   python predict.py --image test.jpg


---
 
   
