🩺 ECG Cardio Diagnostic System

A Deep Learning + Machine Learning–based ECG Classification System that automatically classifies ECG images into 4 cardiac conditions:

NP – Normal Person

AH – Abnormal Heartbeat

MI – Myocardial Infarction

HMI – History of Myocardial Infarction

This project replicates a research-level ECG diagnostic pipeline using CNNs, ML classifiers, and 5-fold cross-validation.

✨ Features

🧠 Custom CNN model for ECG feature extraction

🌲 Random Forest, Naïve Bayes, KNN, SVM, MLP machine learning classifiers

🔍 512-dimensional deep features extracted from CNN

🔄 5-Fold Cross Validation for reliable accuracy

📊 Training graphs, confusion matrices & evaluation metrics

🖼️ Automated preprocessing (crop, resize, normalization, augmentation)

🚀 High accuracy (Random Forest ≈ 93%)

📂 Dataset Details
Total Images: 928
Class	Description	Count
NP	Normal Person	284
AH	Abnormal Heartbeat	233
MI	Myocardial Infarction	239
HMI	History of MI	172
Folder Structure
ECG-Dataset/
│── NP/
│── AH/
│── MI/
└── HMI/

🛠️ Preprocessing Pipeline

Every ECG image goes through:

✂️ Cropping (removes top/bottom text)

📏 Resize → 227 × 227 × 3

🎚️ Normalization → pixel / 255

🔄 Augmentation

Rotation

Horizontal flip

Translation

Zoom

Final Augmented Dataset Size: ~ 4400+ images
🔄 5-Fold Cross Validation

Dataset split per fold:

Split	Images
Training	742
Testing	186

➡ Guarantees stable performance and reduces overfitting.

🧠 Models Used
1️⃣ Custom CNN Architecture

3× Conv → LeakyReLU → BatchNorm → MaxPool

Dense + Conv feature branch

Concatenation

1×1 Convolution

Dense (512)

Output: Dense (4 classes, Softmax)

2️⃣ Machine Learning Classifiers (on CNN features)

Trained on 512-dimensional CNN feature vectors:

🌲 Random Forest

📘 Gaussian Naïve Bayes

🔢 KNN

📈 SVM

🧩 MLP (Neural Network)

📊 Results (Average over 5 folds)
Model	Accuracy
⭐ Random Forest	93.10%
KNN	82.65%
GaussianNB	76.83%
MLP	48.93%
SVM	30.60%
Custom CNN	~38%

➡ Random Forest performed best due to strong handling of high-dimensional CNN features.

🧮 Why Random Forest Achieved Highest Accuracy?

Combines multiple decision trees

Handles non-linear ECG patterns

Robust to noise

Works well with medium-sized datasets

Reduces overfitting using bagging

🧰 Tech Stack

Python

TensorFlow / Keras – CNN model

scikit-learn – ML classifiers

OpenCV – image preprocessing

albumentations – augmentation

NumPy / Pandas – feature storage

Matplotlib / Seaborn – visualizations

▶️ How to Run the Project
pip install -r requirements.txt
python train_cnn.py
python extract_features.py
python train_ml_models.py

Predict on a new ECG:
python predict.py --image test.jpg

🩻 System Workflow Diagram (optional - ask me to generate)
ECG Image → Preprocessing → CNN Feature Extraction → ML Classifier → Final Prediction
