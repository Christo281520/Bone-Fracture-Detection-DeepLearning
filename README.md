# Bone-Fracture-Detection-DeepLearning
Bone fracture detection using machine learning / deep learning techniques for classifying X-ray images.


Bone Fracture Detection Using Deep Learning

A complete machine learning + deep learning pipeline for detecting bone fractures from X-ray images.
This project performs:

Image loading

Preprocessing

Segmentation

Feature extraction

ANN model training

Evaluation (confusion matrix + classification report)

UI interface using IPyWidgets

Web UI using Gradio

📁 Repository Structure
Bone-Fracture-Detection-ML/
│
├── Bone_fracture_detection.ipynb   ← Full project notebook
├── fracture_ann.h5                 ← Saved ANN model
├── model_architecture.png          ← Neural network architecture
└── README.md

📦 Dataset

The dataset is downloaded from Kaggle:

🔗 https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays

It contains:

train/ → fractured / not fractured

val/ → fractured / not fractured

Dataset is automatically downloaded using:

import kagglehub
path = kagglehub.dataset_download("vuppalaadithyasairam/bone-fracture-detection-using-xrays")

🛠 Technologies Used
Deep Learning

TensorFlow

Keras

ANN classifier

ImageDataGenerator

Image Processing

OpenCV

Segmentation (contours)

SIFT keypoint visualization

GLCM, LBP, Chain code features

ML Tools

Scikit-learn

Confusion Matrix

Classification Report

User Interfaces

IPyWidgets

Gradio Web App

🧠 Project Workflow
1️⃣ Data Extraction

Dataset ZIP is extracted using:

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")

2️⃣ Data Loading

Images are loaded using OpenCV:

image = cv2.imread(file)
image = cv2.resize(image, (224, 224))

3️⃣ Preprocessing
✔ Segmentation

Contours are extracted to highlight bone structures.

contours, _ = cv2.findContours(...)

✔ Normalization
image = image / 255.0

⭐ Model Architecture (ANN)

model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
📈 Training Results
Epoch	Training Accuracy	Validation Accuracy
1	54%	63%
5	76%	73%
7	79%	84%
9	81%	86%
14	83%	88%

✔ Best validation accuracy: 88.7%

📊 Evaluation
✔ Confusion Matrix

Generated using:

cm = confusion_matrix(y_test, pred)

✔ Classification Report
precision recall f1-score support
fractured         xx     xx      xx     xxx
not fractured     xx     xx      xx     xxx

🖥️ User Interfaces
🔸 1. Jupyter UI (IPyWidgets)

Allows uploading an X-ray and viewing prediction immediately.

upload_button = widgets.FileUpload(...)


Predicts:

Fractured / Not Fractured

🔸 2. Gradio Web App UI

Run this in notebook:

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs="text",
    title="Bone Fracture Detection",
    description="Upload an X-ray image to classify fracture."
).launch()


Launches a live ML web application.

🧪 How to Run This Project
Option 1 — Google Colab

Upload notebook

Upload fracture_ann.h5

Run all cells

Option 2 — Local PC
pip install tensorflow opencv-python numpy matplotlib seaborn gradio
python app.py

🚀 Future Improvements

Replace ANN with CNN (VGG16 / ResNet50)

Grad-CAM heatmaps

Better segmentation pipeline

Data augmentation

Flask or FastAPI deployment

👨‍💻 Developed By

Christo Thomas
MCA Graduate (2023–2025)
Full-Stack Developer | ML Enthusiast
📧 Email: crisssthomas15@gmail.com

🔗 GitHub: https://github.com/Christo281520
