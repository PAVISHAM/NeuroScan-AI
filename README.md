# NeuroScan-AI
# 🧠 NeuroScan AI: Brain Tumour Classifier

NeuroScan AI is a deep learning-based web application designed to classify brain MRI images using a VGG16-based Convolutional Neural Network (CNN). The system allows users to upload an MRI scan and receive a predicted tumour category along with the model's confidence score.

> **Note:** This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

---

## 📌 Project Overview

Brain tumour classification from MRI images requires careful analysis of medical images. NeuroScan AI demonstrates how Artificial Intelligence and Deep Learning can be applied to medical image classification.

The project uses a **VGG16-based CNN model** to analyse MRI images and classify them into predefined categories. The trained model is integrated into a **Streamlit web application**, providing a simple and interactive interface for users to upload an MRI image and view the prediction.

---

## 🎯 Objectives

- To develop a deep learning-based brain MRI classification system.
- To apply CNN techniques for medical image classification.
- To use VGG16 and transfer learning for extracting meaningful image features.
- To preprocess MRI images before model prediction.
- To predict the category of an uploaded MRI image.
- To display prediction confidence along with the result.
- To integrate the trained model into a user-friendly Streamlit application.

---

## 🧠 Classification Categories

The system classifies MRI images into the following categories:

- 🔴 **Meningioma Tumour**
- 🟣 **Pituitary Tumour**
- 🟢 **No Tumour**

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming and model development |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| VGG16 | Pre-trained CNN architecture |
| CNN | Image classification |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Streamlit | Web application development |

---

## 🔄 System Workflow

```text
MRI Image Upload
       ↓
Image Preprocessing
       ↓
Image Resizing
       ↓
VGG16-based CNN Model
       ↓
Feature Extraction
       ↓
Tumour Classification
       ↓
Prediction + Confidence Score
       ↓
Streamlit Web Interface
