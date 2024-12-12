# Brain-Tumor-Classification-MRI-with-explainable-SOTA-model

Welcome to the **Brain Tumor Classification MRI** project! This repository focuses on classifying brain MRI images into four distinct categories using state-of-the-art (SOTA) deep learning models. In addition to achieving high classification accuracy, this project emphasizes explainability by providing gradient-weighted class activation map (Grad-CAM) visualizations for model interpretation.

## Features

- **Classification Categories**:
  - **Benign Tumor**
  - **Malignant Tumor**
  - **Pituitary Tumor**
  - **Normal**
  
- **Dataset**:
  - The dataset used is available on [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data).
  - The dataset contains high-quality MRI scans categorized into the aforementioned classes.

- **Models Used**:
  - **MaxViT**
  - **SwinV2-S**
  - **EfficientNetV2-S**
  - **ResNet50**
  - **VGG16**

- **Explainability**:
  - Model decisions are explained using **Grad-CAM** visualizations, highlighting the regions of the MRI images that influenced the predictions.
  - Evaluation metrics such as accuracy, precision, recall, and F1-score are provided for performance analysis.
