# Plant-Disease-Detection-System
A machine learning based lant disease detection web application using deep learning (MobileNetV2 + TensorFlow) that identifies plant leaf diseases from uploaded images and provides prediction confidence, disease information, and remedies through a modern Flask-based web interface.

 **Features**

 1. Plant leaf disease detection using Deep Learning  
 2. Detects 38 plant disease classes  
 3. Upload leaf image through web interface  
 4. Displays prediction confidence score  
 5. Shows disease description and remedy  
 6. Modern responsive UI design  
 7. Built using MobileNetV2 + TensorFlow  
 6. Flask-based deployment  

---

# Model Information

This project uses:

- **MobileNetV2**
- **Transfer Learning**
- **TensorFlow**
- **CNN**

The model was trained on the **PlantVillage Dataset** containing:

- 87,000+ plant leaf images
- 38 disease classes
- 14 plant species

# Supported Plants

The model can detect diseases from:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper Bell
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

---

# Disease Classes

Examples of detectable diseases:

- Tomato Early Blight
- Tomato Late Blight
- Apple Black Rot
- Grape Black Rot
- Potato Early Blight
- Corn Common Rust
- Strawberry Leaf Scorch
- Pepper Bacterial Spot
- Powdery Mildew
- Healthy Leaves

and many more...

---

# Technologies Used

## Backend
- Python
- Flask

## Machine Learning
- TensorFlow
- MobileNetV2
- NumPy
- OpenCV

## Frontend
- HTML
- CSS

---

# Dataset

Download dataset from Kaggle:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

---

# Application Workflow

1️. User uploads leaf image  
2️. Image preprocessing performed  
3️. MobileNetV2 model predicts disease  
4️. Disease name + confidence displayed  
5️. Description & remedy shown on webpage  

---

# Model Performance

- Model Architecture: MobileNetV2
- Training Type: Transfer Learning
- Accuracy: ~80–90% (depending on epochs)
- Dataset Classes: 38

---

# Future Improvements

- Real-time camera detection
- Mobile app integration
- Cloud deployment
- Multi-language support
- Fruit disease detection
- Disease severity estimation

---
