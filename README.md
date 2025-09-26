# respiratory-disease-prediction
MSc dissertation project: Machine learning models (CNN, SVM, XGBoost) for predicting respiratory diseases from audio recordings and demographic data.

# Respiratory Disease Prediction Using Audio Features and Machine Learning

> 🎓 MSc Dissertation Project – University of Essex (2025)  
> 🩺 Predicting respiratory diseases from audio recordings and demographic data using machine learning models.

---

## 🚀 Overview
This project explores **non-invasive diagnosis of respiratory diseases** using audio features (MFCCs, spectrograms) and demographic data (age, gender, BMI).  
Machine learning models such as **CNN, SVM, XGBoost, and Ensembles** were applied to classify diseases from the [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database).

---

## 🔑 Features
- Audio preprocessing: MFCC & log-mel spectrogram extraction  
- Demographic integration (age, gender, BMI, etc.)  
- Models: CNN, SVM (tuned), XGBoost, and ensemble approaches  
- Evaluation: Accuracy, F1-score, confusion matrices, ROC curves  
- Handles class imbalance with oversampling (`imblearn`)  
- Figures for feature importance, demographic analysis, and errors  

---

## 📊 Results
- CNN achieved **~XX% accuracy** on the test set  
- Tuned SVM improved recall for minority classes  
- XGBoost identified **Age, Crackles, and Wheezes** as key predictors  
- See [docs/dissertation.pdf](docs/dissertation.pdf) for detailed results  

---

## ⚡ Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/rishimano/respiratory-disease-prediction.git
cd respiratory-disease-prediction
