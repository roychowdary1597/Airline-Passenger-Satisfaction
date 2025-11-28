# ✈️ **Airline Passenger Satisfaction Analysis & ANN Model**

## 📌 **Overview**
This project analyzes an **Airline Passenger Satisfaction Survey dataset** to understand which factors influence passenger satisfaction.  
Using Exploratory Data Analysis (EDA) and an **Artificial Neural Network (ANN)**, the project predicts whether a passenger is *Satisfied* or *Dissatisfied*.

---

## 🎯 **Problem Statement**
Airlines collect customer feedback, but identifying the root causes of satisfaction or dissatisfaction is challenging.  
This project aims to analyze key service and demographic factors and build a predictive ANN model to support data-driven decision-making.

---

## 🎯 **Objectives**
- Perform EDA on airline passenger satisfaction data (100k+ rows).  
- Identify major factors influencing satisfaction.  
- Build an **ANN classification model**.  
- Evaluate performance using accuracy, precision, recall, and AUC.  
- Provide insights for improving airline services.

---

## 📂 **Dataset Information**
- **Dataset Size:** Worked with more than 100,000 rows  
- **Features Include:**  
  - Demographics → Age, Gender, Travel Type, Class  
  - Service Ratings → WiFi, Food, Cleanliness, Seat Comfort  
  - Flight Details → Arrival/Departure Delay  
  - **Target Variable** → Satisfaction

---

## 🛠️ **Technologies Used**
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## 🔍 **Project Workflow**

### 1️⃣ **Data Preprocessing**
- Handling missing values  
- Label encoding categorical features  
- Normalizing numerical columns  
- Removing duplicate records  

### 2️⃣ **Exploratory Data Analysis**
- Satisfaction distribution analysis  
- Correlation between service ratings and satisfaction  
- Delay impact analysis  
- Visualization using heatmaps & bar charts  

### 3️⃣ **ANN Model Development**
- Input Layer: All processed features  
- Hidden Layers: Dense layers with ReLU  
- Output Layer: Sigmoid (Binary classification)  
- Loss Function: Binary Cross-Entropy  
- Optimizer: Adam  

### 4️⃣ **Model Evaluation**
- Accuracy  
- Precision, Recall  
- Confusion Matrix  
- ROC-AUC  

---

## 📊 **Key Insights**
- WiFi service, Seat Comfort, Cleanliness, and Online Boarding strongly influence satisfaction.  
- Long delays reduce satisfaction significantly.  
- Business-class passengers show higher satisfaction levels.  
- ANN model achieved strong prediction performance.

---



