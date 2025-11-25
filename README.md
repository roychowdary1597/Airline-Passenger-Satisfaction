✈️ Airline Passenger Satisfaction Analysis & ANN Classification Model
📌 Overview

This project focuses on analyzing an Airline Passenger Satisfaction Survey dataset to understand the factors that influence customer satisfaction.
Using Exploratory Data Analysis (EDA) and an Artificial Neural Network (ANN) model, the project identifies key drivers of satisfaction and predicts whether a passenger is Satisfied or Dissatisfied.

🎯 Problem Statement

Airlines collect large amounts of survey data, but identifying the exact causes of customer satisfaction remains difficult.
The goal of this project is to analyze survey data, extract insights, and build a predictive ANN model that helps airlines improve customer experience and reduce dissatisfaction.

🎯 Objectives

Analyze and preprocess airline survey data (~100k+ rows).

Identify key factors that influence passenger satisfaction.

Build an ANN classification model to predict satisfaction.

Evaluate model performance using accuracy, precision, recall, and AUC.

Provide data-driven insights for airlines to improve service quality.

📂 Dataset Information

Dataset Size: 100,000+ records

Features Include:

Demographics: Age, Gender, Type of Travel, Class

Service ratings: WiFi, Food & Drink, Cleanliness, Seating Comfort, etc.

Flight-related: Departure delay, Arrival delay

Target: Satisfaction (Satisfied / Neutral or Dissatisfied)

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn

ML Framework: TensorFlow/Keras

Modeling: ANN (Artificial Neural Network)

Environment: Jupyter Notebook / VS Code

🔍 Workflow
1️⃣ Data Preprocessing

Handling missing values

Label encoding categorical features

Normalizing continuous variables

Removing duplicates

2️⃣ Exploratory Data Analysis

Distribution of satisfaction levels

Correlation analysis

Impact of delays & service ratings

Visualization of high-impact features

3️⃣ ANN Model Development

Input Layer: All selected features

Hidden Layers: Dense layers with ReLU

Output Layer: Sigmoid (Binary classification)

Loss Function: Binary Cross-Entropy

Optimizer: Adam

4️⃣ Evaluation Metrics

Accuracy

Precision & Recall

Confusion Matrix

ROC-AUC

📊 Key Insights

In-flight WiFi, Seat Comfort, Cleanliness, and Online Boarding significantly affect satisfaction.

Passengers with long delays are more likely to be dissatisfied.

Business-class passengers generally report higher satisfaction.

ANN model performs well with high prediction accuracy.
