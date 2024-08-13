# Employee Attrition Prediction Model
## Overview
This project aims to predict employee attrition and department classification using a deep learning model. The model takes employee data as input and outputs the probability of attrition and department classification.

## Data

Employee data (features):
- Age
- HourlyRate
- DistanceFromHome
- Education
- YearsAtCompany
- StockOptionLevel
- YearsSinceLastPromotion
- JobInvolvement
- JobSatisfaction
- OverTime
Attrition (target):
- Binary classification (0: stayed, 1: left)
Department (target):
- Multi-class classification (0: Sales, 1: Technical, 2: Support)

## Model
- Deep neural network with two output branches:

Attrition branch (binary classification):
 - Sigmoid activation function
 - Binary cross-entropy loss

Department branch (multi-class classification):
 - Softmax activation function
 - Categorical cross-entropy loss

## Metrics

- Total loss (combined loss of both branches)
- Attrition accuracy
- Department accuracy

## Improvement Suggestions
- Handle class imbalance using techniques like oversampling, undersampling, or weighted losses
- Experiment with different architectures, such as adding more layers or using transfer learning
- Try alternative activation functions or optimizers
- Use techniques like feature engineering or dimensionality reduction to improve data quality
- Collect more data to increase the model's generalizability

## Usage

- Preprocess data using preprocess_data.py
- Train the model using train_model.py
- Evaluate the model using evaluate_model.py
- Use the model for predictions on new data

## Requirements
- Python 3.x
- Keras 2.x
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
