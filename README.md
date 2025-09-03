Heart Failure Mortality Prediction

This project predicts the mortality risk of patients with heart failure using Support Vector Machine (SVM) and Artificial Neural Network (ANN) models. It is based on clinical patient records and aims to assist in early detection of high-risk cases.

Features

Uses Heart Failure Clinical Records dataset (299 patients, 13 features).

Implements SVM and ANN models for binary classification.

Provides data preprocessing, model training, and evaluation in one Python file.

Performance measured using accuracy, precision, recall, F1-score, and confusion matrix.

Includes visualizations for better understanding of dataset and model performance.

Tech Stack

Python

pandas, numpy → Data handling

matplotlib, seaborn → Visualization

scikit-learn → SVM, preprocessing, evaluation

tensorflow / keras → ANN model

Workflow

Load and preprocess dataset

Perform Exploratory Data Analysis (EDA)

Train SVM and ANN models

Evaluate models using multiple metrics

Predict patient survival from new inputs

Future Improvements

Hyperparameter tuning for better performance

Deploy models with Flask / FastAPI

Build GUI dashboard for doctors and caregivers
