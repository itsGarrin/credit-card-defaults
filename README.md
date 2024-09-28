# Credit Card Default Prediction

## Overview
This project focuses on predicting whether a credit card client will default on their next payment, using the **Credit Card Default UCI dataset**. By applying various machine learning algorithms, we aim to build a model that can help banks and financial institutions make informed decisions on credit approvals, risk assessment, and managing portfolios.

## Dataset
The dataset contains information on **30,000 credit card clients** from a Taiwanese bank, with features related to demographic and financial factors. The target variable indicates whether the client defaulted on their next payment.

### Features
The dataset includes the following columns:

1. **ID**: Unique identifier of each client.
2. **LIMIT_BAL**: The total credit limit assigned to the client.
3. **SEX**: Gender (1 = male, 2 = female).
4. **EDUCATION**: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others).
5. **MARRIAGE**: Marital status (1 = married, 2 = single, 3 = others).
6. **AGE**: Age of the client.
7. **PAY_0 to PAY_6**: History of past payments (from April to September, where -1 = pay duly, 1 = payment delay, etc.).
8. **BILL_AMT1 to BILL_AMT6**: The bill statement amounts from April to September.
9. **PAY_AMT1 to PAY_AMT6**: Amounts paid in previous months.
10. **Default Payment (target)**: Binary variable (1 = default, 0 = no default).

The dataset can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

## Project Goals
The main goals of this project are:
- Perform **exploratory data analysis (EDA)** to gain insights into the dataset.
- Preprocess the data, handling missing values and scaling features.
- Implement **machine learning models** to predict default rates, such as:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - Neural Networks
- Evaluate the performance of these models using metrics like **accuracy, precision, recall, F1-score, and AUC**.
- Provide an analysis of feature importance to understand which factors contribute most to defaults.

## Project Structure

```plaintext
├── data
│   └── credit_card_default.csv    # Dataset file
├── notebooks
│   └── EDA.ipynb                  # Jupyter notebook for exploratory data analysis
│   └── model_training.ipynb       # Jupyter notebook for model training and evaluation
├── src
│   ├── preprocessing.py           # Data cleaning and preprocessing functions
│   ├── train_model.py             # Functions for training models
│   ├── evaluate_model.py          # Functions for evaluating models
├── README.md                      # Project overview (this file)
└── requirements.txt               # Dependencies for the project