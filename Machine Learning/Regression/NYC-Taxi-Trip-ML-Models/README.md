# NYC Taxi Trip ML Models

This repository contains a series of machine learning models developed to analyze and predict various outcomes related to NYC yellow taxi trips. The data is sourced from the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), specifically for January 2019.

## Project Overview

The goal of this project is to implement and compare different machine learning algorithms to predict key metrics such as trip duration and fare amount using NYC taxi trip data. This project covers the following steps:

1. **Data Import and Exploration**: 
   - Load and explore the dataset to understand its structure and key features.
   
2. **Data Cleaning and Preparation**:
   - Process the data by handling missing values, converting data types, and engineering new features.
   
3. **Feature Engineering**:
   - Create new features from existing data to improve model accuracy, including temporal features like time of day and day of the week.
   
4. **Modeling**:
   - Implement and train several machine learning models including:
     - Linear Regression
     - Random Forest
     - XGBoost
     
5. **Model Evaluation**:
   - Evaluate the models using metrics such as RMSE, R-squared, and MAE, and compare their performance.

## Dataset

The dataset used in this project is publicly available and can be accessed [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The specific file used is the yellow taxi trip data for January 2019.

## Results


   ### Random Forest Classification
   
   |                       | Predicted: 0 | Predicted: 1 |
   |-----------------------|--------------|--------------|
   | **Actual: 0**         | 10893        | 4790         |
   | **Actual: 1**         | 3788         | 14425        |
   
   - **Accuracy**: `0.7469`
   - **F1 Score**: `0.7469`
   - **Precision**: `0.7507`
   - **Recall**: `0.7920`

   ---
   
   ### Random Forest Regression
   
   | Metric                    | Value        |
   |---------------------------|--------------|
   | **Mean Absolute Error (MAE)**  | `7.6459`     |
   | **Mean Squared Error (MSE)**   | `168.4280`   |
   | **Root Mean Squared Error (RMSE)** | `12.9780` |
   | **R² Score**                | `0.4396`     |
   
   **Best Hyperparameters**:
   
  ```
  {
    "n_estimators": 600,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 500,
    "bootstrap": False
  }
  ```
     
   ---
   
   ### Gradient Boosting Regressor
   
   | Metric                    | Value        |
   |---------------------------|--------------|
   | **Mean Absolute Error (MAE)**  | `8.8332`     |
   | **Mean Squared Error (MSE)**   | `188.7850`   |
   | **Root Mean Squared Error (RMSE)** | `13.7399` |
   | **R² Score**                | `0.3719`     |

   ---
   
   ### Decision Tree Regression
   
   | Metric                    | Value        |
   |---------------------------|--------------|
   | **Mean Absolute Error (MAE)**  | `10.4349`    |
   | **Mean Squared Error (MSE)**   | `252.0874`   |
   | **Root Mean Squared Error (RMSE)** | `15.8773` |
   | **R² Score**                | `0.1393`     |
   
   
