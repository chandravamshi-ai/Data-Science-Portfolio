# Breast Cancer Diagnosis Predictor

**A live version of the application is available on** [Breast Cancer Diagnosis Predictor](https://breast-cancer-prediction-lr.streamlit.app/)

![Live version Image](https://github.com/chandravamshi-ai/Streamlit-Breast-Cancer-Prediction/blob/master/imgs/breast%20cancer%20prediction%20live.png)


## Overview

The Breast Cancer Diagnosis app is a tool designed to help medical professionals diagnose breast cancer using machine learning. The app predicts whether a breast mass is benign or malignant based on specific measurements. It visually represents the input data using a radar chart and shows the predicted diagnosis along with the probability of being benign or malignant. Measurements can be manually input or obtained directly from a cytology lab machine, though the app itself does not include the lab connection.

This app was developed as an educational exercise using the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Note that this dataset may not be entirely reliable for professional use.


## Installation

To manage dependencies, it's recommended to run this app inside a virtual environment. Using `conda`, you can create a new environment named `breast-cancer-diagnosis`:

```bash
conda create -n breast-cancer-diagnosis python=3.10 
```

Activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

Install the required packages:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies.

## Usage

To start the app, run the following command:

```bash
streamlit run streamlit_app.py
```

This will launch the app in your default web browser. You can upload an image of cells for analysis and adjust various settings to customize the analysis. Once satisfied with the results, you can export the measurements to a CSV file for further analysis.
