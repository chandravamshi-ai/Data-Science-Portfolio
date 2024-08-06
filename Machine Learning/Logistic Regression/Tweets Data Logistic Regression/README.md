# Twitter Sentiment Analysis

## Overview

This project analyzes the sentiment of tweets using Natural Language Processing (NLP) and machine learning techniques. By classifying tweets into positive, negative, and neutral sentiments, it provides insights into public opinions on various topics.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
  
## Features

- Collection and preprocessing of tweet data
- Exploratory Data Analysis (EDA) with visualizations
- Feature extraction using TF-IDF
- Sentiment classification using Logistic Regression
- Model evaluation and performance metrics
- Informative visualizations of sentiment distribution and word clouds

## Dataset

The dataset used in this project is the Twitter training dataset from Kaggle. It includes the following columns:
- `ID`: Unique identifier for each tweet
- `Category`: Category or topic of the tweet
- `Sentiment`: Sentiment label (positive, negative, neutral)
- `Tweet`: The tweet text

## Data Preprocessing

Data preprocessing steps include:
- Cleaning the text (removing special characters, converting to lowercase, etc.)
- Removing stopwords
- Tokenizing and lemmatizing words
- Handling missing values

## Model Training

The model is trained using Logistic Regression with TF-IDF feature extraction. The training process involves:
- Splitting the dataset into training and testing sets
- Extracting features using `TfidfVectorizer`
- Training the Logistic Regression model
- Saving the trained model for future predictions

## Evaluation

The model's performance is evaluated using accuracy score on both the training and test sets.

## Visualization

Several visualizations are generated to provide insights into the data:
- Sentiment distribution
- Word clouds for each sentiment
- Distribution of tweet lengths by sentiment
