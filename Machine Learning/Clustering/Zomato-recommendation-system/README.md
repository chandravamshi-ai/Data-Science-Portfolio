# Zomato Restaurant Recommendation System

## Overview

This project implements a restaurant recommendation system using Zomato dataset. It utilizes various data analysis and machine learning techniques to provide personalized restaurant suggestions based on user preferences and restaurant attributes.

## Table of Contents

1. [Data Exploration](#data-exploration)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Recommendation System](#recommendation-system)

## Data Exploration

The notebook begins by importing necessary libraries and loading the Zomato dataset. It examines the dataset's structure, including column names, data types, and basic statistics.

## Data Preprocessing

This section focuses on cleaning and preparing the data for analysis:

- Handling missing values
- Removing duplicates
- Converting data types
- Renaming columns for clarity

## Feature Engineering

 features are transformed to enhance the recommendation system:
- Applying text processing techniques to cuisine and menu data

## Recommendation System

The core of the project, this section implements the recommendation algorithm:

- Using the TF-IDF algorithm creating TF-IDF vector on Reviews Column
- then finding cosine similarity to find the similar restautrant based on similarity in reviews

## Usage

To use this notebook:

1. Ensure you have Jupyter Notebook installed
2. Install required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
3. Download the Zomato dataset
4. Run the cells in order, following the instructions within the notebook
