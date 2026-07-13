# Cars Data Analysis and Machine Learning

## Project Overview

This project analyzes a cars dataset using Python. It includes data cleaning, exploratory data analysis (EDA), and machine learning models for regression and classification.

---

## Dataset

- Dataset Name: Cars Data1.csv
- Number of Records: 432
- Number of Features: 15

---

## Part 1: Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Checked the dataset shape and data types.
- Identified missing values.
- Filled missing numerical values using the median.
- Removed duplicate records.
- Converted `MSRP` and `Invoice` columns to numeric.
- Converted categorical columns to the category data type.
- Saved the cleaned dataset as `cleaned_data.csv`.

---

## Exploratory Data Analysis (EDA)

The following visualizations were created:

- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

Correlation analysis was performed using Pearson and Spearman methods.

---

## Part 2: Machine Learning

### Regression Model

- Target Variable: **MSRP**
- Model: Linear Regression
- Evaluation Metrics:
  - Mean Squared Error (MSE)
  - R² Score

A Ridge Regression model was also built and compared with Linear Regression.

---

### Classification Model

- Target Variable: High Price Car (`MSRP > Median`)
- Model: Logistic Regression

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score

Threshold analysis was performed using thresholds from **0.30 to 0.70**.

A second Logistic Regression model with **C = 0.01** was trained and compared with the default model (**C = 1.0**).

Bootstrap sampling (500 iterations) was used to calculate the confidence interval for the AUC difference.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Files Included

- Cars_EDA_Assignment.ipynb
- Cars_EDA_Assignment.py
- Cars Data1.csv
- cleaned_data.csv
- README.md

---

## Conclusion

This project demonstrates the complete machine learning workflow, including data preprocessing, visualization, regression, classification, and model evaluation. The models were evaluated using appropriate performance metrics, and the results were interpreted to understand their effectiveness.