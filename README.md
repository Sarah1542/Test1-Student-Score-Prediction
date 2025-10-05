# 🎓 Student Score Predictor: Simple Linear Regression

This project applies supervised machine learning to predict a student's exam score based on their study hours. The goal is to establish a strong, quantifiable relationship between the input feature (study time) and the target variable (exam score) using a simple linear model.

---

## 📁 Dataset

The dataset contains information relating the time students spent studying to the final scores they achieved in an exam.

| Feature | Description |
| :--- | :--- |
| **Hours** | The number of hours a student studied per day (Input Feature - X). |
| **Scores** | The score obtained by the student in the exam (Target Variable - y). |

> 📂 File used: `student_scores.csv`

---

## 🔧 Steps Performed

1.  **Data Exploration & Cleaning** 🧹
    * Loaded the dataset and examined basic statistics (`.info()`, `.describe()`).
    * Confirmed the absence of missing or null values.
    * Visualized the **strong positive linear relationship** between Hours and Scores.

2.  **Feature Preparation** 📏
    * Selected `Hours` as the independent variable (X) and `Scores** as the dependent variable (y).
    * Reshaped the feature array (X) as required by the `scikit-learn` library.

3.  **Data Splitting** ✂️
    * Split the dataset into an **80% training set** and a **20% testing set** to evaluate performance on unseen data.

4.  **Model Training** 🧠
    * Trained the **Simple Linear Regression** model on the training data.

5.  **Model Evaluation** ⭐
    * Used the trained model to predict scores on the test set.
    * Evaluated performance using **R-squared ($R^2$)** and **Mean Squared Error (MSE)**.

6.  **Visualization** 🖼️
    * Plotted the test data points and the resulting **Regression Line** to visually assess the model's fit.

7.  **Prediction Output** 🎯
    * Used the final model to make predictions for new, arbitrary study hours (e.g., 9.25 hours).

---

## 📊 Visualizations

* 📈 Scatter plot showing the initial linear relationship between Study Hours and Exam Scores.
* 🎨 **Regression Line Plot:** A scatter plot of the test data with the best-fit line (the linear regression model) drawn through the points, highlighting the model's accuracy.

---

## 📦 Libraries Used

```bash
pandas
numpy
matplotlib.pyplot
seaborn
scikit-learn (LinearRegression, train_test_split, metrics)
