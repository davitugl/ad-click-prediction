## About Project
This project focuses on predicting digital advertisement clicks using a dataset of user behaviors. Instead of just chasing a high score, the goal was to extract actionable business insights: **Who** clicks, **when** do they click, and **why**?
The **result:** An optimized Logistic Regression model with a **99.2% AUC**, (**accuracy: 98%**) capable of cutting through the noise to identify high-conversion users.
**Feature Engineering**: I replaced noisy geographical data with custom-built features like is_weekend. This single change significantly boosted the model’s predictive power.
**Model Interpretation**: Through Feature Importance analysis, I demonstrated that **weekends** and **age** act as strong positive predictors. Conversely, **daily time spent on site**, **daily internet usage**, and **area income** exhibit a significant negative correlation with the target variable
**Optimization**: Hyperparameter tuning via GridSearchCV to ensure the model generalizes well to new, unseen data.

## Dataset from Kaggle
**Link:** (https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad/data)


## Install Dependencies
Inside the project directory, install all dependencies using Poetry:

```bash
poetry install
```

## Run the Marimo notebook:

```bash
poetry run marimo edit notebooks/ad_click_prediction.py
```

## Live Report
[Live Interactive Report](https://davitugl.github.io/ad-click-prediction/)